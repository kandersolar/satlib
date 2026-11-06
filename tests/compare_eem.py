"""Compare with a fresh EEM run on the frozen Double Black Diamond fixture.

Run from satlib using mono's environment (no network or credentials required):
    ../mono/pv-eem/.venv/bin/python tests/compare_eem.py \
        --mono ../mono --output /tmp/satlib-eem-parity

Angles and decisions must match exactly, including missing values. Irradiance
must agree within 1e-10 W/m^2 absolute tolerance; exact equality is also reported.
The external frozen fixture stays in mono rather than being copied into satlib.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satlib.plantpredict import optimize_irradiance


def compare_eem(mono, output):
    from core_pv_eem_simulation.p1_surface_angles.branches.diffuse_optimization import (
        s04_evaluate_optimization_angles as evaluation,
        s05_search_ideal_angles as search,
        s06_accept_corrected_angles as acceptance,
    )

    fixture = mono / "pv-eem/_validation/inverter1-angle-comparison"
    spec = importlib.util.spec_from_file_location(
        "eem_fit_context", fixture / "fit_context.py",
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    context = loader.combined_context()
    grouped = context["ideal_angles"].drop(
        "baseline_front_epoai", "ideal_tracker_theta", "ideal_front_epoai",
    )
    common = {key: context[key] for key in [
        "baseline", "time_data", "met_time_data", "horizontal_irradiance",
        "simulation_config", "albedo",
    ]}

    # Rerun scoring, search, and movement; do not reuse saved selected targets.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="obj.round has no effect with datetime",
            category=UserWarning,
        )
        baseline = evaluation.evaluate_optimization_angles(
            rotation_data=context["baseline"].rotation_data,
            grouped_rows=grouped,
            axis_azimuth=context["baseline"].axis_azimuth,
            **{key: value for key, value in common.items() if key != "baseline"},
        )
        ideal = search.search_ideal_angles(
            grouped_baselines=grouped, baseline_evaluation=baseline, **common,
        )
        reference = acceptance.accept_corrected_angles(
            ideal_angles=ideal, baseline_evaluation=baseline,
            sample_duration_seconds=context["sample_duration_seconds"], **common,
        ).optimization_data.frame

    keys = ["string_id", "met_name", "time"]
    inputs = (
        ideal.to_pandas()[keys + ["tracker_theta", "representative_gcr"]]
        .merge(context["time_data"].frame, on="time", validate="many_to_one")
        .merge(
            context["horizontal_irradiance"].data.frame[
                ["met_name", "time", "dni", "dhi", "ghi"]
            ], on=["met_name", "time"], validate="many_to_one",
        )
        .sort_values(keys).reset_index(drop=True)
    )
    reference = reference.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(inputs[keys], reference[keys])
    config = context["simulation_config"]
    assert config.soiling.value == "none"
    assert config.transposition.value == "Perez_Driesse"
    mapping = {
        "tracker_theta": "accepted_tracker_theta",
        "end_theta": "accepted_end_tracker_theta",
        "ideal_theta": "ideal_tracker_theta",
        "irradiance": "accepted_front_epoai",
        "baseline_irradiance": "fallback_front_epoai",
        "accepted": "accepted",
    }
    comparison = inputs[keys + ["ghi"]].copy()
    for _, group in inputs.groupby(["string_id", "met_name"]):
        assert group.representative_gcr.nunique() == 1
        result = optimize_irradiance(
            group.tracker_theta, group.solar_apparent_zenith, group.solar_azimuth,
            group.dni, group.dhi, group.ghi, group.dni_extra,
            gcr=group.representative_gcr.iloc[0],
            axis_azimuth=context["baseline"].axis_azimuth,
            albedo=context["albedo"],
            circumsolar=config.circumsolar.value.lower(),
            rotation_speed=config.irradiance_optimization_rotation_speed,
            sample_duration=context["sample_duration_seconds"],
            hesitation_factor=config.irradiance_optimization_hesitation_factor,
        )
        for ours, theirs in mapping.items():
            comparison.loc[group.index, f"satlib_{ours}"] = result[ours]
            comparison.loc[group.index, f"eem_{ours}"] = (
                reference.loc[group.index, theirs].to_numpy()
            )

    metrics = {}
    for name in mapping:
        actual = comparison[f"satlib_{name}"].to_numpy(dtype=float)
        expected = comparison[f"eem_{name}"].to_numpy(dtype=float)
        equal = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
        tolerance = 1e-10 if "irradiance" in name else 0
        metrics[name] = {
            "rows": len(actual), "exact_matches": int(equal.sum()),
            "max_absolute_difference": float(np.nanmax(np.abs(actual - expected))),
            "absolute_tolerance": tolerance,
            "passed": bool(np.allclose(
                actual, expected, rtol=0, atol=tolerance, equal_nan=True,
            )),
        }

    from satlib.plantpredict import controller, irradiance, movement
    from satlib.plantpredict import search as satlib_search

    paths = [Path(module.__file__) for module in [
        evaluation, search, acceptance, controller, irradiance, movement,
        satlib_search,
    ]]
    paths.append(fixture / "stateful-movement/previous-fit-contexts.pkl")
    summary = {
        "project": "double_black_diamond",
        "dates": sorted(inputs.time.dt.strftime("%Y-%m-%d").unique()),
        "rows": len(inputs), "pvlib_version": pvlib.__version__,
        "settings": {
            "gcr": inputs.representative_gcr.unique().tolist(),
            "rotation_speed": config.irradiance_optimization_rotation_speed,
            "hesitation_factor": config.irradiance_optimization_hesitation_factor,
            "sample_duration": context["sample_duration_seconds"],
            "albedo": context["albedo"],
            "circumsolar": config.circumsolar.value, "soiling": config.soiling.value,
        },
        "metrics": metrics,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "comparison.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "source_sha256"}, indent=2))
    assert all(value["passed"] for value in metrics.values()), (
        f"EEM parity failed; see {output / 'comparison.csv'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare_eem(args.mono.resolve(), args.output.resolve())
