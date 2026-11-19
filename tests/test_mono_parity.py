"""Compare with actual PV-EEM code when core_pv_eem_simulation is installed.

Numerical agreement cases use distinct rounded groups. Mono intentionally reuses
results for nearby inputs within rounded groups; standalone scalar evaluations
need not match that approximation. A separate test preserves that behavior.

From satlib, run with mono's environment:
PYTHONPATH=src ../mono/.venv/bin/python -m pytest tests/test_mono_parity.py -q

Set SATLIB_PYTHON to satlib's absolute Python executable path to also compare
against its separately installed NumPy, pandas, and pvlib versions.
"""

from types import SimpleNamespace
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pvlib
import pytest

pytest.importorskip("core_pv_eem_simulation")

from core_pv_eem_simulation._utils.dimensioned_frame import DimensionedFrame, Dimensions
from core_pv_eem_simulation.config import SimulationConfig
from core_pv_eem_simulation.p1_surface_angles.results import SurfaceAnglesResult
from core_pv_eem_simulation.p1_surface_angles.sub_models.calc_surface_angles import calc_surface_angles
from core_pv_eem_simulation.p1_surface_angles.branches.diffuse_optimization.b_surface_angles_diffuse_optimization import run_surface_angles_diffuse_optimization
from core_pv_eem_simulation.p1_surface_angles.branches.diffuse_optimization.s03_prepare_optimization_inputs import prepare_optimization_inputs
from core_pv_eem_simulation.p1_surface_angles.branches.diffuse_optimization.s04_evaluate_optimization_angles import evaluate_optimization_angles
from core_pv_eem_simulation.p3_epoai.sub_models.direct_shade import ModelCircumsolar
from core_pv_eem_simulation.p3_epoai.sub_models.soiling import ModelSoiling
from core_pv_eem_simulation.p3_epoai.sub_models.diffuse_shade import calculate_tracker_diffuse_shade_fraction

from satlib.plantpredict import optimize_irradiance
from satlib.plantpredict.irradiance import front_irradiance


def independent_satlib_call(function, inputs):
    """Optionally also check satlib's own installed dependency environment."""
    executable = os.environ.get("SATLIB_PYTHON")
    if not executable:
        return None
    script = (
        "import json, sys, numpy as np; "
        "from satlib.plantpredict import optimize_irradiance; "
        "from satlib.plantpredict.irradiance import front_irradiance; "
        "data = json.load(sys.stdin); "
        + ("result = [front_irradiance(**row) for row in data]; "
           if function == "front_irradiance" else
           "result = {k: v.tolist() for k, v in optimize_irradiance(**data).items()}; ")
        + "print(json.dumps(result))"
    )
    return json.loads(subprocess.run(
        [executable, "-c", script], input=json.dumps(inputs), text=True,
        capture_output=True, check=True,
    ).stdout)


def comparison_inputs(gcr, pitch, pile_height, axis_azimuth, circumsolar, count):
    rng = np.random.default_rng(438)
    time = pd.date_range("2026-06-21 06:00", periods=count, freq="5min", tz="UTC")
    weather = pd.DataFrame({
        "time": time, "met_name": "test", "dni": rng.uniform(0, 950, count),
        "dhi": rng.uniform(0, 450, count), "soil_percent": rng.uniform(.8, 1, count),
        "solar_apparent_zenith": rng.uniform(5, 89, count),
        "solar_azimuth": rng.uniform(60, 300, count),
        "dni_extra": rng.uniform(1320, 1420, count),
        "tier": 1, "tier_codes": "test",
    })
    weather["ghi"] = weather.dni * pvlib.tools.cosd(weather.solar_apparent_zenith) + weather.dhi
    weather["air_mass_relative"] = pvlib.atmosphere.get_relative_airmass(weather.solar_apparent_zenith)
    theta = pd.Series(rng.uniform(-65, 65, count))
    theta.iloc[:5] = [0, -60, 60, -.3, .3]
    # Keep distinct rounded groups for standalone-versus-batch checks.
    theta.iloc[5:8] = [35., 36., np.nan]
    weather.loc[5:6, ["solar_apparent_zenith", "solar_azimuth"]] = [75., 100.]
    weather.loc[5:6, ["dni", "dhi", "ghi"]] = [500., 200., 329.40952255126035]
    weather.loc[7, ["solar_apparent_zenith", "dni", "dhi", "ghi"]] = [100., 0., 0., 0.]
    weather["air_mass_relative"] = pvlib.atmosphere.get_relative_airmass(weather.solar_apparent_zenith)
    geometry = calc_surface_angles(
        axis_azimuth=axis_azimuth, tracker_theta=theta,
        solar_apparent_zenith=weather.solar_apparent_zenith,
        solar_azimuth=weather.solar_azimuth,
    ).assign(time=time, met_name="test", string_id=1)
    rotations = DimensionedFrame(Dimensions.STRING_MET_TIME, geometry.loc[:, [
        "string_id", "met_name", "time", "tracker_theta", "surface_tilt", "surface_azimuth", "aoi",
    ]])
    strings = DimensionedFrame(Dimensions.STRING, pd.DataFrame({
        "string_id": [1], "racking_equipment_id": [1], "racking_type": ["tracker"],
        "racking_controls_gcr": [gcr], "pitch": [pitch], "racking_controls_algorithm": [2],
    }))
    racking = DimensionedFrame(Dimensions.RACKING_EQUIPMENT, pd.DataFrame({
        "racking_equipment_id": [1], "pile_height": [pile_height], "max_rotation_angle": [65.],
    }))
    baseline = SurfaceAnglesResult(axis_azimuth, strings, None, rotations)
    horizontal = SimpleNamespace(data=DimensionedFrame(Dimensions.MET_TIME, weather))
    config = SimulationConfig(
        circumsolar=ModelCircumsolar(circumsolar.title()), soiling=ModelSoiling.MEASURED,
        irradiance_optimization_rotation_speed=.1,
    )
    return baseline, horizontal, racking, config, weather


def test_mono_preserves_rounded_diffuse_shading_groups():
    tilts = pd.Series([35.00001, 35.00002])
    grouped = calculate_tracker_diffuse_shade_fraction(surface_tilt=tilts, gcr=.45)
    separate = [calculate_tracker_diffuse_shade_fraction(
        surface_tilt=pd.Series([tilt]), gcr=.45,
    ).iloc[0] for tilt in tilts]
    assert grouped.iloc[0] == separate[0]
    assert grouped.iloc[1] == separate[0]
    assert separate[1] != separate[0]


@pytest.mark.parametrize("circumsolar", ["diffuse", "separate"])
@pytest.mark.parametrize("albedo", [0., .35, .8])
@pytest.mark.parametrize("gcr,pitch,pile_height,axis_azimuth", [
    (.3, 6., 1.5, 180.), (.6, 4., 2., 0.), (.95, 3., 1., 155.),
    (1., 3., 1., 90.),
])
def test_front_irradiance_equivalent(gcr, pitch, pile_height, axis_azimuth, circumsolar, albedo):
    baseline, horizontal, racking, config, weather = comparison_inputs(
        gcr, pitch, pile_height, axis_azimuth, circumsolar, 40,
    )
    groups = prepare_optimization_inputs(
        baseline=baseline, racking_data=racking, horizontal_irradiance=horizontal,
    )
    mono = evaluate_optimization_angles(
        rotation_data=baseline.rotation_data, string_data=baseline.string_data,
        racking_data=racking, fixed_layout_data=None, grouped_rows=groups,
        axis_azimuth=axis_azimuth, time_data=DimensionedFrame(Dimensions.TIME, weather),
        met_time_data=horizontal.data, horizontal_irradiance=horizontal,
        simulation_config=config, albedo=albedo,
    )
    inputs = [dict(
        theta=theta, solar_zenith=row.solar_apparent_zenith, solar_azimuth=row.solar_azimuth,
        dni=row.dni, dhi=row.dhi, ghi=row.ghi, dni_extra=row.dni_extra,
        airmass=row.air_mass_relative, gcr=gcr, axis_azimuth=axis_azimuth,
        albedo=albedo, circumsolar=circumsolar, soiling=row.soil_percent,
        diffuse_factors={}, pitch=pitch, pile_height=pile_height,
    ) for theta, row in zip(baseline.rotation_data.frame.tracker_theta, weather.itertuples())]
    actual = [front_irradiance(**row) for row in inputs]
    np.testing.assert_allclose(actual, mono.epoai_data.frame.front_epoai,
                               rtol=1e-12, atol=1e-9)
    independent = independent_satlib_call("front_irradiance", inputs)
    if independent is not None:
        np.testing.assert_allclose(independent, mono.epoai_data.frame.front_epoai,
                                   rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("circumsolar", ["diffuse", "separate"])
@pytest.mark.parametrize("speed,duration,hesitation", [
    (.01, 60, 0.), (.1, 300, .2), (1., 900, .7),
])
def test_tracker_trajectory_equivalent(circumsolar, speed, duration, hesitation):
    gcr, pitch, pile_height, axis_azimuth = .45, 6., 1.5, 180.
    baseline, horizontal, racking, config, weather = comparison_inputs(
        gcr, pitch, pile_height, axis_azimuth, circumsolar, 16,
    )
    config.irradiance_optimization_rotation_speed = speed
    config.irradiance_optimization_hesitation_factor = hesitation
    # Include a zero-GHI interval that still carries the physical tracker state.
    weather.loc[3, ["dni", "dhi", "ghi"]] = 0
    mono = run_surface_angles_diffuse_optimization(
        baseline=baseline, racking_data=racking,
        time_data=DimensionedFrame(Dimensions.TIME, weather),
        met_time_data=horizontal.data, horizontal_irradiance=horizontal,
        simulation_config=config, albedo=.35, sample_duration_seconds=duration,
    ).optimization_data.frame
    inputs = dict(
        tracker_theta=baseline.rotation_data.frame.tracker_theta.tolist(),
        solar_zenith=weather.solar_apparent_zenith.tolist(), solar_azimuth=weather.solar_azimuth.tolist(),
        dni=weather.dni.tolist(), dhi=weather.dhi.tolist(), ghi=weather.ghi.tolist(),
        dni_extra=weather.dni_extra.tolist(),
        gcr=gcr, pitch=pitch, pile_height=pile_height, axis_azimuth=axis_azimuth,
        albedo=.35, circumsolar=circumsolar, soiling=weather.soil_percent.tolist(),
        rotation_speed=speed, sample_duration=duration, hesitation_factor=hesitation,
    )
    actual = optimize_irradiance(**inputs)
    independent = independent_satlib_call("optimize_irradiance", inputs)
    for ours, theirs in [
        ("ideal_theta", "ideal_tracker_theta"),
        ("tracker_theta", "accepted_tracker_theta"),
        ("end_theta", "accepted_end_tracker_theta"),
        ("irradiance", "accepted_front_epoai"),
        ("baseline_irradiance", "fallback_front_epoai"), ("accepted", "accepted"),
    ]:
        if ours in ("irradiance", "baseline_irradiance"):
            np.testing.assert_allclose(actual[ours], mono[theirs],
                                       rtol=1e-12, atol=1e-9, err_msg=ours)
        else:
            np.testing.assert_array_equal(actual[ours], mono[theirs], err_msg=ours)
        if independent is not None:
            if ours in ("irradiance", "baseline_irradiance"):
                np.testing.assert_allclose(independent[ours], mono[theirs],
                                           rtol=1e-12, atol=1e-9, err_msg=ours)
            else:
                np.testing.assert_array_equal(independent[ours], mono[theirs], err_msg=ours)
