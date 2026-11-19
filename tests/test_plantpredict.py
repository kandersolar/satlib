import numpy as np
import pvlib
import pytest
from scipy.integrate import quad

from satlib.plantpredict import optimize_irradiance
from satlib.plantpredict.irradiance import front_irradiance


@pytest.mark.parametrize("theta", [0, -30, 30, -60, 60])
@pytest.mark.parametrize("gcr", [0.3, 0.6, 0.95])
@pytest.mark.parametrize(
    "component,circumsolar",
    [("isotropic", "diffuse"), ("horizon", "diffuse"),
     ("circumsolar", "diffuse"), ("circumsolar", "separate")],
)
def test_front_sky_shading(monkeypatch, theta, gcr, component, circumsolar):
    """Compare shaded POA with sky exposure integrated across the collector."""
    tilt = abs(theta)
    unshaded_view = (1 + np.cos(np.radians(tilt))) / 2
    dhi = 100
    components = dict.fromkeys(
        ["poa_isotropic", "poa_horizon", "poa_circumsolar"], 0.0,
    )
    components[f"poa_{component}"] = dhi * unshaded_view
    monkeypatch.setattr(pvlib.irradiance, "perez_driesse",
                        lambda *args, **kwargs: components)
    monkeypatch.setattr(pvlib.irradiance, "beam_component",
                        lambda *args, **kwargs: 0)
    monkeypatch.setattr(pvlib.bifacial.infinite_sheds, "get_irradiance_poa",
                        lambda *args, **kwargs: {"poa_ground_diffuse": 0})
    monkeypatch.setattr(pvlib.shading, "shaded_fraction1d",
                        lambda *args, **kwargs: 0.25)
    monkeypatch.setattr(pvlib.iam, "physical", lambda *args, **kwargs: 0.8)
    monkeypatch.setattr(
        pvlib.iam, "marion_diffuse",
        lambda *args, **kwargs: {"sky": 1, "horizon": 1, "ground": 1},
    )

    actual = front_irradiance(
        theta, solar_zenith=45, solar_azimuth=135, dni=0, dhi=dhi,
        ghi=dhi, dni_extra=1367, airmass=1.4, gcr=gcr, axis_azimuth=180,
        albedo=0, circumsolar=circumsolar, soiling=1,
        diffuse_factors={}, pitch=6, pile_height=1.5,
    )
    if circumsolar == "separate":
        expected = dhi * unshaded_view * 0.75 * 0.8
    else:
        average_view, _ = quad(
            lambda x: pvlib.bifacial.utils.vf_row_sky_2d(tilt, gcr, x),
            0, 1,
        )
        expected = dhi * average_view
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("circumsolar", ["diffuse", "separate"])
def test_optimization_preserves_baseline_irradiance(circumsolar):
    result = optimize_irradiance(
        [-60, -30, 0, 30, 60], solar_zenith=[75, 50, 30, 50, 75],
        solar_azimuth=[100, 130, 180, 230, 260], dni=200, dhi=300,
        ghi=400, dni_extra=1367, gcr=0.6, pitch=6, pile_height=1.5,
        circumsolar=circumsolar,
    )
    assert np.isfinite(result["irradiance"]).all()
    assert (result["irradiance"] >= result["baseline_irradiance"]).all()
