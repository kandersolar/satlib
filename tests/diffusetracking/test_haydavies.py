import satlib

import numpy as np
import pandas as pd
import pvlib

import pytest


@pytest.fixture
def inputs():
    times = pd.date_range("2019-06-01", "2019-06-02", freq="15min", tz="UTC")
    loc = pvlib.location.Location(40, 10)
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, solar_position=sp)
    dni_et = pvlib.irradiance.get_extra_radiation(times)
    return dict(
        solar_zenith=sp.apparent_zenith,
        solar_azimuth=sp.azimuth,
        dni=cs.dni,
        ghi=cs.ghi,
        dhi=cs.dhi,
        dni_extra=dni_et,
        albedo=0.2,
        axis_azimuth=180
    )


def test_haydavies_analytical_irradiance(inputs):
    # test that this method gets higher irradiance relative to astronomical tracking
    axis_azimuth = inputs.pop("axis_azimuth")
    psza = satlib.geometry.projected_solar_zenith_angle(inputs['solar_zenith'], inputs['solar_azimuth'],
                                                        axis_tilt=0, axis_azimuth=axis_azimuth)
    orient = satlib.geometry.surface_orientation(psza, axis_tilt=0, axis_azimuth=axis_azimuth)
    g_astronomical = pvlib.irradiance.get_total_irradiance(**orient, **inputs, model="haydavies")['poa_global']

    theta = satlib.diffusetracking.haydavies_analytical(**inputs, axis_azimuth=axis_azimuth)
    orient = satlib.geometry.surface_orientation(theta, axis_tilt=0, axis_azimuth=axis_azimuth)
    g_analytical = pvlib.irradiance.get_total_irradiance(**orient, **inputs, model="haydavies")['poa_global']

    # nan for nighttime
    assert np.all((g_analytical > g_astronomical) | np.isnan(g_analytical))


def _bruteforce(ghi, dni, dhi, solar_zenith, solar_azimuth, albedo, dni_extra, axis_azimuth):
    opt = 0 * ghi
    theta = np.linspace(-90, +90, 2000)  # 0.09 degree precision
    tilt = np.abs(theta)
    az = np.where(theta > 0, 270, 90)  # assumes axis_azimuth = 180
    for i in range(len(ghi)):
        g = pvlib.irradiance.get_total_irradiance(tilt, az, solar_zenith.values[i], solar_azimuth.values[i],
                                                  dni.values[i], ghi.values[i], dhi.values[i],
                                                  dni_extra.values[i], albedo=albedo, model="haydavies")
        opt.iloc[i] = theta[np.argmax(g['poa_global'])]
    return opt


def test_haydavies_analytical_bruteforce(inputs):
    # test that output matches brute force search
    expected = _bruteforce(**inputs)
    actual = satlib.diffusetracking.haydavies_analytical(**inputs)
    assert np.max(np.abs(actual - expected)) < 0.09

    # check that all nan values are when sun is below horizon
    zenith = inputs['solar_zenith']
    assert np.all(zenith[np.isnan(actual)] > 90)
    # and none are nan when sun is above horizon
    assert not np.any(zenith[np.isnan(actual)] < 90)


def test_haydavies_analytical_types(inputs):

    # series
    out = satlib.diffusetracking.haydavies_analytical(**inputs)
    assert isinstance(out, pd.Series)

    # scalar
    inputs_scalar = {k: v if np.isscalar(v) else v.values[1] for k, v in inputs.items()}
    out = satlib.diffusetracking.haydavies_analytical(**inputs_scalar)
    assert isinstance(out, float)

    # array
    inputs_array = {k: v if np.isscalar(v) else v.values for k, v in inputs.items()}
    out = satlib.diffusetracking.haydavies_analytical(**inputs_array)
    assert isinstance(out, np.ndarray)
