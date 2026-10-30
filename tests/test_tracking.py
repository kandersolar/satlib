import satlib

import numpy as np
import pandas as pd
import pvlib

import pytest


@pytest.fixture
def sp():
    times = pd.date_range("2020-06-01 04:55", "2020-06-01 19:41", freq="5min", tz="Etc/GMT+5")
    sp = pvlib.solarposition.get_solarposition(times, 40, -80, method="ephemeris")
    return sp


def test_astronomical_axis_azimuth(sp):

    solar_noon = sp['apparent_zenith'].idxmin()

    # morning should be negative, afternoon positive, solar noon ~flat
    theta = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180)
    assert theta.values[5] < 0
    assert theta.values[-5] > 0
    assert abs(theta.loc[solar_noon]) < 1

    # morning should be positive, afternoon negative, solar noon ~flat
    theta = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=0)
    assert theta.values[5] > 0
    assert theta.values[-5] < 0
    assert abs(theta.loc[solar_noon]) < 1


def test_astronomical_tilt():
    azimuth = np.array([45, 135, 225, 315])
    theta = satlib.tracking.astronomical(45, azimuth, axis_tilt=90, axis_azimuth=180)
    expected = np.array([-135, -45, 45, 135])
    assert all(np.isclose(theta, expected))


def test_astronomical_aoi_minimization(sp):
    theta = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180)

    orient = satlib.geometry.surface_orientation(theta, axis_tilt=0, axis_azimuth=180)
    aoi = satlib.geometry.aoi(sp.apparent_zenith, sp.azimuth, orient["surface_tilt"], orient["surface_azimuth"])

    orientp1 = satlib.geometry.surface_orientation(theta + 1, axis_tilt=0, axis_azimuth=180)
    aoip1 = satlib.geometry.aoi(sp.apparent_zenith, sp.azimuth, orientp1["surface_tilt"], orientp1["surface_azimuth"])

    orientm1 = satlib.geometry.surface_orientation(theta - 1, axis_tilt=0, axis_azimuth=180)
    aoim1 = satlib.geometry.aoi(sp.apparent_zenith, sp.azimuth, orientm1["surface_tilt"], orientm1["surface_azimuth"])

    assert (aoi < aoip1).all()
    assert (aoi < aoim1).all()


def test_astronomical_daynight(sp):
    # sp is all above the horizon
    out = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180)
    assert out.notnull().all()

    # should be null at night
    out = satlib.tracking.astronomical(100, 50, axis_tilt=0, axis_azimuth=180)
    assert np.isnan(out)


def test_astronomical_types(sp):
    # scalar in, scalar out:
    out = satlib.tracking.astronomical(sp.apparent_zenith.iloc[5], sp.azimuth.iloc[5], axis_tilt=0, axis_azimuth=180)
    assert np.isscalar(out)

    # array in, array out:
    out = satlib.tracking.astronomical(sp.apparent_zenith.values, sp.azimuth.values, axis_tilt=0, axis_azimuth=180)
    assert isinstance(out, np.ndarray)

    # series in, series out:
    out = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180)
    assert isinstance(out, pd.Series)
