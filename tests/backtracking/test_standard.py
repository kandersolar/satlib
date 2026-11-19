import satlib

import numpy as np
import pandas as pd
import pvlib

import pytest


@pytest.fixture
def theta():
    times = pd.date_range("2020-06-01 04:55", "2020-06-01 19:41", freq="5min", tz="Etc/GMT+5")
    sp = pvlib.solarposition.get_solarposition(times, 40, -80, method="ephemeris")
    theta = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180)
    return theta


def test_backtracking_standard_flat(theta):
    # null in midday, close to zero near dawn/dusk
    out = satlib.backtracking.backtrack_standard(theta, gcr=0.5, cross_axis_tilt=0)

    assert out.iloc[0] < 0
    assert abs(out.iloc[0]) < 1

    assert out.iloc[-1] > 0
    assert abs(out.iloc[-1]) < 1

    n = len(out) // 2
    assert np.isnan(out.iloc[n])


def test_backtracking_standard_sloped(theta):
    # cross-axis tilt -> steeper/shallower tracker positions
    out0 = satlib.backtracking.backtrack_standard(theta, gcr=0.5, cross_axis_tilt=0)
    outp = satlib.backtracking.backtrack_standard(theta, gcr=0.5, cross_axis_tilt=5)
    outn = satlib.backtracking.backtrack_standard(theta, gcr=0.5, cross_axis_tilt=-5)

    # morning
    assert abs(outp.iloc[5]) < abs(out0.iloc[5])
    assert abs(outn.iloc[5]) > abs(out0.iloc[5])
    # evening
    assert abs(outp.iloc[-5]) > abs(out0.iloc[-5])
    assert abs(outn.iloc[-5]) < abs(out0.iloc[-5])


def test_backtracking_standard_gcr(theta):
    # increase gcr -> shallower tracker positions
    out1 = satlib.backtracking.backtrack_standard(theta, gcr=0.4, cross_axis_tilt=0)
    out2 = satlib.backtracking.backtrack_standard(theta, gcr=0.5, cross_axis_tilt=0)

    assert all((abs(out1) > abs(out2)) | np.isnan(out1))


def test_backtracking_standard_types(theta):
    # scalar in, scalar out:
    out = satlib.backtracking.backtrack_standard(theta.iloc[5], gcr=0.5, cross_axis_tilt=0)
    assert np.isscalar(out)

    # array in, array out:
    out = satlib.backtracking.backtrack_standard(theta.values, gcr=0.5, cross_axis_tilt=0)
    assert isinstance(out, np.ndarray)

    # series in, series out:
    out = satlib.backtracking.backtrack_standard(theta, gcr=0.5, cross_axis_tilt=0)
    assert isinstance(out, pd.Series)
