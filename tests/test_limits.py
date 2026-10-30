import satlib
import numpy as np
import pandas as pd

import pytest


def test_clip_position():
    thetas = np.array([-70, -30, 0, 30, 70])
    out = satlib.limits.clip_position(thetas, limit=50)
    assert np.allclose(out, np.array([-50, -30, 0, 30, 50]))
    out = satlib.limits.clip_position(thetas, limit=(-40, 50))
    assert np.allclose(out, np.array([-40, -30, 0, 30, 50]))
    

def test_clip_position_types():
    thetas = np.array([-70, -30, 0, 30, 70, np.nan])
    expect = np.array([-50, -30, 0, 30, 50, np.nan])

    # np array
    out = satlib.limits.clip_position(thetas, 50)
    assert isinstance(out, np.ndarray)
    assert all(np.isclose(out, expect) | np.isnan(expect))

    # pd series
    out = satlib.limits.clip_position(pd.Series(thetas), 50)
    pd.testing.assert_series_equal(out, pd.Series(expect))

    # scalar
    out = satlib.limits.clip_position(-70., 50)
    assert isinstance(out, float)
    assert out == -50


@pytest.fixture
def toy_thetas():
    vals = [-12, -11, -10, -2, -1, 0, 1, 2, 10, 11, 12]
    times = pd.date_range("2020-06-01 12:00", freq="1min", periods=len(vals))
    thetas = pd.Series(vals, index=times)
    return thetas


@pytest.mark.parametrize("how,expected",[
    ("lead", [-12, -11, -7, -2, -1, 0, 1, 5, 10, 11, 12]),
    ("lag", [-12, -11, -10, -5, -1, 0, 1, 2, 7, 11, 12]),
    ("shallower", [-12, -11, -7, -2, -1, 0, 1, 2, 7, 11, 12]),
])
def test_clip_speed(toy_thetas, how, expected):
    expected = pd.Series(expected, index=toy_thetas.index)
    clipped = satlib.limits.clip_speed(toy_thetas, 5, how=how)
    pd.testing.assert_series_equal(clipped, expected)


@pytest.fixture
def discretized():
    psza = np.arange(-90, 90, 0.1)
    times = pd.date_range("2020-06-01 12:00", freq="1min", periods=len(psza))
    psza = pd.Series(psza, index=times)
    discretized = satlib.discretization.discretize_angle(psza, interval=5)
    return discretized


@pytest.mark.parametrize("how",["lead", "lag", "shallower"])
def test_clip_speed_discrete(discretized, how):
    limit = 1
    clipped = satlib.limits.clip_speed(discretized, limit=limit, how=how)
    speed = clipped.diff()  # index is 1-min, so this is deg/min
    assert speed.abs().max() <= limit


def test_clip_speed_shallower():
    # shallower is more complicated than lead/lag;
    # test a tracker curve that hits speed limits in all four quadrants of
    # the (tracker angle, tracker velocity) plane.
    n = 500
    theta = pd.Series(25 * np.sin(np.linspace(0, 4*np.pi, n)), index=pd.date_range("2020-06-01 12:00", freq="6s", periods=n))
    theta = np.clip(theta, -20, 20)
    limit = 5

    clipped = satlib.limits.clip_speed(theta, limit=limit, how="shallower")
    assert (clipped.abs() <= theta.abs() + 1e-6).all()
    speed = clipped.diff() / (clipped.index.diff() / pd.Timedelta("1min"))
    assert speed.abs().max() <= (limit + 1e-6)


def test_clip_speed_nans():
    pass


def test_clip_speed_invalid(toy_thetas):
    with pytest.raises(ValueError, match="Invalid"):
        satlib.limits.clip_speed(toy_thetas, limit=5, how="not a method")


def test__find_windows():
    #    0  1  2  3   4   5  6  7  8   9  10
    x = [1, 2, 0, 1, -1, -1, 0, 0, 1, -1, 1]
    boundaries = satlib.limits._find_windows(np.array(x))
    assert boundaries == [(0, 4, 1), (4, 8, -1), (6, 9, 1), (9, 10, -1), (10, 11, 1)]

    #    0  1  2  3  4  5  6   7   8  9  10
    x = [0, 0, 1, 1, 1, 0, 0, -1, -1, 0, 0]
    boundaries = satlib.limits._find_windows(np.array(x))
    assert boundaries == [(0, 7, 1), (5, 11, -1)]
