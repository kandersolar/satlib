from satlib import stow

import pytest
import numpy as np
import pandas as pd


def test_night_stow():
    psza = np.array([-100, 90, -45, 0, 45, 90, 100])
    out = stow.night_stow(psza)
    exp = np.array([0, np.nan, np.nan, np.nan, np.nan, np.nan, 0])
    np.testing.assert_allclose(out, exp)

    out = stow.night_stow(psza, stow_angle=-5)
    exp = np.array([-5, np.nan, np.nan, np.nan, np.nan, np.nan, -5])
    np.testing.assert_allclose(out, exp)


def test_night_stow_types():

    out = stow.night_stow(100, stow_angle=5)
    assert out == 5

    out = stow.night_stow(np.array([100]), stow_angle=5)
    assert isinstance(out, np.ndarray)
    np.testing.assert_equal(out, np.array([5]))

    out = stow.night_stow(pd.Series([100]), stow_angle=5)
    assert isinstance(out, pd.Series)
    pd.testing.assert_series_equal(out, pd.Series([5.0]))


def test_night_stow_modulo():

    psza = np.array([-720, -540, -360, -180, 0, 180, 360, 540, 720])
    out = stow.night_stow(psza)
    exp = np.array([np.nan, 0, np.nan, 0, np.nan, 0, np.nan, 0, np.nan])
    np.testing.assert_allclose(out, exp)
