import satlib

import numpy as np
import pandas as pd
import pvlib

import pytest


@pytest.fixture
def psza():
    psza = np.linspace(-89.9, 89.9, 100)
    times = pd.date_range("2020-06-01 04:55", periods=len(psza), freq="5min", tz="Etc/GMT+5")
    return pd.Series(psza, times)


def _check_psbt(psza, psbt, max_angle, gcr, shade_fractions):
    assert np.nanmax(psbt) == max_angle
    assert np.nanmin(psbt) == -max_angle

    mask = ~np.isnan(psbt)
    fs = satlib.geometry.shaded_fraction(psza[mask], psbt[mask], collector_width=gcr, pitch=1)
    assert np.allclose(sorted(np.unique(np.round(fs, 6))), sorted(shade_fractions))


@pytest.mark.parametrize("how", ["iterative", "piecewise"])
def test_backtracking_psbt(psza, how):

    # null in midday, close to zero near dawn/dusk, all shade fractions achieved
    psbt = satlib.backtracking.backtrack_psbt(psza, gcr=0.5, max_angle=45, allowed_shade_fractions=[0, 0.5], how=how)

    _check_psbt(psza, psbt, 45, 0.5, [0, 0.5])

    assert psbt.iloc[0] < 0
    assert abs(psbt.iloc[0]) < 1

    assert psbt.iloc[-1] > 0
    assert abs(psbt.iloc[-1]) < 1

    n = len(psbt) // 2
    assert np.isnan(psbt.iloc[n])


@pytest.mark.parametrize("how", ["iterative", "piecewise"])
def test_backtracking_psbt_shaded_fractions(psza, how):

    psbt = satlib.backtracking.backtrack_psbt(psza, gcr=0.5, max_angle=45, allowed_shade_fractions=[0, 0.25, 0.5, 0.75], how=how)
    _check_psbt(psza, psbt, 45, 0.5, [0, 0.25, 0.5, 0.75])

    psbt = satlib.backtracking.backtrack_psbt(psza, gcr=0.5, max_angle=45, allowed_shade_fractions=[0, 0.33, 0.66], how=how)
    _check_psbt(psza, psbt, 45, 0.5, [0, 0.33, 0.66])

    # special case:
    psbt = satlib.backtracking.backtrack_psbt(psza, gcr=0.5, max_angle=45, allowed_shade_fractions=[0], how=how)
    bt = satlib.backtracking.backtrack_standard(psza, gcr=0.5).clip(-45, 45)
    mask = ~np.isnan(bt)
    assert np.allclose(psbt[mask], bt[mask])


@pytest.mark.parametrize("how", ["iterative", "piecewise"])
def test_backtracking_psbt_types(how):

    expected = 15.403112980738285

    psbt = satlib.backtracking.backtrack_psbt(85, gcr=0.5, max_angle=45, allowed_shade_fractions=[0, 0.5], how=how)
    assert np.isclose(psbt, expected)
    assert isinstance(psbt, float)

    psbt = satlib.backtracking.backtrack_psbt(np.array([85]), gcr=0.5, max_angle=45, allowed_shade_fractions=[0, 0.5], how=how)
    assert np.allclose(psbt, expected)
    assert isinstance(psbt, np.ndarray)

    psbt = satlib.backtracking.backtrack_psbt(pd.Series([85]), gcr=0.5, max_angle=45, allowed_shade_fractions=[0, 0.5], how=how)
    assert np.allclose(psbt.values, expected)
    assert isinstance(psbt, pd.Series)


@pytest.mark.parametrize("how", ["iterative", "piecewise"])
def test_backtracking_psbt_unconstrained(psza, how):
    gcr = 0.5
    psbt1 = satlib.backtracking.backtrack_psbt(psza, gcr=gcr, max_angle=45, allowed_shade_fractions=[0, 0.5], how=how)
    psbt2 = satlib.backtracking.backtrack_psbt(psza, gcr=gcr, max_angle=None, allowed_shade_fractions=[0, 0.5], how=how)

    fs1 = satlib.geometry.shaded_fraction(psza, psbt1, collector_width=gcr, pitch=1)
    fs2 = satlib.geometry.shaded_fraction(psza, psbt2, collector_width=gcr, pitch=1)

    # unconstrained means it can spend more time at the greater shaded fraction
    assert np.sum(np.isclose(fs1, 0.5)) < np.sum(np.isclose(fs2, 0.5))
