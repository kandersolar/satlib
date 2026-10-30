from satlib._util import fixtype

import numpy as np
import pandas as pd


def test_fixtype():

    def pd_out(x):
        return fixtype(pd.Series(x))

    def np_out(x):
        return fixtype(np.array(x))

    def scalar_out(x):
        return fixtype(x.item())

    def same_out(x):
        return x

    s = pd.Series([1,2,3], index=[10,11,12])
    a = np.array([1,2,3])
    x = 5.0

    # pandas in, pandas out
    for func in [pd_out, np_out]:
        fixed = func(s)
        pd.testing.assert_series_equal(fixed, s)
        pd.testing.assert_index_equal(fixed.index, s.index)

    # numpy in, numpy out
    fixed = np_out(a)
    assert isinstance(fixed, np.ndarray)
    assert np.all(fixed == a)

    # scalar in, scalar out
    for func in [np_out, same_out]:
        fixed = func(x)
        assert not isinstance(fixed, np.ndarray)
        assert isinstance(fixed, float)
