import pandas as pd
import numpy as np


def _smear(x, limit):
    ts = x.index[1] - x.index[0]
    limit = limit * (ts / pd.Timedelta("1min"))
    diff = x.diff()
    carryover = 0
    for i in range(1, len(diff)):
        d = diff.iloc[i]
        d = d + carryover
        if d > limit:
            diff.iloc[i] = limit
            carryover = d - limit
        elif d < -limit:
            diff.iloc[i] = -limit
            carryover = d + limit
        else:
            diff.iloc[i] = d
            carryover = 0
    x = x.copy()
    x.iloc[1:] = x.iloc[0] + diff.cumsum().iloc[1:]
    return x


def _reverse(x):
    return pd.Series(x.values[::-1], index=x.index)


def clip_speed(theta, limit, how):
    """
    Apply rotation speed limits to a tracker position time
    series.

    Where the input time series moves faster than the specified
    limit, the returned time series:

    * "lead": leads the input time series
    * "lag": lags the input time series
    * "steeper": leads and lags such that it is at steeper tilt
      than the input
    * "shallower": leads and lags such that it is at shallower
      tilt than the input

    Parameters
    ----------
    theta : numeric
        The input time series. [degrees]
    limit : float
        Maximum rotation rate. [degrees per minute]
    how : {"lead", "lag", "steeper", "shallower"}
        Which method to apply.

    Returns
    -------
    limited : numeric
        The time series, subject to the specified rotaiton limit.
        [degrees]
    """
    if how == "lag":
        return _smear(theta, limit)
    if how == "lead":
        return _reverse(_smear(_reverse(theta), limit))
    if how == "steeper":
        x1 = theta[theta < 0]
        x2 = theta[theta > 0]
        y1 = _smear(x1, limit)
        y2 = _reverse(_smear(_reverse(x2), limit))
        out = theta.copy()
        out[theta < 0] = y1
        out[theta > 0] = y2
        return out
    if how == "shallower":
        x1 = theta[theta < 0]
        x2 = theta[theta > 0]
        y1 = _reverse(_smear(_reverse(x1), limit))
        y2 = _smear(x2, limit)
        out = theta.copy()
        out[theta < 0] = y1
        out[theta > 0] = y2
        return out
    raise


def clip_position(theta, limit):
    """
    Apply rotation limits to a tracker position time series.

    Parameters
    ----------
    theta : numeric
        The tracker position time series. [degrees]
    limit : float or tuple of numeric
        The max/min position limit.  If a scalar is passed,
        it is used symmetrically for min and max limits.
        Otherwise, a tuple of (min, max) limits must be passed.

    Returns
    -------
    clipped : numeric
        The position time series with limits applied.
    """
    if np.isscalar(limit):
        limit = (-limit, limit)
    return np.clip(theta, limit[0], limit[1])
