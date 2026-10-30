import pandas as pd
import numpy as np
import warnings


def _forward_smear(x, limit):
    diff = x.diff()
    carryover = 0
    for i in range(1, len(diff)):
        desired_step = diff.iloc[i] + carryover
        if desired_step > limit:
            actual_step = limit
        elif desired_step < -limit:
            actual_step = -limit
        else:
            actual_step = desired_step
        diff.iloc[i] = actual_step
        carryover = desired_step - actual_step

    x = x.copy()
    x.iloc[1:] = x.iloc[0] + diff.cumsum().iloc[1:]
    return x


def _backward_smear(x, limit):
    diff = x.diff()
    for i in range(1, len(diff)):
        desired_step = diff.iloc[i]
        if desired_step > limit:
            actual_step = limit
        elif desired_step < -limit:
            actual_step = -limit
        else:
            actual_step = desired_step

        diff.iloc[i] = actual_step
        excess = desired_step - actual_step
        if excess == 0:
            continue
        # else, backtrack and distribute the excess in earlier timestamps
        for j in range(i-1, 0, -1):
            desired_step = excess + diff.iloc[j]
            if desired_step > limit:
                actual_step = limit
            elif desired_step < -limit:
                actual_step = -limit
            else:
                actual_step = desired_step
            diff.iloc[j] = actual_step
            excess = desired_step - actual_step
            if excess == 0:
                break
        else:
            warnings.warn("clip_speed: excess travel could not be fully distributed. "
                          "Result may be unexpected.")

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
    theta : pandas.Series
        The input time series.  Must have datetime index. [degrees]
    limit : float
        Maximum rotation rate. [degrees per minute]
    how : {"lead", "lag", "steeper", "shallower"}
        Which method to apply.

    Returns
    -------
    limited : pandas.Series
        The time series, subject to the specified rotaiton limit.
        [degrees]
    """
    if how == "lag":
        return _forward_smear(theta, limit)
    if how == "lead":
        return _backward_smear(theta, limit)
    if how == "steeper":
        out = theta.copy()
        out[theta < 0] = clip_speed(theta[theta < 0], limit, "lead")
        out[theta > 0] = clip_speed(theta[theta > 0], limit, "lag")
        return out
    if how == "shallower":
        out = theta.copy()
        out[theta < 0] = clip_speed(theta[theta < 0], limit, "lag")
        out[theta > 0] = clip_speed(theta[theta > 0], limit, "lead")
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
