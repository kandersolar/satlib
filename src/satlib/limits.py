import pandas as pd
import numpy as np


def tracker_movement(start_theta, target_theta, rotation_speed,
                     sample_duration, hesitation_factor=0.2):
    """Calculate average rotation and physical endpoint during one interval.

    Parameters
    ----------
    start_theta, target_theta : numeric
        Starting and requested tracker rotations. [degrees]
    rotation_speed : float
        Positive tracker rotation speed. [degrees per second]
    sample_duration : float
        Positive interval duration. [seconds]
    hesitation_factor : float, default 0.2
        Fraction of the interval spent waiting before moving, between zero
        and one. Waiting is shortened when necessary to reach the target.

    Returns
    -------
    average_theta, end_theta : numeric
        Average rotation during the interval and physical rotation at its end.

    Notes
    -----
    Assumes constant speed while moving. When the target cannot be reached,
    movement starts immediately and continues throughout the interval.
    The next interval must start at ``end_theta``, not ``average_theta``.
    """
    movement = np.asarray(target_theta) - np.asarray(start_theta)
    travel = rotation_speed * sample_duration
    fraction = np.abs(movement) / travel
    hesitation = np.minimum(hesitation_factor, np.maximum(0, 1 - fraction))
    average = np.where(
        fraction <= 1,
        start_theta + (1 - fraction / 2 - hesitation) * movement,
        start_theta + np.sign(movement) * travel / 2,
    )
    endpoint = start_theta + np.sign(movement) * np.minimum(np.abs(movement), travel)
    return average, endpoint


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
