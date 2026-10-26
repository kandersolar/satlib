import pandas as pd
import numpy as np


def discretize_time(theta, interval, phase=0, how="center"):
    """
    Apply time-based discretization to a continuous tracker
    position time series.

    Parameters
    ----------
    theta : pandas.Series
        Continuous tracker position time series to be discretized.
        Must have a datetime index.  1-minute resolution is
        recommended.
    interval : float
        Tracker movement interval in minutes.  Must evenly divide
        the time step of ``theta``.
    phase : int
        The "phase angle" of the discretization.  Controls when
        the step change occurs in each interval.  Ranges from zero
        to ``interval``.
    how : {"center", "lag", "lead"}
        Whether the return time series leads, lags, or is centered on
        ``theta``.

    Returns
    -------
    Discretized version of ``theta``
    """

    method = {
        "lead": "bfill",
        "lag": "ffill",
        "center": "nearest",
    }[how]

    interval = pd.Timedelta(interval, unit='min')
    data_step = (theta.index[1] - theta.index[0])
    if interval % data_step != pd.Timedelta(0, unit='min'):
        raise ValueError("Data time step must evenly divide discretization interval") 
    
    values_per_interval = round(interval / data_step)
    idx = theta.index
    subset = theta.iloc[phase::values_per_interval]
    discretized = subset.reindex(idx, method=method)
    return discretized


def discretize_angle(theta, interval, phase=0, how="center"):
    """
    Apply angle-based discretization to a continuous tracker
    position time series.

    Parameters
    ----------
    theta : pandas.Series
        Continuous tracker position time series to be discretized.
        1-minute time resolution is recommended.
    interval : float
        Tracker movement interval in degrees.
    phase : int
        The "phase angle" of the discretization.  Controls when
        the step change occurs in each interval.  Ranges from zero
        to ``interval``.
    how : {"center", "lag", "lead"}
        Whether the return time series leads, lags, or is centered on
        ``theta``.

    Returns
    -------
    Discretized version of ``theta``
    """

    if how == "lead":
        return discretize_angle(theta.iloc[::-1], interval, phase, how="lag").iloc[::-1]

    discretized = np.zeros_like(theta)

    current_level = round((theta.values[0]-phase) / interval) * interval + phase
    if how in ["lag", "center"]:
        threshold = interval/2 if how == "center" else interval
        for i in range(len(theta)):
            if abs(theta.values[i] - current_level) > threshold:
                current_level = round((theta.values[i] - phase) / interval) * interval + phase
            discretized[i] = current_level
    else:
        raise ValueError(f"invalid how: {how}")
    
    return pd.Series(discretized, theta.index)
