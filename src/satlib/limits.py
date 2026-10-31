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


def _find_windows(x):
    """
    Find boundaries of alternating positive and negative
    windows.  Zeros are included in all windows, so the returned
    boundaries may be overlapping.

    Parameters
    ----------
    x : numpy.array

    Returns
    -------
    boundaries : list of tuple
        List of tuples (start, end, sign) of each window.
    """
    boundaries = []
    current_sign = int(np.sign(x[x!=0][0]))
    start = 0  # first window always start at index zero
    zero_run_start = 0  # start of the current run of zeros
    for i in range(1, len(x)):
        if x[i] != 0 and x[i-1] == 0:
            # we just exited a run of zeros
            if np.sign(x[i]) == current_sign:
                # haven't exited the current window yet.
                continue
        if x[i] != 0 and np.sign(x[i]) != current_sign:
            # this window is over; store its info
            # and set up the next one
            boundaries.append((start, i, current_sign))
            if x[i-1] == 0:
                start = zero_run_start
            else:
                start = i
            current_sign *= -1
            continue
        if x[i] == 0 and x[i-1] != 0:
            # we just found some zeros
            zero_run_start = i
    boundaries.append((start, i+1, current_sign))
    return boundaries


def clip_speed(theta, limit, how):
    """
    Apply rotation speed limits to a tracker position time
    series.

    Where the input time series moves faster than the specified
    limit, the returned time series:

    * "lead": leads the input time series
    * "lag": lags the input time series
    * "shallower": leads and lags such that it is at shallower
      tilt than the input

    Parameters
    ----------
    theta : pandas.Series
        The input time series.  Must have datetime index. [degrees]
    limit : float
        Maximum rotation rate. [degrees per minute]
    how : {"lead", "lag", "shallower"}
        Which method to apply.

    Returns
    -------
    limited : pandas.Series
        The time series, subject to the specified rotation limit.
        [degrees]
    """
    if how not in ['lag', 'lead', 'shallower']:
        raise ValueError(f'Invalid "how": "{how}"')

    if how == "lag":
        return _smear(theta, limit)
    if how == "lead":
        return _reverse(_smear(_reverse(theta), limit))

    # - positive value, positive slope: lag
    # - positive value, negative slope: lead
    # - negative value, positive slope: lead
    # - negative value, negative slope: lag

    window_boundaries = _find_windows((theta * theta.diff()).values[1:])
    out = theta.copy()
    for start, end, sign in window_boundaries:
        start = start - 1 if start > 0 else start
        end = end + 1 if end < len(out) else end
        subset = out.iloc[start:end]
        if sign > 0:
            smeared = _smear(subset, limit)
            # did the smearing "catch up" by the end of the window?
            caught_up = abs(smeared.iloc[-1] - out.iloc[end-1]) < 1e-10
            out.iloc[start:end] = smeared
            if not caught_up:
                # need to extend and continue smearing beyond the end
                # TODO smearing the full array is quite wasteful.
                # create some kind of "smear until equal" alternative.
                smeared = _smear(out.iloc[end-1:], limit)
                # find where it finally does catch up (first equal index)
                new_end = np.argmax(smeared.iloc[1:] == out.iloc[end:])
                out.iloc[end:end+new_end] = smeared.iloc[:new_end]

        elif sign < 0:
            smeared = _reverse(_smear(_reverse(subset), limit))
            # did the smearing "catch up" by the end (start) of the window?
            caught_up = abs(smeared.iloc[0] - out.iloc[start]) < 1e-10
            #import pdb; pdb.set_trace()
            out.iloc[start:end] = smeared
            if not caught_up:
                # need to extend and continue smearing beyond the end (start)
                # TODO smearing the full array is quite wasteful.
                # create some kind of "smear until equal" alternative.
                smeared = _reverse(_smear(_reverse(out.iloc[:start+1]), limit))
                # find where it finally does catch up (last equal index)
                new_start = np.argmax((smeared.iloc[:-1] == out.iloc[:start]).iloc[::-1])
                out.iloc[start-new_start:start] = smeared.iloc[-new_start-1:-1]

    return out

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
