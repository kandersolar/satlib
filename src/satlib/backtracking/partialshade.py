import satlib
from satlib._trigd import sind, cosd, acosd, atand

import numpy as np
import warnings


def _f(fs, gcr):
    return acosd((1-fs)*gcr)


def _g(fs, gcr, max_angle):
    num = 1/((1-fs)*gcr) - cosd(max_angle)
    den = sind(max_angle)
    return atand(num / den)


def _psbt(psza, fs, gcr):
    return psza - np.sign(psza) * acosd(cosd(psza) / ((1-fs)*gcr))


def backtrack_psbt(psza, gcr, allowed_shade_fractions, max_angle):
    """
    Compute partial-shade backtracking positions.

    This method is not yet published.

    Parameters
    ----------
    psza : numeric
        Projected solar zenith angle. [degrees]

    gcr : float
        Ground coverage ratio. [unitless]

    allowed_shade_fractions : list of float
        Allowable shade fractions, as a fraction of total collector bandwidth.
        For example, pass ``[0, 0.5]`` for a 1P array and ``[0, 0.25, 0.5, 0.75]``
        for a 2P array.  All elements must be in the half-open interval [0, 1).
        [unitless]

    max_angle : float, optional
        Maximum allowed tracker position. [degrees]

    Returns
    -------
    positions : numeric
        Partial-shade backtracking tracker positions. [degrees]
    """
    _psza = np.asarray(psza)
    x = np.abs(_psza)
    if 0 not in allowed_shade_fractions:
        allowed_shade_fractions = [0] + list(allowed_shade_fractions)

    allowed_shade_fractions = np.array(sorted(allowed_shade_fractions))
    f0 = _f(0, gcr)
    if max_angle is None:
        critical_points = _f(allowed_shade_fractions, gcr)
    else:
        critical_points = _g(allowed_shade_fractions, gcr, max_angle)
        # always use f0 instead of g0 so that segment 3 is included.
        # (it gets clipped at the end)
        critical_points[0] = f0

    out = np.full(_psza.shape, np.nan)

    for i in range(len(critical_points)):
        mask = (critical_points[i] < x)
        if i < len(critical_points)-1:
            mask &= (x < critical_points[i+1])
        out[mask] = _psbt(_psza[mask], allowed_shade_fractions[i], gcr)

    # clip segment 3
    if max_angle is not None:
        mask = (f0 < x) & (x < _g(0, gcr, max_angle))
        out[mask] = out[mask].clip(-max_angle, max_angle)
    return satlib._util.fixtype(out)
