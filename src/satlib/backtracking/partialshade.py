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


def backtrack_psbt(psza, gcr, max_angle, allowed_shade_fractions):
    """
    Compute partial-shade backtracking positions.

    This method is not yet published.

    Parameters
    ----------
    psza : numeric
        Projected solar zenith angle. [degrees]

    gcr : float
        Ground coverage ratio. [unitless]

    max_angle : float
        Maximum allowed tracker position. [degrees]

    allowed_shade_fractions : list of float
        Allowable shade fractions, as a fraction of total collector bandwidth.
        For example, pass ``[0, 0.5]`` for a 1P array and ``[0, 0.25, 0.5, 0.75]``
        for a 2P array.  All values must be in the half-open interval [0, 1).
        [unitless]

    Returns
    -------
    positions : numeric
        Partial-shade backtracking tracker positions. [degrees]
    """
    x = np.abs(psza)
    if 0 not in allowed_shade_fractions:
        allowed_shade_fractions = [0] + list(allowed_shade_fractions)

    allowed_shade_fractions = np.array(sorted(allowed_shade_fractions))
    g = _g(allowed_shade_fractions, gcr, max_angle)

    out = psza.copy()
    mask = (max_angle < x) & (x < g[0])
    out[mask] = np.sign(psza[mask]) * max_angle

    for i in range(len(g)):
        mask = (g[i] < x)
        if i < len(g)-1:
            mask &= (x < g[i+1])
        out[mask] = _psbt(psza[mask], allowed_shade_fractions[i], gcr)

    return out


def backtrack_psbt_effective_gcr(psza, gcr, allowed_shade_fractions):
    """
    Compute partial-shade backtracking angles using the effective GCR
    method.

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
        for a 2P array. [unitless]

    Returns
    -------
    positions : numeric
        Partial-shade backtracking tracker positions. [degrees]
    """
    candidate_gcrs = (1 - np.array(allowed_shade_fractions)) * gcr
    critical_angles = np.degrees(np.arccos(candidate_gcrs))
    psza = np.array(psza)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        mask = (
            (critical_angles < np.abs(psza)) &
            (np.abs(psza - np.sign(psza) * acosd(cosd(psza)/candidate_gcrs)) < 60)
        )
    masked_gcr = np.where(mask, candidate_gcrs, np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        gcr_eff = np.nanmin(masked_gcr, axis=1)

    out = satlib.backtracking.backtrack_standard(psza, gcr_eff)
    return out
