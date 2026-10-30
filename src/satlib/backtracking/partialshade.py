import satlib
from satlib._trigd import cosd, acosd

import numpy as np
import warnings


def backtrack_psbt(psza, gcr, allowed_shade_fractions):
    """
    Compute partial-shade backtracking angles using the shaded fraction
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
    theta = np.where(np.abs(psza) < 90, psza, np.nan)
    for fss in sorted(allowed_shade_fractions):
        candidate_tr = satlib.backtracking.backtrack_standard(psza, gcr=gcr*(1-fss))
        candidate_fs = satlib.shading.shaded_fraction(psza, candidate_tr, collector_width=gcr, pitch=1)
        theta = np.where(np.abs(candidate_fs - fss) < 1e-4, candidate_tr, theta)
    return theta


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
