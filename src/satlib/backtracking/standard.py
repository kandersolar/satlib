import numpy as np
from satlib._trigd import cosd, acosd
import satlib


def backtrack_standard(psza, gcr, cross_axis_tilt=0):
    """
    Compute backtracking angles for planar terrain.

    Backtracking angles come from the conventional flat-terrain method [1]_
    or the slope-aware method [2]_.

    Parameters
    ----------
    psza : numeric
        Projected solar zenith angle. [degrees]

    gcr : float
        Ground coverage ratio. [unitless]

    cross_axis_tilt : float, default 0.0
        Cross-axis tilt angle, defined around the same axis as ``psza``. [degrees]

    Returns
    -------
    positions : numeric 
        The conventional backtracking positions, or nan if not
        applicable. [degrees]

    See also
    --------
    pvlib.tracking.calc_cross_axis_tilt

    References
    ----------
    .. [1] Lorenzo, E., Narvarte, L., and Muñoz, J. (2011). Tracking and
       back-tracking 19(6), 747–753. :doi:`10.1002/pip.1085`
    .. [2] Anderson, K., and Mikofski, M., "Slope-Aware Backtracking for
       Single-Axis Trackers", Technical Report NREL/TP-5K00-76626, July 2020.
       :doi:`10.2172/1660126`
    """
    term = np.abs(cosd(psza - cross_axis_tilt)) / (gcr * cosd(cross_axis_tilt))

    mask = np.array(np.abs(term) < 1)
    correction_angle = np.where(mask, acosd(np.array(term), where=mask, out=None), np.nan)
    out = psza - np.sign(psza) * correction_angle

    return satlib._util.fixtype(out)
