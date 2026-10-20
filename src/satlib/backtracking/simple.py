import numpy as np
from pvlib.tools import cosd, acosd


def backtrack(psza, gcr, cross_axis_tilt=0):
    """
    Compute simple backtracking angles.

    Backtracking angles come from the conventional flat-terrain method [1]_
    or the slope-aware method [2]_.

    Parameters
    ----------
    psza : numeric
        Project solar zenith angle. [degrees]

    gcr : float
        Ground coverage ratio. [unitless]

    cross_axis_tilt : float, default 0.0
        Cross-axis tilt angle. [degrees]

    Returns
    -------
    numeric: the conventional backtracking position, or nan if not
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

    correction_angle = np.nan * psza
    mask = np.abs(term) < 1
    correction_angle[mask] = -np.sign(psza) * acosd(term[mask])

    return psza + correction_angle
