from satlib._trigd import cosd, acosd

def critical_angles(gcr, cross_axis_tilt=0):
    """
    Compute the boundary angles between backtracking and
    true-tracking.

    The critical angles are the PSZA values when backtracking
    begin and end, given by Equations 20 and 21 in Ref [1]_.

    Parameters
    ----------
    gcr : numeric
        Ground coverage ratio. [unitless]
    cross_axis_tilt : numeric, default 0
        Cross axis slope angle. [degrees]

    Returns
    -------
    critical_angle : numeric
        The negative and positive critical angles. [degrees]

    References
    ----------
    .. [1] K. S. Anderson, A. R. Jensen, and D. M. Riley,
       "A Linear Programming Approach to Backtracking for
       Single-Axis Trackers on Rolling Terrain,"
       IEEE J. Photovoltaics, vol. 16, no. 2, pp. 291–298,
       Mar. 2026, :doi:`10.1109/jphotov.2025.3645781`. 
    """
    theta_min = cross_axis_tilt + acosd(gcr * cosd(cross_axis_tilt))
    theta_max = cross_axis_tilt - acosd(gcr * cosd(cross_axis_tilt))
    return theta_min, theta_max
