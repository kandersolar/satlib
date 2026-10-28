import satlib


def astronomical(solar_zenith, solar_azimuth, axis_tilt, axis_azimuth):
    """
    Calculate the astronomical ("true-tracking") tracking position.

    This is the tracker position that minimizes the angle of incidence
    of direct irradiance [1]_ [2]_.

    Parameters
    ----------
    solar_zenith : numeric
        Sun's apparent zenith in degrees.
    solar_azimuth : numeric
        Sun's azimuth in degrees.
    axis_tilt : numeric
        Axis tilt angle in degrees. From horizontal plane to array plane.
    axis_azimuth : numeric
        Axis azimuth angle in degrees.

    Returns
    -------
    theta : numeric
        In degrees.

    References
    ----------
    .. [1] Lorenzo, E., Narvarte, L., and Muñoz, J. (2011). Tracking and
       back-tracking 19(6), 747–753. :doi:`10.1002/pip.1085`
    .. [2] Anderson, K., and Mikofski, M., "Slope-Aware Backtracking for
       Single-Axis Trackers", Technical Report NREL/TP-5K00-76626, July 2020.
       :doi:`10.2172/1660126`
    """
    return satlib.geometry.projected_solar_zenith_angle(solar_zenith, solar_azimuth,
                                                        axis_tilt, axis_azimuth)
