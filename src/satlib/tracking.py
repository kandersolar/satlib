import numpy as np
from satlib._trigd import sind, cosd, atan2d


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
    # Since elevation = 90 - zenith, sin(90-x) = cos(x) & cos(90-x) = sin(x):
    # Notation from [1], modified to use zenith instead of elevation
    # cos(elevation) = sin(zenith) and sin(elevation) = cos(zenith)
    # Avoid recalculating these values
    sind_solar_zenith = sind(solar_zenith)
    cosd_axis_azimuth = cosd(axis_azimuth)
    sind_axis_azimuth = sind(axis_azimuth)
    sind_axis_tilt = sind(axis_tilt)

    # Sun's x, y, z coords
    sx = sind_solar_zenith * sind(solar_azimuth)
    sy = sind_solar_zenith * cosd(solar_azimuth)
    sz = cosd(solar_zenith)
    # Eq. (4); sx', sz' values from sun coordinates projected onto surface
    sx_prime = sx * cosd_axis_azimuth - sy * sind_axis_azimuth
    sz_prime = (
        sx * sind_axis_azimuth * sind_axis_tilt
        + sy * sind_axis_tilt * cosd_axis_azimuth
        + sz * cosd(axis_tilt)
    )
    # Eq. (5); angle between sun's beam and surface
    theta_T = atan2d(sx_prime, sz_prime)
    return theta_T
