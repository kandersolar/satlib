from satlib._trigd import sind, cosd, acosd, atan2d
import numpy as np
import pandas as pd


def aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """
    Calculates the angle of incidence of the solar vector on a surface.
    This is the angle between the solar vector and the surface normal.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal. [°]
    surface_azimuth : numeric
        Panel azimuth. [°]
    solar_zenith : numeric
        Solar zenith angle. [°]
    solar_azimuth : numeric
        Solar azimuth angle. [°]

    Returns
    -------
    aoi : numeric
        Angle of incidence. [°]
    """

    projection = (
        cosd(surface_tilt) * cosd(solar_zenith) +
        sind(surface_tilt) * sind(solar_zenith) *
        cosd(solar_azimuth - surface_azimuth)
    )

    # pvlib GH 1185
    projection = np.clip(projection, -1, 1)
    
    return acosd(projection)


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


def _unit_normal(axis_azimuth, axis_tilt, theta):
    """
    Unit normal to rotated tracker surface, in global E-N-Up coordinates,
    given by R*(0, 0, 1).T, where:

        R = Rz(-axis_azimuth) Rx(-axis_tilt) Ry(theta)

    Rz is a rotation by -axis_azimuth about the z-axis (axis_azimuth
    is negated to convert from an azimuth angle to a rotation angle). Rx is a
    rotation by -axis_tilt about the x-axis, where axis_tilt is negated
    because pvlib's convention is that the positive y-axis is tilted
    downwards. Ry is a rotation by theta about the y-axis.

    Parameters
    ----------
    axis_azimuth : scalar
    axis_tilt    : scalar
    theta        : scalar or array-like

    Returns
    -------
    ndarray
        Shape ``theta.shape + (3,)``, with a minimum rank of 2.  For 1-D
        ``theta`` of length N this is ``(N, 3)``.
    """

    theta = np.atleast_1d(np.asarray(theta))

    cA, sA = cosd(-axis_azimuth), sind(-axis_azimuth)
    cT, sT = cosd(-axis_tilt), sind(-axis_tilt)

    cTh = cosd(theta)
    sTh = sind(theta)

    x = sA * sT * cTh + cA * sTh
    y = sA * sTh - cA * sT * cTh
    z = cT * cTh

    result = np.stack((x, y, z), axis=-1)

    return result


def surface_orientation(tracker_theta, axis_tilt=0, axis_azimuth=0):
    """
    Calculate the surface tilt and azimuth angles for a given tracker rotation.

    Parameters
    ----------
    tracker_theta : numeric
        Tracker rotation angle as a right-handed rotation around
        the axis defined by ``axis_tilt`` and ``axis_azimuth``.  For example,
        with ``axis_tilt=0`` and ``axis_azimuth=180``, ``tracker_theta > 0``
        results in ``surface_azimuth`` to the West while ``tracker_theta < 0``
        results in ``surface_azimuth`` to the East. [degree]
    axis_tilt : float, default 0
        The tilt of the axis of rotation with respect to horizontal. [degree]
    axis_azimuth : float, default 0
        A value denoting the compass direction along which the axis of
        rotation lies. Measured east of north. [degree]

    Returns
    -------
    dict or DataFrame
        Contains keys ``'surface_tilt'`` and ``'surface_azimuth'`` representing
        the module orientation accounting for tracker rotation and axis
        orientation (degree).
        Where ``surface_tilt``=0, ``surface_azimuth`` is set equal to
        ``axis_azimuth`` - 90.

    References
    ----------
    .. [1] William F. Marion and Aron P. Dobos, "Rotation Angle for the Optimum
       Tracking of One-Axis Trackers", Technical Report NREL/TP-6A20-58891,
       July 2013. :doi:`10.2172/1089596`
    """
    # from [1], Eq. 1
    with np.errstate(invalid='ignore', divide='ignore'):
        surface_tilt = acosd(cosd(tracker_theta) * cosd(axis_tilt))

    # for surface azimuth deviate from [1] to allow for negative tilt.
    # unit normal to rotated tracker surface
    unit_normal = _unit_normal(axis_azimuth, axis_tilt, tracker_theta)

    # project unit_normal to x-y plane to calculate azimuth
    surface_azimuth = np.degrees(
        np.arctan2(unit_normal[..., 0], unit_normal[..., 1]))

    # above always returns a np.array
    # return scalar if `tracker_theta` is a scalar
    if np.isscalar(tracker_theta):
        surface_azimuth = surface_azimuth.item()

    surface_azimuth = np.where(surface_tilt == 0., axis_azimuth - 90.,
                               surface_azimuth)
    # constrain angles to [0, 360)
    surface_azimuth = np.mod(surface_azimuth, 360.0)

    out = {
        'surface_tilt': surface_tilt,
        'surface_azimuth': surface_azimuth,
    }
    if hasattr(tracker_theta, 'index'):
        out = pd.DataFrame(out)
    return out


def axis_tilt(slope_azimuth, slope_tilt, axis_azimuth):
    """
    Calculate tracker axis tilt in the global reference frame when on a sloped
    plane. Axis tilt is the inclination of the tracker rotation axis with
    respect to horizontal, ranging from 0 degrees (horizontal axis) to 90
    degrees (vertical axis).

    Parameters
    ----------
    slope_azimuth : float
        direction of normal to slope on horizontal [degrees]
    slope_tilt : float
        tilt of normal to slope relative to vertical [degrees]
    axis_azimuth : float
        direction of tracker axes on horizontal [degrees]

    Returns
    -------
    axis_tilt : float
        tilt of tracker [degrees]

    See also
    --------
    pvlib.tracking.singleaxis
    pvlib.tracking.calc_cross_axis_tilt

    Notes
    -----
    See [1]_ for derivation of equations.

    References
    ----------
    .. [1] Kevin Anderson and Mark Mikofski, "Slope-Aware Backtracking for
       Single-Axis Trackers", Technical Report NREL/TP-5K00-76626, July 2020.
       :doi:`10.2172/1660126`
    """
    delta_gamma = axis_azimuth - slope_azimuth
    # equations 18-19
    tan_axis_tilt = cosd(delta_gamma) * tand(slope_tilt)
    return np.degrees(np.arctan(tan_axis_tilt))


def _calc_tracker_norm(ba, bg, dg):
    """
    Calculate tracker normal, v, cross product of tracker axis and unit normal,
    N, to the system slope plane.

    Parameters
    ----------
    ba : float
        axis tilt [degrees]
    bg : float
        ground tilt [degrees]
    dg : float
        delta gamma, difference between axis and ground azimuths [degrees]

    Returns
    -------
    vector : tuple
        vx, vy, vz
    """
    cos_ba = cosd(ba)
    cos_bg = cosd(bg)
    sin_bg = sind(bg)
    sin_dg = sind(dg)
    vx = sin_dg * cos_ba * cos_bg
    vy = sind(ba)*sin_bg + cosd(dg)*cos_ba*cos_bg
    vz = -sin_dg*sin_bg*cos_ba
    return vx, vy, vz


def _calc_beta_c(v, dg, ba):
    """
    Calculate the cross-axis tilt angle.

    Parameters
    ----------
    v : tuple
        tracker normal
    dg : float
        delta gamma, difference between axis and ground azimuths [degrees]
    ba : float
        axis tilt [degrees]

    Returns
    -------
    beta_c : float
        cross-axis tilt angle [radians]
    """
    vnorm = np.sqrt(np.dot(v, v))
    beta_c = np.arcsin(
        ((v[0]*cosd(dg) - v[1]*sind(dg)) * sind(ba) + v[2]*cosd(ba)) / vnorm)
    return beta_c


def cross_axis_tilt(slope_azimuth, slope_tilt, axis_azimuth, axis_tilt):
    """
    Calculate the angle, relative to horizontal, of the line formed by the
    intersection between the slope containing the tracker axes and a plane
    perpendicular to the tracker axes.

    Use the cross-axis tilt to avoid row-to-row shade when backtracking on a
    slope not parallel with the axis azimuth. Cross-axis tilt should be
    specified using a right-handed convention. For example, trackers with axis
    azimuth of 180 degrees (heading south) will have a negative cross-axis tilt
    if the tracker axes plane slopes down to the east and positive cross-axis
    tilt if the tracker axes plane slopes down to the west.

    Parameters
    ----------
    slope_azimuth : float
        direction of the normal to the slope containing the tracker axes, when
        projected on the horizontal. [degrees]
    slope_tilt : float
        angle of the slope containing the tracker axes, relative to horizontal.
        [degrees]
    axis_azimuth : float
        direction of tracker axes projected on the horizontal. [degrees]
    axis_tilt : float
        tilt of trackers relative to horizontal. [degree]

    Returns
    -------
    cross_axis_tilt : float
        angle, relative to horizontal, of the line formed by the intersection
        between the slope containing the tracker axes and a plane perpendicular
        to the tracker axes [degrees]

    See also
    --------
    pvlib.tracking.singleaxis
    pvlib.tracking.calc_axis_tilt

    Notes
    -----
    See [1]_ for derivation of equations.

    References
    ----------
    .. [1] Kevin Anderson and Mark Mikofski, "Slope-Aware Backtracking for
       Single-Axis Trackers", Technical Report NREL/TP-5K00-76626, July 2020.
       :doi:`10.2172/1660126`
    """
    # delta-gamma, difference between axis and slope azimuths
    delta_gamma = axis_azimuth - slope_azimuth
    # equation 22
    v = _calc_tracker_norm(axis_tilt, slope_tilt, delta_gamma)
    # equation 26
    beta_c = _calc_beta_c(v, delta_gamma, axis_tilt)
    return np.degrees(beta_c)


def projected_solar_zenith_angle(solar_zenith, solar_azimuth,
                                 axis_tilt, axis_azimuth):
    r"""
    Calculate projected solar zenith angle in degrees.

    This solar zenith angle is projected onto the plane whose normal vector is
    defined by ``axis_tilt`` and ``axis_azimuth``. The normal vector is in the
    direction of ``axis_azimuth`` (clockwise from north) and tilted from
    horizontal by ``axis_tilt``.

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
        North = 0°; East = 90°; South = 180°; West = 270°

    Returns
    -------
    Projected_solar_zenith : numeric
        In degrees.

    Notes
    -----
    This projection has a variety of applications in PV. For example:

    - Projecting the sun's position onto the plane perpendicular to
      the axis of a single-axis tracker (i.e. the plane
      whose normal vector coincides with the tracker torque tube)
      yields the tracker rotation angle that maximizes direct irradiance
      capture. This tracking strategy is called *true-tracking*.

    - Self-shading in large PV arrays is often modeled by assuming
      a simplified 2-D array geometry where the sun's position is
      projected onto the plane perpendicular to the PV rows.
      The projected zenith angle is then used for calculations
      regarding row-to-row shading.

    Examples
    --------
    Calculate the ideal true-tracking angle for a horizontal north-south
    single-axis tracker:

    >>> rotation = projected_solar_zenith_angle(solar_zenith, solar_azimuth,
    >>>                                         axis_tilt=0, axis_azimuth=180)

    Calculate the projected zenith angle in a south-facing fixed tilt array
    (note: the ``axis_azimuth`` of a fixed-tilt row points along the length
    of the row):

    >>> psza = projected_solar_zenith_angle(solar_zenith, solar_azimuth,
    >>>                                     axis_tilt=0, axis_azimuth=90)

    References
    ----------
    .. [1] K. Anderson and M. Mikofski, 'Slope-Aware Backtracking for
       Single-Axis Trackers', National Renewable Energy Laboratory.
       Tech. Report NREL/TP-5K00-76626, Jul. 2020. :doi:`10.2172/1660126`.
    """
    # Assume the tracker reference frame is right-handed. Positive y-axis is
    # oriented along tracking axis; from north, the y-axis is rotated clockwise
    # by the axis azimuth and tilted from horizontal by the axis tilt. The
    # positive x-axis is 90 deg clockwise from the y-axis and parallel to
    # horizontal (e.g., if the y-axis is south, the x-axis is west); the
    # positive z-axis is normal to the x and y axes, pointed upward.

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
