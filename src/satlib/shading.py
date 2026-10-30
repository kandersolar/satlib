from satlib._trigd import sind, cosd

import numpy as np


def shaded_fraction(psza, shaded_row_rotation, collector_width,
                    pitch, surface_to_axis_offset=0, cross_axis_slope=0,
                    shading_row_rotation=None):
    r"""
    Shaded fraction in the vertical dimension of tilted rows, or perpendicular
    to the axis of horizontal rows.

    If ``shading_row_rotation`` isn't provided, it is assumed that
    the shaded and shading rows have equal rotation.

    .. warning::
        The function assumes that the roles of the shaded and shading rows
        remain the same during the day. In the case where the shading and
        shaded rows change throughout the day, e.g. a N-S single-axis tracker,
        the inputs must be switched depending on the sign of the projected
        solar zenith angle.

    Parameters
    ----------
    psza : numeric
        Projected solar zenith angle. [degrees]
    shaded_row_rotation : numeric
        Right-handed rotation of the row receiving the shade, with respect
        to ``axis_azimuth``. [degrees]
    collector_width : numeric
        Vertical length of a tilted row. The returned ``shaded_fraction``
        is the ratio of the shadow over this value.
    pitch : numeric
        Axis-to-axis horizontal spacing of the row.
    surface_to_axis_offset : numeric, default 0
        Distance between the rotating axis and the collector surface.
        May be used to account for a torque tube offset.
    cross_axis_slope : numeric, default 0
        Angle of the plane containing the rows' axes from
        horizontal. Right-handed rotation with respect to the rows axes.
        [degrees]
    shading_row_rotation : numeric, optional
        Right-handed rotation of the row casting the shadow, with respect
        to the row axis. [degrees]

    Returns
    -------
    shaded_fraction : numeric
        The fraction of the collector width shaded by an adjacent row. A
        value of 1 is completely shaded and 0 is no shade.

    References
    ----------
    .. [1] K. S. Anderson and A. R. Jensen, "Shaded fraction and backtracking
       in single-axis trackers on rolling terrain," Journal of Renewable and
       Sustainable Energy, vol. 16, no. 2, Mar. 2024, doi:`10.1063/5.0202220`. 
    """
    # For nomenclature you may refer to [1].

    # rotation of row casting the shadow defaults to shaded row's one
    if shading_row_rotation is None:
        shading_row_rotation = shaded_row_rotation

    # calculate repeated elements
    thetas_1_S_diff = shading_row_rotation - psza
    thetas_2_S_diff = shaded_row_rotation - psza
    thetaS_rotation_diff = psza - cross_axis_slope

    cos_theta_2_S_diff_abs = np.abs(cosd(thetas_2_S_diff))

    # Eq. (12) of [1]
    t_star = (
        0.5
        + np.abs(cosd(thetas_1_S_diff)) / cos_theta_2_S_diff_abs / 2
        + (
            np.sign(psza)
            * surface_to_axis_offset
            / collector_width
            / cos_theta_2_S_diff_abs
            * (sind(thetas_2_S_diff) - sind(thetas_1_S_diff))
        )
        - (
            pitch
            / collector_width
            * cosd(thetaS_rotation_diff)
            / cos_theta_2_S_diff_abs
            / cosd(cross_axis_slope)
        )
    )

    return np.clip(t_star, 0, 1)