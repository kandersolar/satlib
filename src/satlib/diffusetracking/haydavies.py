import numpy as np

from satlib._util import fixtype


def haydavies_analytical(solar_zenith, solar_azimuth, dni, dhi, ghi, albedo,
                         dni_extra, axis_azimuth=180):
    """
    Calculate the irradiance-maximizing tracker position using the Hay-Davies
    model for sky diffuse irradiance.

    This method has not been published.

    Parameters
    ----------
    solar_zenith : numeric
        Solar zenith angle. [°]
    solar_azimuth : numeric
        Solar azimuth angle. [°]
    dni : numeric
        Direct normal irradiance. [W/m^2]
    dhi : numeric
        Diffuse horizontal irradiance. [W/m^2]
    ghi : numeric
        Global horizontal irradiance. [W/m^2]
    albedo : numeric
        Ground albedo. [unitless]
    dni_extra : numeric
        Extraterrestrial direct irradiance. [W/m^2]
    axis_azimuth : float, default 180
        Compass direction along which the tracker axis is oriented.
        Clockwise from north. [°]

    Returns
    -------
    tracker_theta : numeric
        Tracker rotation angle as a right-handed rotation around the axis
        defined by ``axis_azimuth``. Negative when the rotation is clockwise
        and positive when counterclockwise, looking along the tracker axis
        from ``axis_azimuth``. ``tracker_theta = 0`` is horizontal. [°]

    Notes
    -----
    This implementation assumes the tracker axis is horizontal.
    """
    theta_z = np.radians(solar_zenith)
    gamma_s = np.radians(solar_azimuth)
    direction = np.sign((((solar_azimuth - axis_azimuth) + 180) % 360) - 180)
    gamma = np.radians(axis_azimuth + direction * 90)

    A = dni / dni_extra
    C1 = dni + dhi * A / np.cos(theta_z)
    C2 = ghi * albedo - dhi * (1 - A)
    
    num = -direction * np.sin(theta_z) * np.cos(gamma_s - gamma)
    with np.errstate(invalid="ignore"):
        # C1 is zero at night
        den = (C2 / 2 / C1) - np.cos(theta_z)

    theta = np.arctan(num / den)
    return fixtype(np.degrees(theta))
