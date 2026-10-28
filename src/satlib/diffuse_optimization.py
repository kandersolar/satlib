import numpy as np
import pvlib
import satlib

def munoz2024_analytical(solar_zenith, solar_azimuth, dni, dhi, ghi, albedo,axis_azimuth=180):
    """
    Calculate the tracker rotation angle using the CENER analytical model.
 
    Implements eq. 12 of [1]_. The optimal rotation angle of a horizontal
    single-axis tracker is found by taking the derivative of the total irradiance
    with respect to the tracker angle and setting it to zero.

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


    References
    ----------
    .. [1] I. Muñoz, A. Guinda, G. Olivares, S. Díaz, A. M. Gracia-Amillo, and
       L. Casajús, "Evaluation of Horizontal Single-Axis Solar Tracker
       Algorithms in Terms of Energy Production and Operational Performance,"
       Solar RRL, vol. 8, no. 1, p. 2300507, 2024.
       :doi:`10.1002/solr.202300507`
    """

    surface_azimuth, axis_flag = satlib.convertSATangles.get_surface_azimuth(solar_azimuth, axis_azimuth)
    
    solar_zenith = np.radians(solar_zenith)
    solar_azimuth = np.radians(solar_azimuth)
    surface_azimuth = np.radians(surface_azimuth)

    numerator = dni * np.sin(solar_zenith) * np.cos(solar_azimuth - surface_azimuth)
    denominator = (dhi - ghi*albedo)/2 + dni * np.cos(solar_zenith)
    beta_opt = np.arctan(numerator / denominator)

    #return tracker theta:  negative when clockwise looking along TT from axis azimuth
    #                       positive when counterclockwise looking along TT from axis azimuth  
    beta_opt = np.where(axis_flag, -1*beta_opt, beta_opt)

    return np.degrees(beta_opt)

def munoz2024_diffuse(tracker_theta,solar_zenith,solar_azimuth, dni_extra, dni, dhi, ghi, albedo, axis_tilt=0, axis_azimuth=180):
    """
    Calculate HSAT tracker angle using the CENER diffuse model, section 2.1.1 of [1]_.

    When GHI > POA, ``tracker_theta = 0``

    When GHI < POA, ``tracker_theta = tracker_theta_ref`` 

    To calculate POA at the reference tracker angle:
    ``G_{poa} = G_{sky\,diffuse} + G_{direct} + G_{ground}``

    Find diffuse irradiance contribution from Perez model.

    Find ground-reflected irradiance assuming Lambertian ground reflection.

    Parameters
    ----------
    tracker_theta : numeric
        Reference tracker rotation angle, [°]
    solar_zenith : numeric
        Solar zenith angle. [°]
    solar_azimuth : numeric
        Solar azimuth angle. [°]
    dni_extra : numeric
        Extraterrestrial normal irradiance. [W/m^2]
    dni : numeric
        Direct normal irradiance. [W/m^2]
    dhi : numeric
        Diffuse horizontal irradiance. [W/m^2]
    ghi : numeric
        Global horizontal irradiance. [W/m^2]
    albedo : numeric
        Ground albedo. [unitless]
    axis_tilt : float, default 0
        Tilt of the tracker axis from horizontal. [°]
    axis_azimuth : float, default 180
        Compass direction along which the tracker axis is oriented.
        Clockwise from north. [°]

    Returns
    -------
    tracker_theta : numeric
        New tracker rotation angle. [°]

    References
    ----------
    .. [1] I. Muñoz, A. Guinda, G. Olivares, S. Díaz, A. M. Gracia-Amillo, and
       L. Casajús, "Evaluation of Horizontal Single-Axis Solar Tracker
       Algorithms in Terms of Energy Production and Operational Performance,"
       Solar RRL, vol. 8, no. 1, p. 2300507, 2024.
       :doi:`10.1002/solr.202300507`
    """

    #Compare G_poa with GHI. If GHI > G_poa, set tracker angle to 0 (flat). If GHI < G_poa, set tracker angle to reference tracking angle.
    
    # get surface orientation
    surface_orientation = pvlib.tracking.calc_surface_orientation(tracker_theta, axis_tilt = axis_tilt, axis_azimuth=axis_azimuth)

    # get air mass from solar zenith angle
    airmass = pvlib.atmosphere.get_relative_airmass(solar_zenith)

    # get diffuse irradiance contribution using perez model
    poa_sky_diffuse = pvlib.irradiance.perez(surface_orientation['surface_tilt'],surface_orientation['surface_azimuth'],\
                                dhi, dni, dni_extra,solar_zenith, solar_azimuth,airmass)
        
    #get direct and ground-reflected irradiance contributions
    poa_direct = pvlib.irradiance.beam_component(surface_orientation['surface_tilt'], surface_orientation['surface_azimuth'],\
                                                solar_zenith,solar_azimuth,dni)
    poa_ground = pvlib.irradiance.get_ground_diffuse(surface_orientation['surface_tilt'], ghi, albedo)

    # take sum of three irradiance components to get G_poa
    poa_total = poa_sky_diffuse + poa_direct + poa_ground

    # get Munoz2024 diffuse tracker angle from reference tracking angle and GHI vs GPOA comparison 
    tracker_angle_set = munoz2024_diffusetrackerangle(tracker_theta,poa_total,ghi)

    return tracker_angle_set

def munoz2024_diffusetrackerangle(tracker_theta,POA,GHI):
    """
    Calculate tracker angle according to CENER diffuse algorithm, section 2.1.1 of [1]_.

    Set the tracker to 0° when GHI > POA.
    Otherwise, set the tracker to the reference tracking angle.
  
    Parameters
    ----------
    tracker_theta : numeric
        Reference tracker rotation angle. [°]
    POA : numeric
        Plane-of-array irradiance at the reference tracker angle. [W/m^2]
    GHI : numeric
        Global horizontal irradiance. [W/m^2]
 
    Returns
    -------
    tracker_angle_set : numeric
        New tracker rotation angle. [°]
 
    References
    ----------
    .. [1] I. Muñoz, A. Guinda, G. Olivares, S. Díaz, A. M. Gracia-Amillo, and
       L. Casajús, "Evaluation of Horizontal Single-Axis Solar Tracker
       Algorithms in Terms of Energy Production and Operational Performance,"
       Solar RRL, vol. 8, no. 1, p. 2300507, 2024.
       :doi:`10.1002/solr.202300507`
    """

    # Find timestamps where GHI > GPOA
    flat_tracking_flag = GHI > POA

    # take the reference tracking angle if GHI < GPOA, and 0 if GHI > GPOA
    tracker_angle_set = np.where(flat_tracking_flag, 0, tracker_theta)

    return tracker_angle_set