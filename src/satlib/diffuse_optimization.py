import numpy as np

def munoz2024_analytical(solar_zenith, solar_azimuth, dni, dhi, ghi, albedo):
    # not sure how surface_azimuth is supposed to be known.
    # assume north-south axis and choose east/west based on sun position?
    surface_azimuth = np.where(solar_azimuth < 180, 90, 270)
    
    solar_zenith = np.radians(solar_zenith)
    solar_azimuth = np.radians(solar_azimuth)
    surface_azimuth = np.radians(surface_azimuth)

    numerator = dni * np.sin(solar_zenith) * np.cos(solar_azimuth - surface_azimuth)
    denominator = (dhi - ghi*albedo)/2 + dni * np.cos(solar_zenith)
    beta_opt = np.arctan(numerator / denominator)
    return np.degrees(beta_opt)




def kellygibson2009_tracking1(tracker_theta,ghi, ghi_clear,threshold=0.3):
    """
    Implement the "enhanced tracking system number 1" from Kelly & Gibson (2009).

    Note: [1]_ does not provide a threshold value, so a default clear-sky threshold of 0.3 is assumed.

    When the ratio of GHI to clear-sky GHI is less than a threshold, set the tracker to horizontal[1]_:

    if ``GHI < GHI_clear * threshold``:
    ``tracker_set = 0``
    if ``GHI >= GHI_clear`` * threshold:
    ``tracker_set = tracker_theta_ref`` (reference tracking angle)

    Parameters
    ----------
    tracker_theta : numeric
        Reference tracker rotation angle [degrees]
    ghi : numeric
        Global horizontal irradiance. [W/m^2]
    ghi_clear : numeric
        Clear-sky global horizontal irradiance. [W/m^2]
    threshold : float, default 0.3
        Clear-sky index (``ghi / ghi_clear``) threshold. When clear-sky index
        is below this value, set tracker flat. [unitless]

    Returns
    -------
    tracker_angle_set : numeric
        Tracker rotation angle after applying the algorithm. [degrees]

    References
    ----------
    .. [1] N. A. Kelly and T. L. Gibson, "Improved photovoltaic energy output
       for cloudy conditions with a solar tracking system," Solar Energy,
       vol. 83, no. 11, pp. 2092-2102, Nov. 2009.
       :doi:`10.1016/j.solener.2009.08.009`
    """
    flat_tracking_flag = ghi < ghi_clear*threshold
    tracker_angle_set = np.where(flat_tracking_flag, 0, tracker_theta)

    return tracker_angle_set


def kellygibson2009_tracking2(tracker_theta,ghi, poa_lat,threshold=1.3):
    """
    Implement the "enhanced tracking system number 2" from Kelly & Gibson (2009).

    Assume a horizontally mounted and a latitude-tilt, equator-facing reference cell.
    When the horizontal measurement (H, or GHI) exceeds the latitude-tilt measurement (L) 
    multiplied by a threshold, set the tracker to horizontal [1]_:
    
    if ``GHI > L * threshold``:
    ``tracker_set = 0``


    if ``GHI <= L * threshold``:
    ``tracker_set = tracker_theta_ref`` (reference tracking angle)

    Parameters
    ----------
    tracker_theta : numeric
        Reference tracker rotation angle. [degrees]
    ghi : numeric
        Global horizontal irradiance. [W/m^2]
    poa_lat : numeric
        Irradiance measured by a latitude-tilt, equator-facing reference
        cell (L). [W/m^2]
    threshold : float, default 1.3
        Ratio of ``ghi`` to ``poa_lat`` above which the tracker is set
        flat. [unitless]
 
    Returns
    -------
    tracker_angle_set : numeric
        Tracker rotation angle after applying the algorithm. [degrees]

    References
    ----------
    .. [1] N. A. Kelly and T. L. Gibson, "Improved photovoltaic energy output
       for cloudy conditions with a solar tracking system," Solar Energy,
       vol. 83, no. 11, pp. 2092-2102, Nov. 2009.
       :doi:`10.1016/j.solener.2009.08.009`
    """

    # If GHI > G_poa * theshold, rotate to 0 degrees (horizontal)
    # If GHI < G_poa * threshold, follow standard tracking algorithm
    flat_tracking_flag = ghi > poa_lat*threshold

    # take the reference tracking angle if GHI < GPOA_lat*threshold, and 0 if GHI > GPOA_lat*threshold
    tracker_angle_set = np.where(flat_tracking_flag, 0, tracker_theta)

    return tracker_angle_set

def kellygibson2009_tracking3(tracker_theta,dni,threshold=10):
    """
    Implement the "enhanced tracking system number 3" from Kelly & Gibson (2009).
    

    Note: [1]_ lists only "near-zero" DNI, so a default threshold of 10 W/m2 is applied.
    [1]_ calls for combining a GHI and DHI measurement to calculate DNI from the closure equation; 
    here, we assume that DNI is available.

    When DNI is near-zero, set the tracker to horizontal [1]_:
    
    ``DNI < threshold``:
    ``tracker_set = 0``
    
    ``DNI >= threshold``:
    ``tracker_set = tracker_theta_ref`` (reference tracking angle)

    Parameters
    ----------
    tracker_theta : numeric
        Reference tracker rotation angle. [degrees]
    dni : numeric
        Direct normal irradiance. [W/m^2]
    threshold : float, default 10
        DNI below which the tracker is set flat. [W/m^2]

    Returns
    -------
    tracker_angle_set : numeric
        Tracker rotation angle after applying the algorithm. [degrees]

    References
    ----------
    .. [1] N. A. Kelly and T. L. Gibson, “Improved photovoltaic energy output 
    for cloudy conditions with a solar tracking system,” Sol. Energy, 
    vol. 83, no. 11, pp. 2092-2102, Nov. 2009, doi: 10.1016/j.solener.2009.08.009.
    """

    # If DNI < threshold, rotate to 0 degrees (horizontal)
    # If DNI >= threshold, follow standard tracking algorithm
    flat_tracking_flag = dni < threshold

    # take the reference tracking angle if GHI < GPOA_lat*threshold, and 0 if GHI > GPOA_lat*threshold
    tracker_angle_set = np.where(flat_tracking_flag, 0, tracker_theta)

    return tracker_angle_set