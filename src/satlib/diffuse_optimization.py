import numpy as np
import pvlib

def munoz2024_analytical(solar_zenith, solar_azimuth, dni, dhi, ghi, albedo,axis_azimuth=180):
# CENER "analytical model"
# eq. 12 in I. Muñoz, A. Guinda, G. Olivares, S. Díaz, A. M. Gracia-Amillo, and L. Casajús, 
# “Evaluation of Horizontal Single-Axis Solar Tracker Algorithms in Terms of Energy Production 
# and Operational Performance,” Solar RRL, vol. 8, no. 1, p. 2300507, 2024, doi: 10.1002/solr.202300507.

    # get surface azimuth from solar azimuth and axis azimuth: 
    # if solar azimuth is between axis azimuth - 180 and axis azimuth, surface azimuth = axis azimuth - 90
    # if solar azimuth is between axis azimuth and axis azimuth + 180, surface azimuth = axis azimuth + 90
    relative = (solar_azimuth - axis_azimuth) % 360 # get the solar azimuth angle with respect to axis aximuth

    #if the solar azimuth is >180 degrees past the axis azimuth: surface azimuth is -90 degrees from axis azimuth
    #if the solar azimuth is <180 degrees past the axis azimuth: surface azimuth is +90 degrees from axis azimuth
    surface_azimuth = np.where(relative > 180, (axis_azimuth - 90) % 360, (axis_azimuth + 90) %360)
    
    solar_zenith = np.radians(solar_zenith)
    solar_azimuth = np.radians(solar_azimuth)
    surface_azimuth = np.radians(surface_azimuth)

    numerator = dni * np.sin(solar_zenith) * np.cos(solar_azimuth - surface_azimuth)
    denominator = (dhi - ghi*albedo)/2 + dni * np.cos(solar_zenith)
    beta_opt = np.arctan(numerator / denominator)

    #return tracker theta:  negative when clockwise looking along TT from axis azimuth
    #                       positive when counterclockwise looking along TT from axis azimuth  
    beta_opt = np.where(relative > 180, -1*beta_opt, beta_opt)

    return np.degrees(beta_opt)

def munoz2024_diffuse(tracker_theta,solar_zenith,solar_azimuth, dni_extra, dni, dhi, ghi, albedo, axis_tilt=0, axis_azimuth=180,method='perez-driesse'):
# CENER "diffuse model"
# section 2.1.1 in I. Muñoz, A. Guinda, G. Olivares, S. Díaz, A. M. Gracia-Amillo, and L. Casajús, 
# “Evaluation of Horizontal Single-Axis Solar Tracker Algorithms in Terms of Energy Production 
# and Operational Performance,” Solar RRL, vol. 8, no. 1, p. 2300507, 2024, doi: 10.1002/solr.202300507.

    #Compare G_poa with GHI. If GHI > G_poa, set tracker angle to 0 (flat). If GHI < G_poa, set tracker angle to reference tracking angle.
    
    # get surface orientation
    surface_orientation = pvlib.tracking.calc_surface_orientation(tracker_theta, axis_tilt = axis_tilt, axis_azimuth=axis_azimuth)

    if method == 'perez-driesse':
        poa_sky_diffuse = pvlib.irradiance.perez_driesse(surface_orientation['surface_tilt'],surface_orientation['surface_azimuth'],\
                                dhi, dni, dni_extra,solar_zenith, solar_azimuth)
    elif method == 'isotropic':
        poa_sky_diffuse = pvlib.irradiance.isotropic(surface_orientation['surface_tilt'], dhi)
    elif method == 'haydavies':
        poa_sky_diffuse = pvlib.irradiance.haydavies(surface_orientation['surface_tilt'], surface_orientation['surface_azimuth'],\
                                                    dhi, dni, dni_extra, solar_zenith, solar_azimuth)
        
    #get direct and ground-reflected irradiance contributions
    poa_direct = pvlib.irradiance.beam_component(surface_orientation['surface_tilt'], surface_orientation['surface_azimuth'],\
                                                solar_zenith,solar_azimuth,dni)
    poa_ground = pvlib.irradiance.get_ground_diffuse(surface_orientation['surface_tilt'], ghi, albedo)

    # take sum of three irradiance components to get G_poa
    poa_total_ref = poa_sky_diffuse + poa_direct + poa_ground

    # Find timestamps where GHI > GPOA
    flat_tracking_flag = ghi > poa_total_ref

    # take the reference tracking angle if GHI < GPOA, and 0 if GHI > GPOA
    tracker_angle_set = np.where(flat_tracking_flag, 0, tracker_theta)

    return tracker_angle_set


