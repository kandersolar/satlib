import numpy as np

def clip_tracking_angle(tracker_theta, ref_tracker_theta,backtracking_mask,max_angle=60):
    # clip a series of tracker angles to be less than:
    # while backtracking: the backtracking angle 
    # while not backtracking: the maximum tracker angle
    # tracker_theta: the tracking angle timeseries
    # ref_tracker_theta: the reference tracking angle timeseries (traditional backtracking)
    # backtracking_mask: a boolean mask indicating when backtracking is active
    # max_angle: the maximum tracker angle (degrees)

    clipped_theta = np.where(backtracking_mask, #when backtracking
                        np.clip(tracker_theta, -1*np.abs(ref_tracker_theta), np.abs(ref_tracker_theta)), #clip to backtracking angle
                        np.clip(tracker_theta, -1*max_angle, max_angle)) #otherwise, clip to max_angle

    return clipped_theta

def get_surface_azimuth(solar_azimuth, axis_azimuth):
    # for an HSAT tracker, get surface azimuth from solar azimuth and axis azimuth (all in degrees)
    # the surface azimuth is either +90 or -90 degrees from the axis azimuth, depending on the solar azimuth angle:
        # if solar azimuth is from (axis azimuth - 180) to axis azimuth, surface azimuth = axis azimuth - 90
        # if solar azimuth is from axis azimuth to (axis azimuth + 180), surface azimuth = axis azimuth + 90

    # get the solar azimuth angle with respect to axis aximuth
    relative_angle = (solar_azimuth - axis_azimuth) % 360 

    # flag what side of the axis the solar azimuth is on
    axis_flag = relative_angle > 180

    #if the solar azimuth is >180 degrees past the axis azimuth: surface azimuth = axis azimuth - 90
    #if the solar azimuth is <180 degrees past the axis azimuth: surface azimuth = axis azimuth + 90
    surface_azimuth = np.where(axis_flag, (axis_azimuth - 90) % 360, (axis_azimuth + 90) %360)

    return surface_azimuth, axis_flag