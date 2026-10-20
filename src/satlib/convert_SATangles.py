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