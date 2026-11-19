"""Physical tracker movement over each sampling interval."""

import numpy as np


def tracker_movement(start_theta, target_theta, rotation_speed,
                     sample_duration, hesitation_factor=0.2):
    """Calculate average rotation and physical endpoint during one interval.

    Parameters
    ----------
    start_theta, target_theta : numeric
        Starting and requested tracker rotations. [degrees]
    rotation_speed : float
        Positive tracker rotation speed. [degrees per second]
    sample_duration : float
        Positive interval duration. [seconds]
    hesitation_factor : float, default 0.2
        Fraction of the interval spent waiting before moving, between zero
        and one. Waiting is shortened when necessary to reach the target.

    Returns
    -------
    average_theta, end_theta : numeric
        Average rotation during the interval and physical rotation at its end.

    Notes
    -----
    Assumes constant speed while moving. When the target cannot be reached,
    movement starts immediately and continues throughout the interval.
    The next interval must start at ``end_theta``, not ``average_theta``.
    """
    movement = np.asarray(target_theta) - np.asarray(start_theta)
    travel = rotation_speed * sample_duration
    fraction = np.abs(movement) / travel
    hesitation = np.minimum(hesitation_factor, np.maximum(0, 1 - fraction))
    average = np.where(
        fraction <= 1,
        start_theta + (1 - fraction / 2 - hesitation) * movement,
        start_theta + np.sign(movement) * travel / 2,
    )
    endpoint = start_theta + np.sign(movement) * np.minimum(np.abs(movement), travel)
    return average, endpoint
