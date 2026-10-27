from satlib._trigd import sind
import numpy as np


def twist_sag(theta, max_twist, module, n_modules):
    """
    Simulate weight-induced twisting along the length of
    a torque tube.

    This model approximately represents the physics of
    weight-induced torque. As the tracker increases its target tilt
    ``theta`` from horizontal, the center of mass of the modules
    and racking moves horizontally away from the center of rotation.
    This creates a moment arm and thus a torque in the direction of
    ``theta`` [1]_.

    Parameters
    ----------
    theta : numeric
        The intended tracker position. [degrees]
    max_twist : numeric
        A scale parameter indicating the maximum amount
        of twist. [degrees]
    module : int
        Which module along the torque tube to compute the
        twist for.  Ranges from 0 to n_modules-1.
    n_modules : int
        Total numer of modules along the torque tube.

    Returns
    -------
    twisted : numeric
        The twisted position for the specified module.

    References
    ----------
    .. [1] K. S. Anderson and C. W. Hansen, "Simulated Performance
       Effect of Torque Tube Twisting in Single-Axis Tracking PV
       Arrays," 2024 IEEE 52nd Photovoltaic Specialist Conference
       (PVSC). IEEE, pp. 1371–1377, Jun. 09, 2024.
       :doi:`10.1109/pvsc57443.2024.10749340`.
    """
    theta_end = theta + max_twist * sind(theta)
    return theta + (theta_end - theta) * (2 * module / (n_modules - 1) - 1)


def twist_lag(theta, max_twist, module, n_modules):
    """
    Simulate friction-induced twisting along the length of
    a torque tube.

    This model approximately represents the hysteresis of
    friction-induced twisting, causing portions of the torque tube
    away from the drive mechanism to lag behind the target
    position [1]_.

    Parameters
    ----------
    theta : numeric
        The intended tracker position. [degrees]
    max_twist : numeric
        A scale parameter indicating the maximum amount
        of twist. [degrees]
    module : int
        Which module along the torque tube to compute the
        twist for.  Ranges from 0 to n_modules-1.
    n_modules : int
        Total numer of modules along the torque tube.

    Returns
    -------
    twisted : numeric
        The twisted position for the specified module.

    Notes
    -----
    This implementation fixes an error in [1]_ for intermediate
    positions along the torque tube length.

    References
    ----------
    .. [1] K. S. Anderson and C. W. Hansen, "Simulated Performance
       Effect of Torque Tube Twisting in Single-Axis Tracking PV
       Arrays," 2024 IEEE 52nd Photovoltaic Specialist Conference
       (PVSC). IEEE, pp. 1371–1377, Jun. 09, 2024.
       :doi:`10.1109/pvsc57443.2024.10749340`.
    """
    max_twist = abs(max_twist * (2 * module / (n_modules - 1) - 1))
    theta = np.array(theta)
    n = len(theta)
    twisted = np.empty(n)
    twisted[0] = theta[0]  # special case
    for i in range(1, n):
        prev = twisted[i-1]
        targ = theta[i]
        diff = targ - prev
        if abs(diff) > max_twist:
            twisted[i] = targ - np.sign(diff) * max_twist
        else:
            twisted[i] = prev

    return twisted
