"""
Notes:
    - Decision variable list goes across all trackers for T=0, then all
      trackers for T=1, then all trackers for T=2, ..., and then deltas for
      all trackers for T=2, then all deltas for T=3, ...
"""

import numpy as np
import scipy

from satlib._trigd import cosd, sind, acosd, atand


def _get_feasibility_mask(psza, x, y):

    masks = []

    for theta_t in psza:
        mask = np.full_like(x, True, dtype=bool)
        # identify "impossible" cases. While there are any rows that
        # cannot be kept shade-free, repeatedly remove the worst case from
        # consideration until the remaining rows are solvable.
        while True:
            beta_c = np.degrees(np.arctan(np.diff(y[mask])/np.diff(x[mask])))
            is_possible = np.abs(theta_t - beta_c) <= 90

            if np.all(is_possible):
                # problem is now solvable
                break

            # nix the rear member of the worst-case pair
            idx_worst_pair = np.argmax(np.abs(theta_t - beta_c))
            left = idx_worst_pair
            right = idx_worst_pair + 1
            idx_worst_pair_rear = left if theta_t < 0 else right
            temp = mask[mask]
            temp[idx_worst_pair_rear] = False
            mask[mask] = temp

        masks.append(mask)

    mask = np.array(masks)

    return mask


def _get_mechanical_bounds(psza, n_trackers, l, theta_max):
    lb1 = np.maximum(0, l * cosd(-theta_max - psza))
    lb2 = np.maximum(0, l * cosd(+theta_max - psza))
    ub = np.where(abs(psza) < theta_max, l, l * cosd(theta_max - abs(psza)))
    lb = np.minimum(lb1, lb2)
    lb = np.repeat(lb, n_trackers)
    ub = np.repeat(ub, n_trackers)
    return list(zip(lb, ub))  # linprog wants sequence of pairs


def _get_self_shading_constraint(psza, x, y, is_feasible, max_shaded_fraction):
    # works only for PSZA > 0    

    n_trackers = len(x)
    n_times = len(psza)

    rows = []
    cols = []
    vals = []

    widths = []

    row = 0
    for T, _psza in enumerate(psza):
        for i in range(n_trackers-1):
            for j in range(i+1, n_trackers):
                if is_feasible[T, j]:
                    break
            else:
                # no more trackers that can be kept shade-free
                break

            if _psza > 0:
                left = 0.5
                right = 0.5 - max_shaded_fraction
            else:
                left = 0.5 - max_shaded_fraction
                right = 0.5

            rows.append(row)
            cols.append(T * n_trackers + i)
            vals.append(left)

            rows.append(row)
            cols.append(T * n_trackers + j)
            vals.append(right)

            beta_c = np.arctan((y[j] - y[i]) / (x[j] - x[i]))
            w = sind(90 - _psza + np.degrees(beta_c)) * (x[j] - x[i]) / np.cos(beta_c)
            widths.append(w)

            row += 1

    A_ub = scipy.sparse.csc_matrix((vals, (rows, cols)),
                                   shape=(row, n_trackers * n_times))
    b_ub = np.array(widths)
    return A_ub, b_ub


def _get_gang_constraint(psza, n_trackers, gang_size):
    n_gangs = n_trackers // gang_size

    rows = []
    cols = []
    vals = []

    row = 0
    for T, _psza in enumerate(psza):
        for gang_id in range(n_gangs):
            for i in range(gang_size - 1):

                rows.append(row)
                cols.append(T * n_trackers + gang_id * gang_size + i)
                vals.append(1)

                rows.append(row)
                cols.append(T * n_trackers + gang_id * gang_size + i + 1)
                vals.append(-1)

                row += 1

    A_eq = scipy.sparse.csc_matrix((vals, (rows, cols)))
    b_eq = np.zeros(A_eq.shape[0])
    return A_eq, b_eq


def _get_maximum_irradiance_constraint(psza, is_feasible, maximum_total_width, n_trackers):

    A_ub = np.hstack([
        -1 * np.array([is_feasible.ravel().astype(int)]),  # widths
        np.zeros((1, n_trackers*(len(psza)-2)))  # for deltas
    ])
    A_ub = scipy.sparse.csc_matrix(A_ub)

    b_ub = np.array([-0.999999*maximum_total_width])
    return A_ub, b_ub


def _get_smoothness_constraint(psza, n_trackers, l):
    # note: this is optimizing smoothness in interception width, not
    # rotation angle
    n_times = len(psza)

    rows = []
    cols = []
    vals = []

    row = 0
    for T in range(1, n_times - 1):
        for i in range(n_trackers):

            # first "half" of the absolute value
            rows.append(row)
            cols.append((T-1) * n_trackers + i)
            vals.append(1)

            rows.append(row)
            cols.append(T * n_trackers + i)
            vals.append(-2)

            rows.append(row)
            cols.append((T+1) * n_trackers + i)
            vals.append(1)

            rows.append(row)
            cols.append(n_trackers * n_times + (T-1) * n_trackers + i)
            vals.append(-1)  # delta


            # second "half" of the absolute value
            rows.append(row + 1)
            cols.append((T-1) * n_trackers + i)
            vals.append(-1)

            rows.append(row + 1)
            cols.append(T * n_trackers + i)
            vals.append(2)

            rows.append(row + 1)
            cols.append((T+1) * n_trackers + i)
            vals.append(-1)

            rows.append(row + 1)
            cols.append(n_trackers * n_times + (T-1) * n_trackers + i)
            vals.append(-1)  # delta

            row += 2

    A_ub = scipy.sparse.csc_matrix((vals, (rows, cols)),
                                   shape=(row, n_times * n_trackers + (n_times-2) * n_trackers))
    b_ub = np.zeros(A_ub.shape[0])
    return A_ub, b_ub


def _run_half_day(psza, x, y, l, theta_max, gang_size, max_shaded_fraction,
                  **kwargs):
    # positive PSZA only

    n_times = len(psza)
    n_trackers = len(x)
    assert n_trackers % gang_size == 0

    # identify rows where the terrain prevents successful backtracking,
    # i.e. the rows where keeping shading below max_shaded_fraction is impossible
    is_feasible = _get_feasibility_mask(psza, x, y)


    ## first optimization: get maximum possible total interception width
    A_shading, b_shading = _get_self_shading_constraint(psza, x, y, is_feasible, max_shaded_fraction)

    if gang_size > 1:
        A_gang, b_gang = _get_gang_constraint(psza, n_trackers, gang_size)
    else:
        A_gang = b_gang = None

    # optimization variables
    c = -1 * is_feasible.ravel().astype(int)  # only count irradiance for non-shaded trackers

    mechanical_bounds = _get_mechanical_bounds(psza, n_trackers, l, theta_max)

    opt_result = scipy.optimize.linprog(c, A_shading, b_shading, A_gang, b_gang,
                                        bounds=mechanical_bounds, **kwargs)

    if not opt_result.success:
        raise RuntimeError("Could not solve initial optimization: " + opt_result.message)

    # maximum possible interception width, across all timestamps and rows,
    # is the sum of individual interception widths:
    maximum_total_width = np.sum(opt_result.x[is_feasible.ravel()])


    ## second optimization: maximum interception width, maximum smoothness
    A_maxirrad, b_maxirrad = _get_maximum_irradiance_constraint(psza, is_feasible, maximum_total_width, n_trackers)

    A_smoothness, b_smoothness = _get_smoothness_constraint(psza, n_trackers, l)

    # re-use the shading constraint from before, but add columns to support deltas:
    A_shading = scipy.sparse.hstack((A_shading, np.zeros((A_shading.shape[0], (n_times-2) * n_trackers))))

    if gang_size > 1:
        A_eq = scipy.sparse.hstack((A_gang, np.zeros((A_gang.shape[0], (n_times-2) * n_trackers))))
        b_eq = b_gang
    else:
        A_eq = b_eq = None

    A_ub = scipy.sparse.vstack([A_maxirrad, A_shading, A_smoothness])
    b_ub = np.concatenate([b_maxirrad, b_shading, b_smoothness])

    c = np.zeros(n_trackers*n_times + (n_times-2) * n_trackers)  # optimization variables
    c[-((n_times-2) * n_trackers):] = 1  # minimize sum of deltas

    # same bounds as before for the widths; deltas get separate bounds
    mechanical_bounds = mechanical_bounds + [(0, None)] * ((n_times-2) * n_trackers)

    opt_result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=mechanical_bounds, **kwargs)

    if not opt_result.success:
        raise RuntimeError("Could not solve final optimization: " + opt_result.message)

    w_opt = opt_result.x[:-((n_times-2) * n_trackers)]  # drop deltas

    # allow a bit of nuisance numerical error
    w_opt[(w_opt > l) & (w_opt < l + 1e-3)] = l

    w_opt = w_opt.reshape((n_times, n_trackers)).T
    theta = psza - np.sign(psza) * acosd(w_opt / l)
    theta = theta.T

    #deltas = opt_result.x[-((n_times-2) * n_trackers):]

    return theta #, deltas, is_feasible


def backtrack_lp2d(psza, x, y, collector_bandwidth, theta_max, gang_size=1,
                   max_shaded_fraction=0, **kwargs):
    """
    Compute backtracking angles for rolling terrain using the LP2D method.

    The LP2D method [1]_ solve a linear optimization problem to determine
    backtracking schedules for all trackers in an array simultaneously.

    The terrain slope is assumed to vary only in the direction perpendicular
    to the tracker axes.

    Parameters
    ----------
    psza : numeric
        Project solar zenith angle. [degrees]

    x : np.array
        x-coordinates for each tracker axis, in the coordinate system
        aligned with the tracker axes. [meters]

    y : np.array
        y-coordinates for each tracker axis, in the coordinate system
        aligned with the tracker axes. [meters]

    collector_bandwidth : float
        Width of each tracker table. [meters]

    theta_max : float
        Maximum allowable tracker position. [degrees]

    gang_size : int, default 1
        Number of trackers per gang.  Trackers in a gang are constrained
        to track together.

    max_shaded_fraction : float, default 0
        Maximum tolerable shaded collector fraction. [unitless]
    
    kwargs : optional
        Additional parameters to pass through to ``scipy.optimize.linprog``.
    
    Returns
    -------
    positions : np.ndarray
        Optimized tracker positions. Shape is (len(psza), len(x)). [degrees]

    References
    ----------
    .. [1] K. S. Anderson, A. R. Jensen, and D. M. Riley, "A Linear Programming
       Approach to Backtracking for Single-Axis Trackers on Rolling Terrain,"
       IEEE J. Photovoltaics, vol. 16, no. 2, pp. 291–298, Mar. 2026,
       :doi:`10.1109/jphotov.2025.3645781`. 
    """

    # determine the boundaries between backtracking & truetracking
    pitch = np.diff(x)
    beta_c = atand(np.diff(y) / pitch)
    beta_c_max = np.max(beta_c)
    beta_c_min = np.min(beta_c)
    pitch = x[1] - x[0]  # assumes all x evenly spaced

    limit1 = beta_c_min + acosd(cosd(beta_c_min) * collector_bandwidth / pitch)
    limit2 = beta_c_max - acosd(cosd(beta_c_max) * collector_bandwidth / pitch)

    # The optimization code is much simpler if you can assume PSZA > 0, so
    # split the problem into two parts and invert the negative half

    idx = np.arange(len(psza))
    idx1 = idx[psza > limit1]
    idx2 = idx[psza < limit2]

    theta_1 = _run_half_day(psza[idx1], x, y, collector_bandwidth, theta_max, gang_size,
                            max_shaded_fraction, **kwargs)

    theta_2 = _run_half_day(-psza[idx2][::-1], -x[::-1], y[::-1], collector_bandwidth, theta_max, gang_size,
                            max_shaded_fraction, **kwargs)
    theta_2 = -theta_2[::-1, ::-1]

    # default to truetracking
    theta = np.repeat([psza], len(x), axis=0).T
    theta = np.clip(theta, -theta_max, theta_max)

    # and then fill in the backtracking portions
    theta[idx1, :] = theta_1
    theta[idx2, :] = theta_2
    return theta
