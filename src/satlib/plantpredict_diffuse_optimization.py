import numpy as np

from satlib.limits import tracker_movement


def _search_irradiance_angle(baseline_theta, score, scores):
    """Search integer angles between horizontal and the exact baseline."""
    baseline_score = score(baseline_theta)
    lower = int(np.ceil(min(baseline_theta, 0)))
    upper = int(np.floor(max(baseline_theta, 0)))
    golden_fraction = (5**0.5 - 1) / 2
    left = lower + int(np.floor((1 - golden_fraction) * (upper - lower)))
    right = lower + int(np.ceil(golden_fraction * (upper - lower)))
    while upper - lower > 3:
        left_score, right_score = score(left), score(right)
        if left_score < right_score:
            lower = left
            left = right
            right += int(np.ceil((1 - golden_fraction) * (upper - right)))
        elif left_score > right_score:
            upper = right
            right = left
            left -= int(np.ceil((1 - golden_fraction) * (left - lower)))
        else:
            lower, upper = left, right
            left = lower + int(np.floor((1 - golden_fraction) * (upper - lower)))
            right = lower + int(np.ceil(golden_fraction * (upper - lower)))
    for theta in range(lower, upper + 1):
        score(theta)

    # Inspect cached probes as well as the final interval. Equal improvements
    # prefer the lower angle; a tie with the exact baseline keeps the baseline.
    theta, best_score = baseline_theta, baseline_score
    for candidate, candidate_score in sorted(scores.items()):
        if np.isfinite(candidate_score) and candidate_score > best_score:
            theta, best_score = candidate, candidate_score
    return theta, best_score


def optimize_irradiance(tracker_theta, ghi, score_irradiance, *,
                        rotation_speed=None, sample_duration=300,
                        hesitation_factor=0.2, initial_theta=None):
    """Optimize a chronological tracker trajectory using a supplied scorer.

    Adapted from the pv-eem irradiance optimization algorithm. Search between
    each baseline angle and horizontal, then compare the reachable optimized
    and baseline paths using the same irradiance model and weather inputs.

    Parameters
    ----------
    tracker_theta : one-dimensional array-like
        Baseline tracker angles in chronological order, with position limits
        and any desired backtracking already applied. [degrees]
    ghi : one-dimensional array-like
        Global horizontal irradiance, aligned with ``tracker_theta``. Zero
        irradiance disables optimization for that interval. [W/m^2]
    score_irradiance : callable
        ``score_irradiance(theta, index)`` returns a scalar irradiance score
        for rotation ``theta`` at integer time position ``index``. Larger is
        better. The caller owns surface geometry, irradiance transposition,
        and optional optical losses. Keep weather inputs fixed across angle
        evaluations. The scorer must return finite values for valid angles.
    rotation_speed : float, optional
        Positive speed in degrees per second. If omitted, movement is instant.
    sample_duration : float, default 300
        Positive duration of each equally spaced interval. [seconds]
    hesitation_factor : float, default 0.2
        Waiting fraction used by :func:`satlib.limits.tracker_movement`.
    initial_theta : float, optional
        Physical position before the first valid interval. Defaults to that
        interval's baseline angle. [degrees]

    Returns
    -------
    result : dict of numpy.ndarray
        ``tracker_theta`` is the accepted interval-average angle;
        ``end_theta`` is the physical endpoint carried to the next interval;
        ``ideal_theta`` is the desired angle before movement correction;
        ``irradiance`` is the score at the accepted average angle;
        ``baseline_irradiance`` is the score at the reachable baseline average;
        ``accepted`` indicates a strict improvement over the reachable baseline.

    Notes
    -----
    The integer golden-section search assumes a unimodal irradiance response
    between baseline and horizontal; it is not a global search for arbitrary
    scoring functions. The exact, possibly fractional baseline is also scored.
    Ties retain the baseline; ties among improved angles choose the lower angle.

    With movement limits, scores are evaluated at interval-average angles,
    rather than integrated along movement paths. Candidate and fallback paths
    start at the same previous physical endpoint. NaN baseline angles are
    preserved and leave that endpoint unchanged. Call separately for each
    tracker; project grouping and score sharing remain the caller's concern.
    """
    baseline = np.asarray(tracker_theta, dtype=float)
    ghi = np.asarray(ghi, dtype=float)
    result = {
        key: np.full(baseline.shape, np.nan)
        for key in ["tracker_theta", "end_theta", "ideal_theta",
                    "irradiance", "baseline_irradiance"]
    }
    result["accepted"] = np.zeros(baseline.shape, dtype=bool)
    endpoint = initial_theta
    for index, reference in enumerate(baseline):
        if not np.isfinite(reference):
            result["tracker_theta"][index] = reference
            result["end_theta"][index] = reference
            result["ideal_theta"][index] = reference
            continue

        scores = {}

        def score(theta):
            theta = float(theta)
            if theta not in scores:
                scores[theta] = score_irradiance(theta, index)
            return scores[theta]

        if ghi[index] != 0:
            ideal, _ = _search_irradiance_angle(reference, score, scores)
        else:
            ideal = reference
        if endpoint is None:
            endpoint = reference
        if rotation_speed is None:
            candidate_average, candidate_end = ideal, ideal
            fallback_average, fallback_end = reference, reference
        else:
            candidate_average, candidate_end = tracker_movement(
                endpoint, ideal, rotation_speed, sample_duration, hesitation_factor
            )
            fallback_average, fallback_end = tracker_movement(
                endpoint, reference, rotation_speed, sample_duration, hesitation_factor
            )
        candidate_score = score(candidate_average)
        fallback_score = score(fallback_average)
        accepted = (ghi[index] != 0 and np.isfinite(candidate_score)
                    and candidate_score > fallback_score)
        average = candidate_average if accepted else fallback_average
        endpoint = candidate_end if accepted else fallback_end
        result["tracker_theta"][index] = average
        result["end_theta"][index] = endpoint
        result["ideal_theta"][index] = ideal
        result["irradiance"][index] = candidate_score if accepted else fallback_score
        result["baseline_irradiance"][index] = fallback_score
        result["accepted"][index] = accepted
    return result
