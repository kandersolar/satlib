"""Integer angle search between the baseline and horizontal."""

import numpy as np

from .irradiance import cached_front_irradiance


def search_irradiance_angle(baseline_theta, irradiance_inputs, scores):
    """Search integer angles between horizontal and the exact baseline."""
    baseline_score = cached_front_irradiance(
        baseline_theta, irradiance_inputs, scores,
    )
    lower = int(np.ceil(min(baseline_theta, 0)))
    upper = int(np.floor(max(baseline_theta, 0)))
    golden_fraction = (5**0.5 - 1) / 2
    left = lower + int(np.floor((1 - golden_fraction) * (upper - lower)))
    right = lower + int(np.ceil(golden_fraction * (upper - lower)))
    while upper - lower > 3:
        left_score = cached_front_irradiance(left, irradiance_inputs, scores)
        right_score = cached_front_irradiance(right, irradiance_inputs, scores)
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
        cached_front_irradiance(theta, irradiance_inputs, scores)

    # Inspect cached probes as well as the final interval. Equal improvements
    # prefer the lower angle; a tie with the exact baseline keeps the baseline.
    theta, best_score = baseline_theta, baseline_score
    for candidate, candidate_score in sorted(scores.items()):
        if np.isfinite(candidate_score) and candidate_score > best_score:
            theta, best_score = candidate, candidate_score
    return theta, best_score
