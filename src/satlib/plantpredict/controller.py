"""Coordinate angle selection, tracker movement, and path acceptance."""

import numpy as np
import pvlib

from .irradiance import cached_front_irradiance
from .movement import tracker_movement
from .search import search_irradiance_angle


def optimize_irradiance(tracker_theta, solar_zenith, solar_azimuth, dni, dhi,
                        ghi, dni_extra, *, gcr, axis_azimuth=180, albedo=0.2,
                        circumsolar="diffuse", soiling=1.0,
                        rotation_speed=None, sample_duration=300,
                        hesitation_factor=0.2, initial_theta=None):
    """Optimize a chronological tracker trajectory using front irradiance.

    Adapted from the pv-eem irradiance optimization algorithm. Search between
    each baseline angle and horizontal, then compare the reachable optimized
    and baseline paths using the same irradiance model and weather inputs.
    The built-in model uses Perez–Driesse sky diffuse transposition, direct
    row shading, Passias sky shading, and physical direct and diffuse incidence
    angle modifiers with uncoated glass (n=1.526, K=4, L=0.0032 m).
    Relative airmass is calculated using the Kasten–Young 1989 model.

    Parameters
    ----------
    tracker_theta : one-dimensional array-like
        Baseline tracker angles in chronological order, with position limits
        and any desired backtracking already applied. [degrees]
    solar_zenith, solar_azimuth : one-dimensional array-like
        Apparent solar zenith and solar azimuth, aligned with tracker angles.
        [degrees]
    dni, dhi, ghi : one-dimensional array-like
        Direct normal, diffuse horizontal, and global horizontal irradiance.
        Zero GHI disables optimization for that interval. [W/m^2]
    dni_extra : float or one-dimensional array-like
        Extraterrestrial direct normal irradiance at each interval. [W/m^2]
    gcr : float
        Collector width divided by row pitch, greater than zero and at most
        one. Rows have equal rotations on flat terrain.
    axis_azimuth : float, default 180
        Azimuth of the horizontal tracker axis, clockwise from north. [degrees]
    albedo : float, default 0.2
        Ground reflectance.
    circumsolar : {"diffuse", "separate"}, default "diffuse"
        Apply sky shading and diffuse IAM to circumsolar irradiance, or apply
        direct shading and direct IAM instead. Matches pv-eem's default.
    soiling : float or one-dimensional array-like, default 1
        Fraction of irradiance transmitted after soiling, between zero and one.
    rotation_speed : float, optional
        Positive speed in degrees per second. If omitted, movement is instant.
    sample_duration : float, default 300
        Positive duration of each equally spaced interval. [seconds]
    hesitation_factor : float, default 0.2
        Waiting fraction used by :func:`tracker_movement`.
    initial_theta : float, optional
        Physical position before the first valid interval. Defaults to that
        interval's baseline angle. [degrees]

    Returns
    -------
    result : dict of numpy.ndarray
        ``tracker_theta`` is the accepted interval-average angle;
        ``end_theta`` is the physical endpoint carried to the next interval;
        ``ideal_theta`` is the desired angle before movement correction;
        ``irradiance`` is front effective irradiance at the accepted average;
        ``baseline_irradiance`` is its value at the reachable baseline average;
        ``accepted`` indicates a strict improvement over the reachable baseline.

    """
    baseline = np.asarray(tracker_theta, dtype=float)
    airmass = pvlib.atmosphere.get_relative_airmass(
        np.asarray(solar_zenith, dtype=float)
    )
    solar_zenith, solar_azimuth, dni, dhi, ghi, dni_extra, soiling, airmass = [
        np.broadcast_to(np.asarray(value, dtype=float), baseline.shape)
        for value in (solar_zenith, solar_azimuth, dni, dhi, ghi,
                      dni_extra, soiling, airmass)
    ]
    diffuse_factors = {}
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

        irradiance_inputs = {
            "solar_zenith": solar_zenith[index],
            "solar_azimuth": solar_azimuth[index],
            "dni": dni[index],
            "dhi": dhi[index],
            "ghi": ghi[index],
            "dni_extra": dni_extra[index],
            "airmass": airmass[index],
            "gcr": gcr,
            "axis_azimuth": axis_azimuth,
            "albedo": albedo,
            "circumsolar": circumsolar,
            "soiling": soiling[index],
            "diffuse_factors": diffuse_factors,
        }

        if ghi[index] != 0:
            ideal, _ = search_irradiance_angle(reference, irradiance_inputs, scores)
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
        candidate_score = cached_front_irradiance(
            candidate_average, irradiance_inputs, scores,
        )
        fallback_score = cached_front_irradiance(
            fallback_average, irradiance_inputs, scores,
        )
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
