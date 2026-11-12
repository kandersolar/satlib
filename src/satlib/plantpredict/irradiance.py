"""Front irradiance and optical losses for PlantPredict optimization."""

import numpy as np
import pvlib


def front_irradiance(theta, solar_zenith, solar_azimuth, dni, dhi, ghi,
                      dni_extra, airmass, gcr, axis_azimuth, albedo,
                      circumsolar, soiling, diffuse_factors):
    """Evaluate the algorithm's front optical model at a candidate rotation."""
    if dni == 0 and dhi == 0 and ghi == 0:
        return 0.0
    orientation = pvlib.tracking.calc_surface_orientation(
        theta, axis_tilt=0, axis_azimuth=axis_azimuth,
    )
    tilt = np.asarray(orientation["surface_tilt"]).item()
    azimuth = np.asarray(orientation["surface_azimuth"]).item()
    sky = pvlib.irradiance.perez_driesse(
        tilt, azimuth, dhi, dni, dni_extra, solar_zenith, solar_azimuth,
        airmass=airmass, return_components=True,
    )
    aoi = pvlib.irradiance.aoi(tilt, azimuth, solar_zenith, solar_azimuth)
    beam = pvlib.irradiance.beam_component(
        tilt, azimuth, solar_zenith, solar_azimuth, dni,
    )
    ground = pvlib.irradiance.get_ground_diffuse(tilt, ghi, albedo)

    # EEM rounds grouping keys, but evaluates the original geometry values.
    direct_loss = 1 - pvlib.shading.shaded_fraction1d(
        solar_zenith, solar_azimuth, axis_azimuth,
        theta, collector_width=1, pitch=1 / gcr,
        axis_tilt=0,
    )
    mask = pvlib.shading.masking_angle(
        tilt, gcr, slant_height=0,
    )
    diffuse_loss = 1 - pvlib.shading.sky_diffuse_passias(mask)
    direct_iam = pvlib.iam.physical(aoi, n=1.526, K=4, L=0.0032, n_ar=1)
    if tilt not in diffuse_factors:
        diffuse_factors[tilt] = pvlib.iam.marion_diffuse(
            "physical", tilt, n=1.526, K=4, L=0.0032,
        )
    iam = diffuse_factors[tilt]
    circumsolar_factor = (direct_loss * direct_iam if circumsolar == "separate"
                         else diffuse_loss * iam["sky"])
    return float(soiling * (
        beam * direct_loss * direct_iam
        + ground * direct_loss * iam["ground"]
        + sky["poa_isotropic"] * diffuse_loss * iam["sky"]
        + sky["poa_horizon"] * diffuse_loss * iam["horizon"]
        + sky["poa_circumsolar"] * circumsolar_factor
    ))


def cached_front_irradiance(theta, inputs, scores):
    """Reuse irradiance evaluations for an angle within one timestamp."""
    theta = float(theta)
    if theta not in scores:
        scores[theta] = front_irradiance(theta, **inputs)
    return scores[theta]
