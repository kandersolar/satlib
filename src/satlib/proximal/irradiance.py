"""Front irradiance and optical losses for PlantPredict optimization."""

import numpy as np
import pvlib


def front_irradiance(theta, solar_zenith, solar_azimuth, dni, dhi, ghi,
                      dni_extra, airmass, gcr, axis_azimuth, albedo,
                      circumsolar, soiling, diffuse_factors, pitch, pile_height):
    """Evaluate the algorithm's front optical model at a candidate rotation."""
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
    ground = pvlib.bifacial.infinite_sheds.get_irradiance_poa(
        surface_tilt=tilt, surface_azimuth=azimuth,
        solar_zenith=solar_zenith, solar_azimuth=solar_azimuth,
        gcr=gcr, height=pile_height, pitch=pitch,
        ghi=ghi, dhi=dhi, dni=dni, albedo=albedo, model="isotropic",
    )["poa_ground_diffuse"]

    direct_loss = 1 - pvlib.shading.shaded_fraction1d(
        solar_zenith, solar_azimuth, axis_azimuth,
        theta, collector_width=1, pitch=1 / gcr,
        axis_tilt=0,
    )
    sky_view_factor = pvlib.bifacial.utils.vf_row_sky_2d_integ(
        surface_tilt=tilt, gcr=gcr, x0=0, x1=1,
    )
    unshaded_sky_view_factor = (1 + pvlib.tools.cosd(tilt)) / 2
    # Preserve PV-EEM's shade-fraction conversion and arithmetic order.
    diffuse_shade_fraction = 1 - sky_view_factor / unshaded_sky_view_factor
    sky_transmission = 1 - diffuse_shade_fraction
    direct_iam = pvlib.iam.physical(aoi, n=1.526, K=4, L=0.0032, n_ar=1)
    if tilt not in diffuse_factors:
        diffuse_factors[tilt] = pvlib.iam.marion_diffuse(
            "physical", tilt, n=1.526, K=4, L=0.0032,
        )
    iam = diffuse_factors[tilt]
    # Reusing isotropic transmission for horizon and diffuse circumsolar
    # irradiance is an approximation, matching PV-EEM.
    circumsolar_poa = sky["poa_circumsolar"]
    if circumsolar == "separate":
        circumsolar_poa = circumsolar_poa * direct_loss * soiling * direct_iam
    else:
        circumsolar_poa = circumsolar_poa * sky_transmission * soiling * iam["sky"]
    # Component order and staged losses match PV-EEM's front-only scorer.
    return float(np.array([
        beam * direct_loss * soiling * direct_iam,
        circumsolar_poa,
        sky["poa_horizon"] * sky_transmission * soiling * iam["horizon"],
        sky["poa_isotropic"] * sky_transmission * soiling * iam["sky"],
        np.asarray(ground).item() * soiling * iam["ground"],
    ]).sum())


def cached_front_irradiance(theta, inputs, scores):
    """Reuse irradiance evaluations for an angle within one timestamp."""
    theta = float(theta)
    if theta not in scores:
        scores[theta] = front_irradiance(theta, **inputs)
    return scores[theta]
