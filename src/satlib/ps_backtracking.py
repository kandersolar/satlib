import numpy as np
import warnings
import pvlib


def get_effective_gcr(psza, gcr, allowed_shade_fractions=0):
    candidate_gcrs = (1 - np.array(allowed_shade_fractions)) * gcr
    critical_angles = np.degrees(np.arccos(candidate_gcrs))
    psza = np.array(psza)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        mask = (
            (critical_angles < np.abs(psza)) &
            (np.abs(psza - np.sign(psza) * np.degrees(np.arccos(np.cos(np.radians(psza))/candidate_gcrs))) < 60)
        )
    masked_gcr = np.where(mask, candidate_gcrs, np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        gcr_eff = np.nanmin(masked_gcr, axis=1)

    return gcr_eff


def partial_shade_tracking(sp, gcr, allowed_shade_fractions):
    psza = pvlib.shading.projected_solar_zenith_angle(sp.apparent_zenith, sp.azimuth,
                                                      axis_tilt=0, axis_azimuth=180)
    gcr_eff = get_effective_gcr(psza, gcr, allowed_shade_fractions)
    tr = pvlib.tracking.singleaxis(sp.apparent_zenith, sp.azimuth, max_angle=60,
                                   gcr=gcr_eff, axis_azimuth=180, backtrack=True)
    return tr


def partial_shade_tracking2(sp, gcr, allowed_shade_fractions):
    psza = pvlib.shading.projected_solar_zenith_angle(sp.apparent_zenith, sp.azimuth,
                                                      axis_tilt=0, axis_azimuth=180)
    theta = psza.copy()
    theta[sp.apparent_elevation < 0] = np.nan
    for fss in sorted(allowed_shade_fractions):
        candidate_tr = pvlib.tracking.singleaxis(sp.apparent_zenith, sp.azimuth, max_angle=60,
                                                 gcr=gcr*(1-fss), axis_azimuth=180, backtrack=True)
        candidate_tr = candidate_tr.tracker_theta
        candidate_fs = pvlib.shading.shaded_fraction1d(sp.apparent_zenith, sp.azimuth, 180, candidate_tr, collector_width=gcr, pitch=1)
        theta = np.where(np.abs(candidate_fs - fss) < 1e-4, candidate_tr, theta)
    return theta
