import numpy as np

def munoz2024(solar_zenith, solar_azimuth, dni, dhi, ghi, albedo):
    # not sure how surface_azimuth is supposed to be known.
    # assume north-south axis and choose east/west based on sun position?
    surface_azimuth = np.where(solar_azimuth < 180, 90, 270)
    
    solar_zenith = np.radians(solar_zenith)
    solar_azimuth = np.radians(solar_azimuth)
    surface_azimuth = np.radians(surface_azimuth)

    numerator = dni * np.sin(solar_zenith) * np.cos(solar_azimuth - surface_azimuth)
    denominator = (dhi - ghi*albedo)/2 + dni * np.cos(solar_zenith)
    beta_opt = np.arctan(numerator / denominator)
    return np.degrees(beta_opt)
