"""
Optimize tracker angles with a custom irradiance model
=====================================================

The optimizer coordinates angle selection and tracker movement. The caller
supplies the irradiance calculation, so the same search can use incident
irradiance or a model including shading and other optical losses.

This example uses pvlib's isotropic sky model for a horizontal north-south
tracker. Weather is prepared once and stays fixed while comparing angles.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import satlib

times = pd.date_range("2026-06-21 06:00", "2026-06-21 18:00",
                      freq="5min", tz="Etc/GMT+7")
site = pvlib.location.Location(35, -110, tz="Etc/GMT+7")
sun = site.get_solarposition(times)
weather = site.get_clearsky(times)

# Add a diffuse-dominated period around midday.
cloudy = (times.hour >= 10) & (times.hour < 14)
weather.loc[cloudy, "dni"] = 30
weather.loc[cloudy, "dhi"] = 250
weather["ghi"] = (
    weather["dni"] * np.cos(np.radians(sun["apparent_zenith"]))
    + weather["dhi"]
)
reference = pvlib.tracking.singleaxis(
    sun["apparent_zenith"], sun["azimuth"],
    axis_tilt=0, axis_azimuth=180, max_angle=60, backtrack=True, gcr=0.35,
)["tracker_theta"].to_numpy()


def score_irradiance(theta, index):
    orientation = pvlib.tracking.calc_surface_orientation(
        theta, axis_tilt=0, axis_azimuth=180,
    )
    position = sun.iloc[index]
    irradiance = weather.iloc[index]
    components = pvlib.irradiance.get_total_irradiance(
        orientation["surface_tilt"], orientation["surface_azimuth"],
        position["apparent_zenith"], position["azimuth"],
        irradiance["dni"], irradiance["ghi"], irradiance["dhi"],
        albedo=0.2, model="isotropic",
    )
    return np.asarray(components["poa_global"]).item()


result = satlib.diffuse_optimization.optimize_irradiance(
    reference, weather["ghi"].to_numpy(), score_irradiance,
    rotation_speed=1, sample_duration=300, hesitation_factor=0.2,
)

plt.plot(times, reference, label="Baseline")
plt.plot(times, result["ideal_theta"], label="Ideal")
plt.plot(times, result["tracker_theta"], label="Accepted interval average")
plt.ylabel("Tracker rotation [degrees]")
plt.legend()
plt.gcf().autofmt_xdate()
