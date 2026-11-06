"""
Optimize tracker angles with the PlantPredict algorithm
======================================================

The optimizer calculates front irradiance internally using Perez–Driesse
transposition, row shading, and physical incidence angle losses. Supply solar
position, weather, and row geometry for a horizontal north-south tracker.
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



result = satlib.plantpredict.optimize_irradiance(
    reference, sun["apparent_zenith"], sun["azimuth"],
    weather["dni"], weather["dhi"], weather["ghi"],
    pvlib.irradiance.get_extra_radiation(times), gcr=0.35,
    rotation_speed=1, sample_duration=300, hesitation_factor=0.2,
)

plt.plot(times, reference, label="Baseline")
plt.plot(times, result["ideal_theta"], label="Ideal")
plt.plot(times, result["tracker_theta"], label="Accepted interval average")
plt.ylabel("Tracker rotation [degrees]")
plt.legend()
plt.gcf().autofmt_xdate()
