"""
Critical angles
===============

Computing backtracking critical angles.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import satlib
import pvlib

# %%

times = pd.date_range("2026-12-21 7:30:00", "2026-12-21 16:30:00", freq="1min", tz="UTC")
sp = pvlib.solarposition.get_solarposition(times, 45, 0)
sp = sp.loc[sp["apparent_elevation"] > 0, :]

gcr = 0.4
cross_axis_tilt = 5
tt = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180)
bt = satlib.backtracking.backtrack_standard(tt, gcr=gcr, cross_axis_tilt=cross_axis_tilt)

theta = np.where(np.isnan(bt), tt, bt)
theta = pd.Series(theta, index=sp.index)

# %%

critical_angles = satlib.geometry.critical_angles(gcr, cross_axis_tilt=cross_axis_tilt)
print(critical_angles)

# %%

# convert angles to times by searching PSZA
timings = (
    abs(tt - critical_angles[0]).idxmin(),
    abs(tt - critical_angles[1]).idxmin(),
)

# %%

theta.plot()

for angle in critical_angles:
    plt.axhline(angle, c='k', ls=':')

for timing in timings:
    plt.axvline(timing, c='k', ls=':')
