"""
Smearing
========

Applying speed limits to tracker position time series.
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

tt = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180)
bt = satlib.backtracking.backtrack_standard(tt, gcr=0.4)

theta = np.where(np.isnan(bt), tt, bt)
theta = pd.Series(theta, index=sp.index)

# %%

theta.plot(c='k', label='baseline')
for how in ["lead", "lag"]:
    smeared = satlib.limits.clip_speed(theta, 1, how=how)
    smeared.plot(label=how)
    plt.legend()

# %%

theta.plot(c='k', label='baseline')
smeared = satlib.limits.clip_speed(theta, 1, how="shallower")
smeared.plot(label=how)
plt.legend()

# %%

n = 500
theta = pd.Series(25 * np.sin(np.linspace(0, 4*np.pi, n)), index=pd.date_range("2020-06-01 12:00", freq="6s", periods=n))
theta = np.clip(theta, -20, 20)
limit = 5

theta.plot(c='k', label='baseline')
smeared = satlib.limits.clip_speed(theta, 5, how="shallower")
smeared.plot(label=how)
plt.legend()