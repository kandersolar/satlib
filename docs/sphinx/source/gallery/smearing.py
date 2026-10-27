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
bt = satlib.backtracking.simple.backtrack(tt, gcr=0.4)

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
for how in ["steeper", "shallower"]:
    smeared = satlib.limits.clip_speed(theta, 1, how=how)
    smeared.plot(label=how)
    plt.legend()

