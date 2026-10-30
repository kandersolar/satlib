"""
Twisting
========

Applying weight- and friction-induced twist to tracker position time series.
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

theta.plot(c='k', label='target')
for module in [0, 10, 20, 30, 39]:
    twisted = satlib.twist.twist_sag(theta, max_twist=10, module=module, n_modules=40)
    twisted.plot(label=f"{module=}")

plt.legend()

# %%

theta.plot(c='k', label='target')
for module in [0, 10, 20, 30, 39]:
    twisted = satlib.twist.twist_lag(theta, max_twist=10, module=module, n_modules=40)
    twisted = pd.Series(twisted, index=sp.index)
    twisted.plot(label=f"{module=}")

plt.legend()
