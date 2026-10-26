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

tr = pvlib.tracking.singleaxis(sp.apparent_zenith, sp.azimuth, axis_azimuth=180, gcr=0.4, backtrack=True)
theta = tr['tracker_theta']

# %%

theta.plot(c='k', label='baseline')
for how in ["catch-up", "anticipate"]:
    smeared = satlib.smear.smear(theta, 1, how=how)
    smeared.plot(label=how)
    plt.legend()

# %%

theta.plot(c='k', label='baseline')
for how in ["steeper", "shallower"]:
    smeared = satlib.smear.smear(theta, 1, how=how)
    smeared.plot(label=how)
    plt.legend()

