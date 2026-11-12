"""
Partial-shade backtracking
==========================

Computing partial-shade backtracking positions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import satlib
import pvlib

# %%

cw = 2
p = 5
gcr = cw / p

times = pd.date_range("2026-12-21 7:30:00", "2026-12-21 16:30:00", freq="15s", tz="UTC")
sp = pvlib.solarposition.get_solarposition(times, 45, 0)
sp = sp.loc[sp["apparent_elevation"] > 0, :]

tt = satlib.tracking.astronomical(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180)

bt1 = satlib.backtracking.backtrack_psbt(tt, gcr=gcr, allowed_shade_fractions=[0, 0.5], max_angle=None)
bt2 = satlib.backtracking.backtrack_psbt(tt, gcr=gcr, allowed_shade_fractions=[0, 0.5], max_angle=45)

fs1 = satlib.geometry.shaded_fraction(tt, bt1, cw, p)
fs2 = satlib.geometry.shaded_fraction(tt, bt2, cw, p)

# %%

fig, axes = plt.subplots(2, 1, sharex=True)

tt.plot(ax=axes[0], label="true-tracking", c='k', ls=':')
bt1.plot(ax=axes[0], label="max_angle=None")
bt2.plot(ax=axes[0], label="max_angle=45")
fs1.plot(ax=axes[1], label="max_angle=None")
fs2.plot(ax=axes[1], label="max_angle=45")

axes[0].set_ylabel("Tracker position [degrees]")
axes[1].set_ylabel("Shaded fraction [-]")

for ax in axes:
    ax.legend()
