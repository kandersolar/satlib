"""
Discretization
==============

Applying time and position discretization to tracker position time series.
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

fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharey=True)

for ax, how in zip(axes, ["lead", "lag", "center"]):
    theta.plot(ax=ax, c='k')
    satlib.discretization.discretize_time(theta, 15, 0, how).plot(ax=ax)
    satlib.discretization.discretize_time(theta, 15, 7, how).plot(ax=ax)

    ax.text(0.05, 0.95, how, transform=ax.transAxes, va="top")

axes[0].set_ylabel("Tracker position [°]")
fig.tight_layout()
fig.subplots_adjust(wspace=0.05)


# %%

fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharey=True)

for ax, how in zip(axes, ["lead", "lag", "center"]):
    theta.plot(ax=ax, c='k')
    satlib.discretization.discretize_angle(theta, interval=10, phase=0, how=how).plot(ax=ax)
    satlib.discretization.discretize_angle(theta, interval=10, phase=3, how=how).plot(ax=ax)
    satlib.discretization.discretize_angle(theta, interval=10, phase=7, how=how).plot(ax=ax)

    ax.text(0.05, 0.95, how, transform=ax.transAxes, va="top")

axes[0].set_ylabel("Tracker position [°]")
fig.tight_layout()
fig.subplots_adjust(wspace=0.05)
