Function reference
==================

Tracking
--------

Functions for determining the optimal angle of a tracker.

.. autosummary::
   :toctree: reference

   satlib.tracking.astronomical
   satlib.diffuse_optimization.munoz2024_analytical
   satlib.diffuse_optimization.kellygibson2009_tracking1
   satlib.diffuse_optimization.kellygibson2009_tracking2
   satlib.diffuse_optimization.kellygibson2009_tracking3
   satlib.diffuse_optimization.munoz2024_diffuse
   satlib.diffuse_optimization.munoz2024_diffusetrackerangle


Backtracking
------------

Functions for avoiding and managing row-to-row shade.

.. autosummary::
   :toctree: reference

   satlib.backtracking.backtrack_standard
   satlib.backtracking.backtrack_psbt
   satlib.backtracking.backtrack_psbt_effective_gcr
   satlib.backtracking.backtrack_lp2d


Discretization
--------------

Functions for modeling discontinuous tracker movement.

.. autosummary::
   :toctree: reference

   satlib.discretization.discretize_time
   satlib.discretization.discretize_angle


Twisting
--------

Functions for modeling twisting along the length of a torque tube.

.. autosummary::
   :toctree: reference

   satlib.twist.twist_lag
   satlib.twist.twist_sag


Limits
------

Functions for applying limits to tracker position and speed.

.. autosummary::
   :toctree: reference

   satlib.limits.clip_position
   satlib.limits.clip_speed


Geometry
--------

Functions for calculating geometric quantities for trackers.

.. autosummary::
   :toctree: reference

   satlib.geometry.aoi
   satlib.geometry.surface_orientation
   satlib.geometry.critical_angles
   satlib.geometry.axis_tilt
   satlib.geometry.cross_axis_tilt
   satlib.geometry.projected_solar_zenith_angle


Shading
-------

Functions for row-to-row shading.

.. autosummary::
   :toctree: reference

   satlib.shading.shaded_fraction
