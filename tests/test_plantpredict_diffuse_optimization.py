import unittest

import numpy as np

from satlib.plantpredict import optimize_irradiance
from satlib.plantpredict.irradiance import front_irradiance


class TestPlantPredictDiffuseOptimization(unittest.TestCase):
    def test_eem_irradiance_with_unrounded_shading_inputs(self):
        # EEM's Double Black Diamond fixture, 2026-10-27 08:20 America/Chicago.
        # Rounding solar position or row pitch before shading breaks parity.
        actual = front_irradiance(
            theta=-48.74,
            solar_zenith=80.39188670206246,
            solar_azimuth=115.58468652421527,
            dni=46.900953055756055, dhi=72.30771349336007, ghi=80.062,
            dni_extra=1378.268168435841, airmass=5.7971266649504996,
            gcr=0.425, axis_azimuth=180, albedo=0.2,
            circumsolar="diffuse", soiling=1, diffuse_factors={},
        )
        np.testing.assert_allclose(actual, 107.11491749611892, rtol=0, atol=1e-10)

    def test_normal_beam_has_no_optical_loss(self):
        result = optimize_irradiance(
            [40], [0], [90], [800], [0], [800], 1367, gcr=0.35,
        )
        np.testing.assert_allclose(result["tracker_theta"], [0])
        np.testing.assert_allclose(result["irradiance"], [800])
        self.assertTrue(result["accepted"][0])

    def test_horizontal_diffuse_irradiance(self):
        result = optimize_irradiance(
            [-40], [0], [90], [0], [300], [300], 1367, gcr=0.35,
        )
        np.testing.assert_allclose(result["tracker_theta"], [0])
        # Integrated transmission through the model's uncoated glass.
        np.testing.assert_allclose(result["irradiance"], [283.13925028])
        self.assertTrue(result["accepted"][0])

    def test_zero_weather_and_missing_angles(self):
        result = optimize_irradiance(
            [20.5, np.nan, -20.5], [100] * 3, [90] * 3,
            [0] * 3, [0] * 3, [0] * 3, 1367, gcr=0.35,
        )
        np.testing.assert_allclose(
            result["tracker_theta"], [20.5, np.nan, -20.5], equal_nan=True,
        )
        np.testing.assert_allclose(
            result["irradiance"], [0, np.nan, 0], equal_nan=True,
        )
        self.assertFalse(result["accepted"].any())

    def test_movement_carries_physical_endpoint(self):
        result = optimize_irradiance(
            [40, 40], [0, 0], [90, 90], [800, 800], [0, 0],
            [800, 800], 1367, gcr=0.35, rotation_speed=1,
            sample_duration=10,
        )
        np.testing.assert_allclose(result["ideal_theta"], [0, 0])
        np.testing.assert_allclose(result["tracker_theta"], [35, 25])
        np.testing.assert_allclose(result["end_theta"], [30, 20])
        self.assertTrue(result["accepted"].all())

    def test_soiling_scales_both_paths(self):
        arguments = ([40], [60], [90], [500], [150], [400], 1367)
        for circumsolar in ["diffuse", "separate"]:
            with self.subTest(circumsolar=circumsolar):
                clean = optimize_irradiance(
                    *arguments, gcr=0.35, circumsolar=circumsolar,
                )
                dirty = optimize_irradiance(
                    *arguments, gcr=0.35, circumsolar=circumsolar, soiling=0.8,
                )
                np.testing.assert_allclose(
                    dirty["tracker_theta"], clean["tracker_theta"],
                )
                for key in ["irradiance", "baseline_irradiance"]:
                    np.testing.assert_allclose(dirty[key], clean[key] * 0.8)

    def test_row_shading_reduces_irradiance(self):
        arguments = ([60], [80], [270], [800], [100], [240], 1367)
        # Zero GHI disables optimization, so retain the supplied angle while
        # comparing row density with identical weather and surface orientation.
        arguments = (*arguments[:5], [0], arguments[6])
        sparse = optimize_irradiance(*arguments, gcr=0.2)
        dense = optimize_irradiance(*arguments, gcr=0.8)
        self.assertLess(dense["irradiance"][0], sparse["irradiance"][0])


if __name__ == "__main__":
    unittest.main()
