"""Synthetic CT tests for screw-trajectory sampling and bone-density features."""

import importlib.util
import math
import unittest


HAS_IMAGING_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "SimpleITK", "pyvista", "vtk")
)

if HAS_IMAGING_DEPS:
    import numpy as np
    import pyvista as pv
    import SimpleITK as sitk

    from pedicle_planning.ai.trajectory import (
        SEGMENT_NAMES,
        LinearCalibration,
        ScrewPlan,
        TrajectorySamplingConfig,
        compute_trajectory_features,
        physical_to_continuous_index,
        sample_image_trilinear,
        sample_screw_points,
    )


def _image_from_function(func, size=(60, 30, 30), spacing=(1.0, 1.0, 1.0),
                         origin=(0.0, 0.0, 0.0), direction=None):
    """Build an image whose voxel values equal func(x, y, z) at voxel-center physical points."""
    image = sitk.Image(list(size), sitk.sitkFloat32)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    origin_arr = np.asarray(origin)
    matrix = np.asarray(image.GetDirection()).reshape(3, 3) * np.asarray(spacing)
    k, j, i = np.meshgrid(np.arange(size[2]), np.arange(size[1]), np.arange(size[0]), indexing="ij")
    idx = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(np.float64)
    phys = origin_arr + idx @ matrix.T
    values = func(phys[:, 0], phys[:, 1], phys[:, 2]).reshape(size[2], size[1], size[0])
    out = sitk.GetImageFromArray(values.astype(np.float32))
    out.CopyInformation(image)
    return out


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class ScrewPlanTests(unittest.TestCase):
    def test_plan_normalizes_direction_and_computes_tip(self):
        plan = ScrewPlan((1.0, 2.0, 3.0), (0.0, 0.0, 2.0), 40.0, 6.5)
        self.assertEqual(plan.direction, (0.0, 0.0, 1.0))
        np.testing.assert_allclose(plan.tip, (1.0, 2.0, 43.0))

    def test_from_entry_tip(self):
        plan = ScrewPlan.from_entry_tip((0, 0, 0), (3, 4, 0), 5.0)
        self.assertAlmostEqual(plan.length, 5.0)
        np.testing.assert_allclose(plan.tip, (3, 4, 0))

    def test_invalid_plans_rejected(self):
        with self.assertRaises(ValueError):
            ScrewPlan((0, 0, 0), (0, 0, 0), 40.0, 6.5)
        with self.assertRaises(ValueError):
            ScrewPlan((0, 0, 0), (1, 0, 0), 0.0, 6.5)
        with self.assertRaises(ValueError):
            ScrewPlan((0, 0, 0), (1, 0, 0), 40.0, -1.0)

    def test_sample_points_cover_cylinder_and_interface(self):
        plan = ScrewPlan((0, 0, 0), (1, 0, 0), 20.0, 6.0)
        samples = sample_screw_points(plan, TrajectorySamplingConfig(INTERFACE_THICKNESS=1.0))
        self.assertTrue(np.all((samples["t"] > 0) & (samples["t"] < 20.0)))
        self.assertLessEqual(samples["r"].max(), 4.0 + 1e-9)
        np.testing.assert_allclose(samples["r"][samples["is_core"]].max(), 3.0, atol=0.5)
        self.assertTrue(np.any(~samples["is_core"]))
        radial = np.hypot(samples["points"][:, 1], samples["points"][:, 2])
        np.testing.assert_allclose(radial, samples["r"], atol=1e-9)


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class ImageSamplingTests(unittest.TestCase):
    def test_trilinear_is_exact_for_linear_field(self):
        image = _image_from_function(lambda x, y, z: 10 * x + 2 * y - z + 5, spacing=(0.8, 1.2, 1.5))
        points = np.array([[3.3, 4.1, 7.7], [20.05, 10.5, 12.25]])
        expected = 10 * points[:, 0] + 2 * points[:, 1] - points[:, 2] + 5
        np.testing.assert_allclose(sample_image_trilinear(image, points), expected, rtol=1e-5)

    def test_points_outside_image_are_nan(self):
        image = _image_from_function(lambda x, y, z: x)
        values = sample_image_trilinear(image, np.array([[-5.0, 1.0, 1.0], [5.0, 1.0, 1.0]]))
        self.assertTrue(math.isnan(values[0]))
        self.assertAlmostEqual(values[1], 5.0, places=5)

    def test_continuous_index_respects_direction_matrix(self):
        image = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        image.SetOrigin((50.0, 0.0, 0.0))
        image.SetSpacing((2.0, 1.0, 1.0))
        image.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, 1))
        point = (44.0, 3.0, 2.0)
        expected = image.TransformPhysicalPointToContinuousIndex(point)
        np.testing.assert_allclose(physical_to_continuous_index(image, np.array([point]))[0], expected)


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class TrajectoryFeatureTests(unittest.TestCase):
    def setUp(self):
        # Screw along +x through the middle of the volume.
        self.plan = ScrewPlan((5.0, 15.0, 15.0), (1.0, 0.0, 0.0), 40.0, 6.0)

    def test_constant_volume(self):
        image = _image_from_function(lambda x, y, z: np.full_like(x, 300.0))
        features = compute_trajectory_features(image, self.plan)
        for name in SEGMENT_NAMES:
            stats = features.segments[name]["all"]
            self.assertAlmostEqual(stats.mean, 300.0, places=3)
            self.assertAlmostEqual(stats.std, 0.0, places=3)
        self.assertAlmostEqual(features.tip_sphere.mean, 300.0, places=3)
        self.assertAlmostEqual(features.outside_image_fraction, 0.0)
        self.assertAlmostEqual(features.in_bone_length, 40.0)
        self.assertEqual(features.unit, "HU")

    def test_axial_gradient_assigns_segment_means(self):
        image = _image_from_function(lambda x, y, z: 10.0 * x)
        features = compute_trajectory_features(image, self.plan)
        for k, name in enumerate(SEGMENT_NAMES):
            lo, hi = features.segment_bounds_mm[k], features.segment_bounds_mm[k + 1]
            expected = 10.0 * (self.plan.entry[0] + 0.5 * (lo + hi))
            self.assertAlmostEqual(features.segments[name]["core"].mean, expected, delta=1.0)
        self.assertAlmostEqual(features.tip_sphere.mean, 10.0 * self.plan.tip[0], delta=1.0)
        self.assertLess(features.segments["entry"]["all"].mean, features.segments["tip"]["all"].mean)

    def test_custom_segment_bounds(self):
        image = _image_from_function(lambda x, y, z: 10.0 * x)
        bounds = (0.0, 5.0, 15.0, 20.0, 30.0, 40.0)
        features = compute_trajectory_features(image, self.plan, segment_bounds_mm=bounds)
        self.assertEqual(features.segment_bounds_mm, bounds)
        self.assertAlmostEqual(features.segments["pedicle"]["core"].mean, 10.0 * (5.0 + 10.0), delta=1.0)
        with self.assertRaises(ValueError):
            compute_trajectory_features(image, self.plan, segment_bounds_mm=(0, 5, 3, 20, 30, 40))

    def test_core_and_interface_are_separated(self):
        # Low-density channel of radius 3.5 mm around the screw axis, dense bone outside.
        def field(x, y, z):
            return np.where(np.hypot(y - 15.0, z - 15.0) <= 3.5, 100.0, 700.0)

        image = _image_from_function(field, spacing=(0.5, 0.5, 0.5), size=(120, 60, 60))
        plan = ScrewPlan((5.0, 15.0, 15.0), (1.0, 0.0, 0.0), 40.0, 5.0)
        features = compute_trajectory_features(image, plan, config=TrajectorySamplingConfig(INTERFACE_THICKNESS=2.0))
        core = features.whole["core"]
        interface = features.whole["interface"]
        self.assertAlmostEqual(core.mean, 100.0, delta=1.0)
        self.assertGreater(interface.mean, 300.0)
        self.assertAlmostEqual(core.low_density_fraction, 1.0)

    def test_direction_matrix_does_not_change_features(self):
        def field(x, y, z):
            return 5.0 * x + 3.0 * y + 200.0

        plain = _image_from_function(field)
        flipped = _image_from_function(field, origin=(59.0, 29.0, 0.0),
                                       direction=(-1, 0, 0, 0, -1, 0, 0, 0, 1))
        a = compute_trajectory_features(plain, self.plan).to_flat_dict()
        b = compute_trajectory_features(flipped, self.plan).to_flat_dict()
        for key in ("pedicle_core_mean", "tip_sphere_mean", "whole_all_std"):
            self.assertAlmostEqual(a[key], b[key], places=3)

    def test_bone_mask_restricts_statistics_and_purchase_length(self):
        image = _image_from_function(lambda x, y, z: np.where(x < 25.0, 50.0, 400.0))
        mask = sitk.Cast(_image_from_function(lambda x, y, z: (x >= 25.0).astype(np.float64)), sitk.sitkUInt8)
        features = compute_trajectory_features(image, self.plan, bone_mask=mask)
        self.assertAlmostEqual(features.in_bone_length, 20.0, delta=1.0)
        self.assertAlmostEqual(features.whole["core"].bone_fraction, 0.5, delta=0.03)
        self.assertGreaterEqual(features.whole["core"].mean, 395.0)
        self.assertEqual(features.segments["entry"]["core"].n_valid, 0)
        self.assertTrue(math.isnan(features.segments["entry"]["core"].mean))

    def test_bone_mesh_membership(self):
        image = _image_from_function(lambda x, y, z: np.full_like(x, 250.0))
        box = pv.Box(bounds=(25.0, 58.0, 2.0, 28.0, 2.0, 28.0)).triangulate()
        features = compute_trajectory_features(image, self.plan, bone_mesh=box)
        self.assertAlmostEqual(features.in_bone_length, 20.0, delta=1.0)

    def test_trajectory_leaving_image(self):
        image = _image_from_function(lambda x, y, z: np.full_like(x, 300.0))
        plan = ScrewPlan((40.0, 15.0, 15.0), (1.0, 0.0, 0.0), 40.0, 6.0)
        features = compute_trajectory_features(image, plan)
        self.assertGreater(features.outside_image_fraction, 0.3)
        self.assertLess(features.in_bone_length, 25.0)
        self.assertAlmostEqual(features.segments["entry"]["all"].mean, 300.0, places=3)
        self.assertTrue(math.isnan(features.segments["tip"]["all"].mean))

    def test_calibration_is_applied(self):
        calibration = LinearCalibration.from_two_points(0.0, 0.0, 1000.0, 800.0)
        self.assertAlmostEqual(calibration.slope, 0.8)
        image = _image_from_function(lambda x, y, z: np.full_like(x, 250.0))
        features = compute_trajectory_features(image, self.plan, calibration=calibration)
        self.assertEqual(features.unit, "mg/cm3")
        self.assertAlmostEqual(features.whole["all"].mean, 200.0, places=3)
        with self.assertRaises(ValueError):
            LinearCalibration.from_two_points(100.0, 1.0, 100.0, 2.0)

    def test_flat_dict_is_complete(self):
        image = _image_from_function(lambda x, y, z: 10.0 * x)
        flat = compute_trajectory_features(image, self.plan).to_flat_dict()
        for name in SEGMENT_NAMES:
            for region in ("core", "interface", "all"):
                self.assertIn(f"{name}_{region}_mean", flat)
        for key in ("tip_sphere_mean", "in_bone_length", "in_bone_fraction", "screw_diameter"):
            self.assertIn(key, flat)
        self.assertTrue(all(np.isfinite(v) for v in flat.values()))


if __name__ == "__main__":
    unittest.main()
