"""Synthetic-vertebra tests for whole vertebral-body trabecular bone density."""

import importlib.util
import math
import unittest


HAS_IMAGING_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "SimpleITK", "pyvista", "vtk")
)

if HAS_IMAGING_DEPS:
    import numpy as np
    import SimpleITK as sitk

    from pedicle_planning.ai.trajectory import LinearCalibration
    from pedicle_planning.ai.vertebra_frame import VertebraFrameConfig, estimate_vertebra_frame, image_axes_frame
    from pedicle_planning.ai.vertebral_body import VertebralBodyConfig, compute_vertebral_body_bmd

    from synthetic_vertebra import synthetic_vertebra_ct


FAST = None if not HAS_IMAGING_DEPS else VertebraFrameConfig(MAX_SAMPLE_POINTS=8000)


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class VertebralBodyBMDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct, cls.mask, cls.axes = synthetic_vertebra_ct()
        cls.frame = estimate_vertebra_frame(cls.mask, config=FAST)
        cls.bmd = compute_vertebral_body_bmd(cls.ct, cls.mask, cls.frame)

    def test_midbody_slab_mean(self):
        # cancellous HU = 120 + 8 |height|; mid slab |height| <= 5 mm -> mean 140
        self.assertAlmostEqual(self.bmd.midbody.mean, 140.0, delta=2.0)
        self.assertLess(self.bmd.midbody.p90, 165.0)
        self.assertEqual(self.bmd.unit, "HU")

    def test_cortex_endplates_and_posterior_elements_excluded(self):
        trab = self.bmd.trabecular
        self.assertEqual(trab.cortical_fraction, 0.0)
        self.assertLess(trab.p90, 200.0)  # posterior elements are 300 HU, cortex 1000 HU
        # 3 mm margin keeps |height| <= 9 mm -> mean 156
        self.assertAlmostEqual(trab.mean, 156.0, delta=3.0)
        self.assertGreater(trab.mean, self.bmd.midbody.mean)
        self.assertLess(self.bmd.trabecular_volume_mm3, self.bmd.body_volume_mm3)

    def test_body_height(self):
        self.assertAlmostEqual(self.bmd.body_height_mm, 24.0, delta=1.5)

    def test_rotated_vertebra_with_estimated_frame(self):
        ct, mask, _ = synthetic_vertebra_ct(20.0, 10.0, 8.0)
        bmd = compute_vertebral_body_bmd(ct, mask, estimate_vertebra_frame(mask, config=FAST))
        self.assertAlmostEqual(bmd.midbody.mean, 140.0, delta=2.0)
        self.assertEqual(bmd.trabecular.cortical_fraction, 0.0)
        self.assertAlmostEqual(bmd.body_height_mm, 24.0, delta=1.5)

    def test_image_axes_frame_matches_upright_case(self):
        bmd = compute_vertebral_body_bmd(self.ct, self.mask, image_axes_frame(self.mask))
        self.assertAlmostEqual(bmd.midbody.mean, 140.0, delta=2.0)

    def test_thicker_slab_and_no_margin(self):
        thick = compute_vertebral_body_bmd(self.ct, self.mask, self.frame,
                                           config=VertebralBodyConfig(MIDBODY_SLAB_MM=16.0))
        self.assertAlmostEqual(thick.midbody.mean, 152.0, delta=2.5)  # |height| <= 8 mm
        raw = compute_vertebral_body_bmd(self.ct, self.mask, self.frame,
                                         config=VertebralBodyConfig(CORTICAL_MARGIN_MM=0.0))
        self.assertGreater(raw.trabecular.cortical_fraction, 0.05)

    def test_calibration_and_flat_dict(self):
        cal = LinearCalibration(slope=0.5, intercept=10.0)
        bmd = compute_vertebral_body_bmd(self.ct, self.mask, self.frame, calibration=cal)
        self.assertAlmostEqual(bmd.midbody.mean, 0.5 * self.bmd.midbody.mean + 10.0, places=6)
        self.assertEqual(bmd.unit, "mg/cm3")
        flat = self.bmd.to_flat_dict()
        for key in ("vertebra_midbody_mean", "vertebra_trabecular_p10", "vertebra_body_height_mm",
                    "vertebra_frame_symmetry_score"):
            self.assertIn(key, flat)
        self.assertTrue(all(np.isfinite(v) for v in flat.values()))

    def test_validation(self):
        with self.assertRaises(ValueError):
            VertebralBodyConfig(MIDBODY_SLAB_MM=0.0).validate()
        empty = sitk.Image(10, 10, 10, sitk.sitkUInt8)
        with self.assertRaises(ValueError):
            compute_vertebral_body_bmd(self.ct, empty, self.frame)


if __name__ == "__main__":
    unittest.main()
