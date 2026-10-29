"""Synthetic-vertebra tests for automatic per-vertebra anatomical frames."""

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

    from pedicle_planning.ai.candidates import CandidateGridConfig, generate_candidates
    from pedicle_planning.ai.trajectory import ScrewPlan
    from pedicle_planning.ai.vertebra_frame import (
        VertebraFrameConfig,
        estimate_vertebra_frame,
        image_axes_frame,
    )

    from synthetic_vertebra import CENTER, _synthetic_vertebra


FAST = None if not HAS_IMAGING_DEPS else VertebraFrameConfig(MAX_SAMPLE_POINTS=8000)


def _angle(u, v):
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(u, v))))))


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class VertebraFrameTests(unittest.TestCase):
    def assertAxesClose(self, frame, axes, tol_deg):
        for k, name in enumerate(("left", "anterior", "cranial")):
            axis = getattr(frame, name)
            self.assertLess(_angle(axis, axes[:, k]), tol_deg, name)
            self.assertGreater(float(np.dot(axis, axes[:, k])), 0.0, f"{name} sign")
        self.assertAlmostEqual(float(np.dot(np.cross(frame.cranial, frame.anterior), frame.left)), 1.0, places=9)

    def test_upright_vertebra(self):
        mask, axes = _synthetic_vertebra()
        frame = estimate_vertebra_frame(mask, config=FAST)
        self.assertAxesClose(frame, axes, 1.0)
        self.assertEqual(frame.method, "symmetry")
        self.assertTrue(frame.reliable)
        local_origin = (frame.origin - np.asarray(CENTER)) @ axes
        np.testing.assert_allclose(local_origin, (0.0, 12.0, 0.0), atol=1.5)

    def test_rotated_and_tilted_vertebrae(self):
        for angles in ((20.0, 10.0, 8.0), (-25.0, -12.0, -10.0), (35.0, 0.0, 0.0), (0.0, 0.0, -20.0)):
            with self.subTest(angles=angles):
                mask, axes = _synthetic_vertebra(*angles)
                self.assertAxesClose(estimate_vertebra_frame(mask, config=FAST), axes, 2.0)

    def test_single_plane_angles_are_reported(self):
        for angles, key in (((15.0, 0.0, 0.0), "axial_rotation_deg"),
                            ((0.0, 10.0, 0.0), "coronal_tilt_deg"),
                            ((0.0, 0.0, 12.0), "sagittal_tilt_deg")):
            with self.subTest(key=key):
                mask, _ = _synthetic_vertebra(*angles)
                reported = estimate_vertebra_frame(mask, config=FAST).angles_to()
                self.assertAlmostEqual(reported[key], max(angles, key=abs), delta=1.5)

    def test_osteophyte_does_not_tilt_cranial_axis(self):
        mask, axes = _synthetic_vertebra(20.0, 10.0, 8.0, spur=True)
        frame = estimate_vertebra_frame(mask, config=FAST)
        self.assertLess(_angle(frame.cranial, axes[:, 2]), 2.0)

    def test_missing_posterior_elements_lower_score_but_keep_axes(self):
        mask, axes = _synthetic_vertebra(20.0, 10.0, 8.0, remove="right_posterior")
        frame = estimate_vertebra_frame(mask, config=FAST)
        self.assertLess(frame.symmetry_score, 0.95)
        self.assertAxesClose(frame, axes, 2.5)

    def test_half_vertebra_is_flagged_unreliable(self):
        mask, _ = _synthetic_vertebra(remove="right_half")
        frame = estimate_vertebra_frame(mask, config=FAST)
        self.assertFalse(frame.reliable)
        self.assertLess(frame.symmetry_score, 0.85)

    def test_side_and_medial_direction(self):
        mask, axes = _synthetic_vertebra(20.0, 0.0, 0.0)
        frame = estimate_vertebra_frame(mask, config=FAST)
        left_entry = np.asarray(CENTER) + axes @ np.array([12.0, -20.0, 0.0])
        right_entry = np.asarray(CENTER) + axes @ np.array([-12.0, -20.0, 0.0])
        self.assertEqual(frame.side_of(left_entry), "left")
        self.assertEqual(frame.side_of(right_entry), "right")
        left_frame = frame.anatomical_frame_for(left_entry)
        right_frame = frame.anatomical_frame_for(right_entry)
        np.testing.assert_allclose(left_frame.medial, -frame.left, atol=1e-9)
        np.testing.assert_allclose(right_frame.medial, frame.left, atol=1e-9)
        np.testing.assert_allclose(left_frame.cranial, frame.cranial, atol=1e-9)
        np.testing.assert_allclose(frame.to_local(frame.origin), (0.0, 0.0, 0.0), atol=1e-9)

    def test_medial_offsets_move_toward_midline_on_both_sides(self):
        mask, axes = _synthetic_vertebra(20.0, 0.0, 0.0)
        frame = estimate_vertebra_frame(mask, config=FAST)
        grid = CandidateGridConfig(MEDIAL_OFFSETS_MM=(2.0,), MEDIAL_ANGLES_DEG=(0.0,),
                                   LENGTH_OFFSETS_MM=(0.0,), DIAMETER_OFFSETS_MM=(0.0,), SNAP_ENTRY=False)
        for local_l in (12.0, -12.0):
            entry = np.asarray(CENTER) + axes @ np.array([local_l, -20.0, 0.0])
            plan = ScrewPlan(tuple(entry), tuple(axes[:, 1]), 40.0, 6.0)
            shifted = generate_candidates(plan, frame.anatomical_frame_for(entry), grid)[1].plan
            self.assertAlmostEqual(abs(frame.to_local(shifted.entry)[0]),
                                   abs(frame.to_local(entry)[0]) - 2.0, delta=0.3)

    def test_image_axes_fallback_and_validation(self):
        mask, _ = _synthetic_vertebra()
        frame = image_axes_frame(mask)
        self.assertEqual(frame.method, "image_axes")
        np.testing.assert_allclose(frame.left, (1.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(frame.anterior, (0.0, -1.0, 0.0), atol=1e-12)
        empty = sitk.Image(10, 10, 10, sitk.sitkUInt8)
        with self.assertRaises(ValueError):
            estimate_vertebra_frame(empty)
        with self.assertRaises(ValueError):
            VertebraFrameConfig(BODY_FRACTION=0.0).validate()


if __name__ == "__main__":
    unittest.main()
