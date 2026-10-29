"""Synthetic-vertebra tests for pedicle isthmus detection and candidate optimization."""

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

    from pedicle_planning.ai.candidates import (
        AnatomicalFrame,
        CandidateGridConfig,
        adjust_plan,
        diameter_within_isthmus,
        entry_on_bone_surface,
        generate_candidates,
        optimize_screw_plan,
        resolve_pivot_point,
        snap_entry_to_surface,
    )
    from pedicle_planning.ai.pedicle import IsthmusDetectionConfig, detect_pedicle_isthmus
    from pedicle_planning.ai.trajectory import ScrewPlan


PEDICLE_TILT_DEG = 10.0
ISTHMUS_X = 14.0
ISTHMUS_HALF_WIDTH = 4.5
CORTEX_MM = 1.0


def _synthetic_vertebra():
    """
    Posterior block (x 0-6) + hourglass elliptical pedicle (x 6-26, axis tilted
    in +y by PEDICLE_TILT_DEG, narrowest at x = 14) + cylindrical body (x 26-60).
    Returns (ct_image, bone_mask); cortical shell 1000 HU, cancellous 200 HU.
    """
    spacing = 0.5
    size = (140, 80, 80)
    k, j, i = np.meshgrid(np.arange(size[2]), np.arange(size[1]), np.arange(size[0]), indexing="ij")
    x, y, z = i * spacing, j * spacing, k * spacing

    posterior = (x <= 6.0) & (np.abs(y - 20.0) <= 15.0) & (np.abs(z - 20.0) <= 12.0)
    half_width = ISTHMUS_HALF_WIDTH + 0.02 * (x - ISTHMUS_X) ** 2
    yc = 20.0 + math.tan(math.radians(PEDICLE_TILT_DEG)) * (x - ISTHMUS_X)
    pedicle = ((x > 6.0) & (x < 26.0)
               & (((y - yc) / half_width) ** 2 + ((z - 20.0) / (1.4 * half_width)) ** 2 <= 1.0))
    body = (x >= 26.0) & (x <= 60.0) & ((y - 20.0) ** 2 + (z - 20.0) ** 2 <= 14.0 ** 2)
    bone = (posterior | pedicle | body).astype(np.uint8)

    mask = sitk.GetImageFromArray(bone)
    mask.SetSpacing((spacing,) * 3)
    eroded = sitk.GetArrayFromImage(sitk.BinaryErode(mask, [int(CORTEX_MM / spacing)] * 3))
    hu = np.where(bone > 0, np.where(eroded > 0, 200.0, 1000.0), 0.0).astype(np.float32)
    ct = sitk.GetImageFromArray(hu)
    ct.CopyInformation(mask)
    return ct, mask


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class CandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        self.frame = AnatomicalFrame(medial=(0.0, 1.0, 0.0), cranial=(0.0, 0.0, 1.0))
        self.baseline = ScrewPlan((1.0, 20.0, 20.0), (1.0, 0.0, 0.0), 45.0, 5.5)

    def test_frame_orthogonalizes_cranial(self):
        frame = AnatomicalFrame(medial=(0.0, 2.0, 0.0), cranial=(0.0, 1.0, 1.0))
        np.testing.assert_allclose(frame.cranial, (0.0, 0.0, 1.0), atol=1e-12)
        with self.assertRaises(ValueError):
            AnatomicalFrame(medial=(0.0, 1.0, 0.0), cranial=(0.0, 2.0, 0.0))

    def test_adjust_plan_sign_conventions(self):
        plan = adjust_plan(self.baseline, self.frame, medial_offset_mm=1.5, cranial_offset_mm=-1.0,
                           medial_angle_deg=5.0, cranial_angle_deg=-3.0, length=50.0, diameter=6.0)
        np.testing.assert_allclose(plan.entry, (1.0, 21.5, 19.0))
        self.assertGreater(plan.direction[1], 0.0)  # tip toward medial
        self.assertLess(plan.direction[2], 0.0)  # tip toward caudal
        self.assertAlmostEqual(math.degrees(math.atan2(plan.direction[1], plan.direction[0])), 5.0, places=6)
        self.assertAlmostEqual(math.degrees(math.asin(-plan.direction[2])), 3.0, places=6)
        self.assertEqual((plan.length, plan.diameter), (50.0, 6.0))

    def test_default_grid_size_and_baseline_first(self):
        candidates = generate_candidates(self.baseline, self.frame)
        self.assertEqual(len(candidates), 90)
        self.assertTrue(candidates[0].is_baseline)
        self.assertEqual(candidates[0].plan, self.baseline)
        self.assertEqual(sum(c.is_baseline for c in candidates), 1)
        keys = {(c.medial_offset_mm, c.medial_angle_deg, c.plan.length, c.plan.diameter) for c in candidates}
        self.assertEqual(len(keys), 90)

    def test_catalog_lengths_and_custom_grid(self):
        grid = CandidateGridConfig(MEDIAL_OFFSETS_MM=(0.0,), MEDIAL_ANGLES_DEG=(0.0,),
                                   CRANIAL_ANGLES_DEG=(-5.0, 0.0, 5.0))
        candidates = generate_candidates(self.baseline, self.frame, grid,
                                         lengths=(40.0, 45.0, 50.0), diameters=(5.5, 6.5))
        self.assertEqual(len(candidates), 1 + 3 * 3 * 2 - 1)
        with self.assertRaises(ValueError):
            CandidateGridConfig(MEDIAL_ANGLES_DEG=(95.0,)).validate()
        with self.assertRaises(ValueError):
            CandidateGridConfig(PIVOT="tip").validate()

    def test_pivot_point_keeps_axis_through_pivot(self):
        pivot = np.array([13.0, 20.0, 20.0])
        plan = adjust_plan(self.baseline, self.frame, medial_angle_deg=5.0, pivot_point=pivot)
        entry, direction = np.asarray(plan.entry), np.asarray(plan.direction)
        to_pivot = pivot - entry
        self.assertLess(np.linalg.norm(to_pivot - np.dot(to_pivot, direction) * direction), 1e-9)
        self.assertAlmostEqual(np.dot(to_pivot, direction), 12.0, places=9)
        self.assertAlmostEqual(entry[1] - 20.0, -12.0 * math.sin(math.radians(5.0)), places=9)
        unchanged = adjust_plan(self.baseline, self.frame, pivot_point=pivot)
        np.testing.assert_allclose(unchanged.entry, self.baseline.entry, atol=1e-12)
        shifted = adjust_plan(self.baseline, self.frame, medial_offset_mm=1.0, medial_angle_deg=5.0,
                              pivot_point=pivot)
        to_shifted_pivot = pivot + np.array([0.0, 1.0, 0.0]) - np.asarray(shifted.entry)
        d = np.asarray(shifted.direction)
        self.assertLess(np.linalg.norm(to_shifted_pivot - np.dot(to_shifted_pivot, d) * d), 1e-9)

    def test_isthmus_pivot_requires_pivot_source(self):
        self.assertIsNone(resolve_pivot_point(self.baseline, "entry"))
        with self.assertRaises(ValueError):
            resolve_pivot_point(self.baseline, "isthmus")
        np.testing.assert_allclose(resolve_pivot_point(self.baseline, "isthmus", pivot_point=(5, 6, 7)), (5, 6, 7))
        with self.assertRaises(ValueError):
            generate_candidates(self.baseline, self.frame, CandidateGridConfig(PIVOT="isthmus"))

    def test_snap_status_without_bone_data(self):
        candidates = generate_candidates(self.baseline, self.frame)
        self.assertEqual(candidates[0].entry_snap_status, "not_requested")
        self.assertTrue(all(c.entry_snap_status == "no_bone_data" for c in candidates[1:]))
        plain = generate_candidates(self.baseline, self.frame, CandidateGridConfig(SNAP_ENTRY=False))
        self.assertTrue(all(c.entry_snap_status == "not_requested" for c in plain))
        with self.assertRaises(ValueError):
            snap_entry_to_surface(self.baseline)

    def test_snap_to_mesh_surface(self):
        box = pv.Box(bounds=(10.0, 40.0, 10.0, 30.0, 10.0, 30.0)).triangulate()
        plan = ScrewPlan((7.0, 20.0, 20.0), (1.0, 0.0, 0.0), 20.0, 5.0)
        snapped, shift = snap_entry_to_surface(plan, bone_mesh=box)
        self.assertAlmostEqual(snapped.entry[0], 10.0, delta=0.25)
        self.assertAlmostEqual(shift, 3.0, delta=0.25)
        np.testing.assert_allclose(np.asarray(snapped.tip) - np.asarray(plan.tip), (shift, 0.0, 0.0), atol=1e-9)
        self.assertEqual((snapped.length, snapped.diameter), (plan.length, plan.diameter))


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class PedicleAndOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct, cls.mask = _synthetic_vertebra()
        cls.frame = AnatomicalFrame(medial=(0.0, 1.0, 0.0), cranial=(0.0, 0.0, 1.0))
        cls.baseline = ScrewPlan((1.0, 20.0, 20.0), (1.0, 0.0, 0.0), 45.0, 5.5)
        cls.pedicle = detect_pedicle_isthmus(cls.mask, cls.baseline, image=cls.ct)

    def test_isthmus_location_and_widths(self):
        ped = self.pedicle
        self.assertAlmostEqual(ped.isthmus_point[0], ISTHMUS_X, delta=1.0)
        self.assertAlmostEqual(ped.isthmus_point[1], 20.0, delta=0.5)
        self.assertAlmostEqual(ped.isthmus_point[2], 20.0, delta=0.5)
        expected_outer = 2 * ISTHMUS_HALF_WIDTH * math.cos(math.radians(PEDICLE_TILT_DEG))
        self.assertAlmostEqual(ped.outer_min_width, expected_outer, delta=0.6)
        self.assertAlmostEqual(ped.inner_min_width, expected_outer - 2 * CORTEX_MM, delta=0.6)
        self.assertLess(ped.isthmus.area_inner, ped.isthmus.area_outer)

    def test_pedicle_axis_recovers_tilt(self):
        match = self.pedicle.trajectory_match(self.baseline)
        self.assertAlmostEqual(match["angle_deg"], PEDICLE_TILT_DEG, delta=1.5)
        self.assertGreater(self.pedicle.axis[1], 0.0)
        aligned = ScrewPlan(tuple(self.pedicle.isthmus_point - 13.0 * self.pedicle.axis),
                            tuple(self.pedicle.axis), 45.0, 5.5)
        aligned_match = self.pedicle.trajectory_match(aligned)
        self.assertLess(aligned_match["angle_deg"], 0.5)
        self.assertLess(aligned_match["isthmus_offset_mm"], 0.3)
        self.assertGreater(aligned_match["outer_clearance_mm"], 1.0)

    def test_wall_clearance_detects_breach(self):
        shifted = adjust_plan(self.baseline, self.frame, medial_offset_mm=3.0)
        self.assertLess(self.pedicle.trajectory_match(shifted)["outer_clearance_mm"], 0.0)

    def test_segment_bounds_follow_anatomy(self):
        bounds = self.pedicle.segment_bounds_for(self.baseline)
        self.assertEqual(len(bounds), 6)
        self.assertTrue(all(b2 >= b1 for b1, b2 in zip(bounds, bounds[1:])))
        isthmus_t = self.pedicle.trajectory_match(self.baseline)["isthmus_t_mm"]
        self.assertLess(bounds[1], isthmus_t)
        self.assertGreater(bounds[2], isthmus_t)
        self.assertGreaterEqual(bounds[1], 4.0)  # pedicle starts after the posterior block (x = 6)
        self.assertLessEqual(bounds[2], 26.0)  # and ends before the body (x = 26)
        self.assertEqual(bounds[-1], self.baseline.length)

    def test_snap_entry_to_mask_surface(self):
        # The posterior bone surface of the synthetic vertebra is at x = 0.
        inside, shift_in = snap_entry_to_surface(self.baseline, bone_mask=self.mask)
        self.assertAlmostEqual(inside.entry[0], 0.0, delta=0.5)
        self.assertLess(shift_in, 0.0)
        outside_plan = ScrewPlan((-3.0, 20.0, 20.0), (1.0, 0.0, 0.0), 45.0, 5.5)
        outside, shift_out = snap_entry_to_surface(outside_plan, bone_mask=self.mask)
        self.assertAlmostEqual(outside.entry[0], 0.0, delta=0.5)
        self.assertAlmostEqual(shift_out, 3.0, delta=0.5)
        stray = ScrewPlan((1.0, 2.0, 2.0), (1.0, 0.0, 0.0), 45.0, 5.5)
        self.assertEqual(snap_entry_to_surface(stray, bone_mask=self.mask)[0], None)

    def test_candidates_are_snapped_and_baseline_kept(self):
        candidates = generate_candidates(self.baseline, self.frame, bone_mask=self.mask)
        self.assertEqual(candidates[0].plan, self.baseline)
        self.assertTrue(all(c.entry_snap_status == "snapped" for c in candidates[1:]))
        for c in candidates[1:]:
            self.assertAlmostEqual(c.plan.entry[0], 0.0, delta=0.6)

    def test_unreachable_entry_is_rejected(self):
        grid = CandidateGridConfig(MEDIAL_OFFSETS_MM=(0.0, 30.0), MEDIAL_ANGLES_DEG=(0.0,),
                                   LENGTH_OFFSETS_MM=(0.0,), DIAMETER_OFFSETS_MM=(0.0,))
        result = optimize_screw_plan(self.ct, self.baseline, self.frame, lambda f: 0.0,
                                     bone_mask=self.mask, pedicle=self.pedicle, grid=grid,
                                     constraints=[entry_on_bone_surface()])
        far = [e for e in result.evaluations if e.candidate.medial_offset_mm == 30.0]
        self.assertEqual(len(far), 1)
        self.assertEqual(far[0].candidate.entry_snap_status, "failed")
        self.assertFalse(far[0].feasible)
        self.assertIn("bone surface", far[0].violations[0])

    def test_isthmus_pivot_preserves_isthmus_crossing(self):
        grid_kwargs = dict(MEDIAL_OFFSETS_MM=(0.0,), MEDIAL_ANGLES_DEG=(-5.0, 5.0),
                           LENGTH_OFFSETS_MM=(0.0,), DIAMETER_OFFSETS_MM=(0.0,))
        base_offset = self.pedicle.trajectory_match(self.baseline)["isthmus_offset_mm"]
        for pivot, expect_preserved in (("isthmus", True), ("entry", False)):
            candidates = generate_candidates(self.baseline, self.frame, CandidateGridConfig(PIVOT=pivot, **grid_kwargs),
                                             pedicle=self.pedicle, bone_mask=self.mask)
            for c in candidates[1:]:
                self.assertEqual(c.pivot, pivot)
                offset = self.pedicle.trajectory_match(c.plan)["isthmus_offset_mm"]
                if expect_preserved:
                    self.assertAlmostEqual(offset, base_offset, delta=0.2)
                    # entry moves laterally for a medial tilt, medially for a lateral tilt
                    self.assertLess(c.plan.entry[1] * np.sign(c.medial_angle_deg), 20.0 * np.sign(c.medial_angle_deg))
                else:
                    self.assertGreater(offset, base_offset + 0.8)

    def test_detection_fails_outside_bone(self):
        stray = ScrewPlan((1.0, 2.0, 2.0), (1.0, 0.0, 0.0), 45.0, 5.5)
        with self.assertRaises(ValueError):
            detect_pedicle_isthmus(self.mask, stray, image=self.ct)
        with self.assertRaises(ValueError):
            IsthmusDetectionConfig(SEARCH_RANGE=(0.5, 0.2)).validate()

    def test_optimization_respects_constraints_and_lowers_risk(self):
        def risk_fn(features):
            # Toy risk: penalize low interface density, pedicle misalignment and short purchase.
            density = features["whole_interface_mean"]
            return float(1.0 / (1.0 + math.exp((density - 400.0) / 100.0))
                         + 0.01 * features["match_angle_deg"]
                         - 0.002 * features["in_bone_length"])

        result = optimize_screw_plan(self.ct, self.baseline, self.frame, risk_fn,
                                     bone_mask=self.mask, pedicle=self.pedicle)
        summary = result.summary()
        self.assertEqual(summary["n_candidates"], 90)
        self.assertTrue(result.baseline.candidate.is_baseline)
        self.assertTrue(np.isfinite(summary["baseline_risk"]))
        self.assertIsNotNone(result.best)
        self.assertTrue(result.best.feasible)
        self.assertLessEqual(summary["best_risk"], summary["baseline_risk"])
        self.assertGreaterEqual(summary["risk_reduction"], 0.0)
        self.assertLess(summary["n_feasible"], summary["n_candidates"])
        for evaluation in result.evaluations:
            if not evaluation.feasible:
                self.assertTrue(evaluation.violations)
                if not evaluation.candidate.is_baseline:
                    self.assertTrue(math.isnan(evaluation.risk))
            if abs(evaluation.candidate.medial_offset_mm) == 2.0 and evaluation.candidate.medial_angle_deg == 0.0:
                self.assertFalse(evaluation.feasible)
        for key in ("medial_offset_mm", "medial_angle_deg", "length_mm", "diameter_mm",
                    "risk_reduction_relative", "pivot", "entry_snap_status",
                    "entry_displacement_medial_mm", "entry_displacement_axial_mm"):
            self.assertIn(key, summary)
        self.assertIn("match_angle_deg", result.best.features)
        self.assertIn("pedicle_core_mean", result.best.features)

    def test_sector_and_patient_features_reach_risk_function(self):
        seen = []

        def risk_fn(features):
            seen.append(features)
            return -features["pedicle_interface_medial_mean"]

        grid = CandidateGridConfig(MEDIAL_OFFSETS_MM=(0.0,), MEDIAL_ANGLES_DEG=(0.0,),
                                   LENGTH_OFFSETS_MM=(0.0,), DIAMETER_OFFSETS_MM=(0.0,))
        optimize_screw_plan(self.ct, self.baseline, self.frame, risk_fn, bone_mask=self.mask,
                            pedicle=self.pedicle, grid=grid,
                            patient_features={"vertebra_midbody_mean": 140.0, "age": 70})
        self.assertTrue(seen)
        for features in seen:
            self.assertEqual(features["vertebra_midbody_mean"], 140.0)
            self.assertEqual(features["age"], 70)
            self.assertIn("transition_interface_caudal_p10", features)
        with self.assertRaises(ValueError):
            optimize_screw_plan(self.ct, self.baseline, self.frame, lambda f: 0.0, bone_mask=self.mask,
                                pedicle=self.pedicle, grid=grid, patient_features={"in_bone_length": 1.0})

    def test_no_feasible_candidate(self):
        result = optimize_screw_plan(self.ct, self.baseline, self.frame, lambda f: 0.0,
                                     bone_mask=self.mask, pedicle=self.pedicle,
                                     constraints=[diameter_within_isthmus(margin_mm=10.0)],
                                     grid=CandidateGridConfig(MEDIAL_OFFSETS_MM=(0.0,),
                                                              MEDIAL_ANGLES_DEG=(0.0,)))
        self.assertIsNone(result.best)
        self.assertEqual(result.n_feasible, 0)
        self.assertTrue(math.isnan(result.summary()["best_risk"]))
        self.assertFalse(result.summary()["baseline_feasible"])


if __name__ == "__main__":
    unittest.main()
