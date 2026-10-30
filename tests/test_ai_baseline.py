"""Synthetic-vertebra tests for automatically generated reference (baseline) screws."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HAS_IMAGING_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "SimpleITK", "pyvista", "vtk")
)
SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "clinical"

if HAS_IMAGING_DEPS:
    import numpy as np
    import SimpleITK as sitk

    from pedicle_planning.ai.baseline import BaselineConfig, generate_baseline_screws
    from pedicle_planning.ai.candidates import snap_entry_to_surface
    from pedicle_planning.ai.pedicle import detect_pedicle_isthmus
    from pedicle_planning.ai.trajectory import sample_mask_nearest
    from pedicle_planning.ai.vertebra_frame import estimate_vertebra_frame

    from synthetic_vertebra import CENTER, synthetic_vertebra_ct


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class BaselineScrewTests(unittest.TestCase):
    """The synthetic pedicles run along the local anterior axis at left/right 12 mm, narrowest at
    anterior -8 mm but joined by transverse processes between -12 and -6 mm."""

    def test_reference_screws_are_safe_and_anatomical(self):
        config = BaselineConfig()
        for rotation in ((0.0, 0.0, 0.0), (20.0, 10.0, 8.0), (-25.0, -12.0, -10.0)):
            ct, mask, axes = synthetic_vertebra_ct(*rotation)
            frame = estimate_vertebra_frame(mask)
            results = generate_baseline_screws(mask, frame)
            for side, sign in (("left", 1.0), ("right", -1.0)):
                with self.subTest(rotation=rotation, side=side):
                    screw = results[side]
                    self.assertNotIsInstance(screw, str, screw)
                    local = (screw.isthmus_center - np.asarray(CENTER)) @ axes
                    self.assertAlmostEqual(local[0], 12.0 * sign, delta=1.0)
                    self.assertGreater(local[1], -10.0)
                    self.assertLess(local[1], -2.0)
                    lo, hi = config.MEDIAL_ANGLE_RANGE_DEG
                    self.assertTrue(lo <= screw.medial_angle_deg <= hi)
                    plan = screw.plan
                    self.assertIn(plan.length, config.LENGTHS_MM)
                    self.assertIn(plan.diameter, config.DIAMETERS_MM)
                    self.assertLessEqual(plan.diameter, config.DIAMETER_FRACTION * screw.isthmus_width_mm + 1e-9)
                    self.assertLessEqual(plan.length, screw.in_bone_length_mm - config.TIP_MARGIN_MM + 1e-9)
                    # entry on the posterior bone surface, tip inside the body
                    _, shift = snap_entry_to_surface(plan, bone_mask=mask)
                    self.assertLess(abs(shift), 1.0)
                    self.assertTrue(sample_mask_nearest(mask, np.array([plan.tip]))[0])
                    self.assertGreater(((np.asarray(plan.tip) - np.asarray(CENTER)) @ axes)[1], 0.0)
                    # independently measured, the screw stays inside the pedicle wall at the isthmus
                    pedicle = detect_pedicle_isthmus(mask, plan, image=ct, frame=frame.anatomical_frame_for(plan.entry))
                    self.assertGreaterEqual(pedicle.trajectory_match(plan)["outer_clearance_mm"], -0.25)

    def test_missing_canal_is_reported(self):
        block = np.zeros((60, 60, 60), dtype=np.uint8)
        block[10:50, 5:55, 10:50] = 1  # solid block: no spinal canal behind the "body"
        mask = sitk.GetImageFromArray(block)
        frame = estimate_vertebra_frame(mask)
        results = generate_baseline_screws(mask, frame)
        self.assertIsInstance(results["left"], str)
        self.assertIsInstance(results["right"], str)

    def test_cli_output_feeds_the_case_pipeline(self):
        maker, pipeline = _load("make_baseline_screws"), _load("run_case_pipeline")
        ct, mask, _ = synthetic_vertebra_ct(15.0, 0.0, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = sitk.Cast(mask, sitk.sitkUInt8) * 23
            sitk.WriteImage(labels, str(root / "labels.nii.gz"))
            sitk.WriteImage(ct, str(root / "ct.nii.gz"))
            self.assertEqual(maker.main(["--mask", str(root / "labels.nii.gz"), "--labels", "23",
                                         "--out", str(root / "baseline")]), 0)
            with open(root / "baseline" / "baseline_screws.csv") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["screw_id"] for r in rows], ["23_left", "23_right"])
            self.assertTrue(all(r["label"] == "23" for r in rows))
            code = pipeline.main(["--ct", str(root / "ct.nii.gz"), "--mask", str(root / "labels.nii.gz"),
                                  "--screws", str(root / "baseline" / "baseline_screws.csv"),
                                  "--no-qc", "--out", str(root / "run")])
            self.assertEqual(code, 0)
            with open(root / "run" / "case_summary.json") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["screws"]["23_left"]["side"], "left")
            self.assertEqual(summary["screws"]["23_right"]["side"], "right")
            self.assertIn("pedicle", summary["screws"]["23_left"])


if __name__ == "__main__":
    unittest.main()
