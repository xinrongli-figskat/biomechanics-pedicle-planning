"""Smoke test of the local end-to-end case script on a synthetic vertebra."""

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
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "clinical" / "run_case_pipeline.py"

if HAS_IMAGING_DEPS:
    import numpy as np
    import SimpleITK as sitk

    from synthetic_vertebra import CENTER, synthetic_vertebra_ct


def _load_script():
    spec = importlib.util.spec_from_file_location("run_case_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _physical(axes, l, a, s):
    return np.asarray(CENTER) + axes @ np.array([l, a, s], dtype=np.float64)


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class CasePipelineScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        ct, mask, axes = synthetic_vertebra_ct(15.0, 0.0, 0.0)
        # label map: vertebra stored as label 27
        labels = sitk.Cast(mask, sitk.sitkUInt8) * 27
        sitk.WriteImage(ct, str(root / "ct.nii.gz"))
        sitk.WriteImage(labels, str(root / "labels.nii.gz"))
        cls.root, cls.axes = root, axes
        # entry on the posterior lamina surface, axis through the pedicle into the body
        screws = []
        for sid, side in (("L_left", 1.0), ("L_right", -1.0)):
            entry = _physical(axes, 13.0 * side, -22.0, 0.0)
            tip = _physical(axes, 8.0 * side, 18.0, 0.0)
            screws.append((sid, entry, tip))
        cls.screws = screws
        cls._write_screws(root / "screws_lps.csv", lambda p: p)
        cls._write_screws(root / "screws_ras.csv", lambda p: (-p[0], -p[1], p[2]))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def _write_screws(cls, path, convert):
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["screw_id", "entry_x", "entry_y", "entry_z", "tip_x", "tip_y", "tip_z", "diameter"])
            for sid, entry, tip in cls.screws:
                writer.writerow([sid, *convert(entry), *convert(tip), 3.0])

    def _run(self, out, screws="screws_lps.csv", *extra):
        argv = ["--ct", str(self.root / "ct.nii.gz"), "--mask", str(self.root / "labels.nii.gz"),
                "--label", "27", "--screws", str(self.root / screws), "--out", str(self.root / out),
                *extra]
        if not HAS_MATPLOTLIB and "--no-qc" not in extra:
            argv.append("--no-qc")
        code = self.script.main(argv)
        with open(self.root / out / "case_summary.json") as handle:
            return code, json.load(handle)

    def test_end_to_end_outputs(self):
        code, summary = self._run("out_demo", "screws_lps.csv", "--risk", "demo")
        self.assertEqual(code, 0)
        vertebra = summary["vertebrae"]["label_27"]
        self.assertEqual(vertebra["frame"]["method"], "symmetry")
        self.assertAlmostEqual(vertebra["frame"]["angles_to_image_axes"]["axial_rotation_deg"], 15.0, delta=2.0)
        self.assertAlmostEqual(vertebra["body_bmd"]["vertebra_midbody_mean"], 140.0, delta=3.0)
        for sid, expected_side in (("L_left", "left"), ("L_right", "right")):
            record = summary["screws"][sid]
            self.assertNotIn("error", record)
            self.assertEqual(record["side"], expected_side)
            self.assertIn("pedicle", record)
            self.assertGreater(record["pedicle"]["outer_min_width_mm"], 6.0)
            self.assertLess(record["pedicle"]["outer_min_width_mm"], 10.0)
            opt = record["optimization"]
            self.assertEqual(opt["n_candidates"], 90)
            self.assertIsNotNone(opt["baseline_risk"])
            self.assertTrue((self.root / "out_demo" / f"candidates_{sid}.csv").exists())
            self.assertTrue((self.root / "out_demo" / f"features_{sid}_baseline.json").exists())
            with open(self.root / "out_demo" / f"candidates_{sid}.csv") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 90)
            self.assertEqual(rows[0]["is_baseline"], "True")
            self.assertIn("pedicle_interface_medial_median", rows[0])
            if HAS_MATPLOTLIB:
                self.assertTrue((self.root / "out_demo" / f"qc_{sid}.png").exists())
        self.assertNotIn(str(self.root), json.dumps(summary))  # input paths are not recorded

    def test_ras_coordinates_match_lps(self):
        _, lps = self._run("out_lps", "screws_lps.csv", "--no-qc")
        _, ras = self._run("out_ras", "screws_ras.csv", "--no-qc", "--coords", "ras")
        for sid in ("L_left", "L_right"):
            a = lps["screws"][sid]["baseline_key_features"]
            b = ras["screws"][sid]["baseline_key_features"]
            for key, value in a.items():
                if value is None:
                    self.assertIsNone(b[key])
                else:
                    self.assertAlmostEqual(value, b[key], places=6)

    def test_without_risk_model_reports_feasibility_only(self):
        _, summary = self._run("out_none", "screws_lps.csv", "--no-qc")
        opt = summary["screws"]["L_left"]["optimization"]
        self.assertIsNone(opt["best_risk"])
        self.assertGreater(opt["n_feasible"], 0)
        self.assertEqual(summary["options"]["risk"], "none")

    def test_qc_write_failure_does_not_fail_the_screw(self):
        original = self.script.write_qc_figure

        def locked(*args, **kwargs):
            raise OSError(22, "Invalid argument")

        self.script.write_qc_figure = locked
        try:
            code, summary = self._run("out_locked", "screws_lps.csv", "--risk", "demo")
        finally:
            self.script.write_qc_figure = original
        if HAS_MATPLOTLIB:
            self.assertEqual(code, 0)
            record = summary["screws"]["L_left"]
            self.assertNotIn("error", record)
            self.assertFalse(record["qc_figure"])
            self.assertIn("qc_error", record)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            self.script.binary_mask(sitk.Image(5, 5, 5, sitk.sitkUInt8), 3)
        with self.assertRaises(ValueError):
            self.script.load_risk_function("not_a_spec")
        bad = self.root / "bad.csv"
        bad.write_text("screw_id,entry_x\nA,1\n")
        with self.assertRaises(ValueError):
            self.script.read_screws(str(bad), "lps", sitk.Image(5, 5, 5, sitk.sitkFloat32))


if __name__ == "__main__":
    unittest.main()
