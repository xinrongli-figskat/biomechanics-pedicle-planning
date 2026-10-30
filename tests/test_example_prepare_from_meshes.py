"""Tests of converting planning-system meshes (bone STL, positioned screws) into pipeline inputs."""

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
SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "clinical" / "prepare_from_meshes.py"

if HAS_IMAGING_DEPS:
    import numpy as np
    import pyvista as pv
    import SimpleITK as sitk

    from synthetic_vertebra import CENTER, synthetic_vertebra_ct


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_from_meshes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _component_count(mask_array):
    return int(sitk.GetArrayFromImage(sitk.ConnectedComponent(
        sitk.GetImageFromArray(mask_array.astype(np.uint8)))).max())


def _surface(mask_array, spacing=1.0):
    """Closed surface of a (z, y, x) mask with identity direction and zero origin."""
    padded = np.pad(mask_array.astype(np.float32), 1)
    grid = pv.ImageData(dimensions=padded.shape[::-1], spacing=(spacing,) * 3,
                        origin=(-spacing,) * 3)
    grid.point_data["m"] = padded.ravel()
    return grid.contour([0.5], scalars="m").triangulate().clean()


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class PrepareFromMeshesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        ct, mask, axes = synthetic_vertebra_ct(15.0, 0.0, 0.0)
        bone = sitk.GetArrayFromImage(mask) > 0
        # a second, separate "vertebra" block above the first one
        bone[96:116, 40:80, 40:80] = True
        sitk.WriteImage(ct, str(root / "ct.nii.gz"))
        _surface(bone).save(str(root / "bone_lps.stl"))
        ras = _surface(bone)
        ras.points = np.asarray(ras.points) * np.array([-1.0, -1.0, 1.0])
        ras.save(str(root / "bone_ras.stl"))

        entry = np.asarray(CENTER) + axes @ np.array([13.0, -22.0, 0.0])
        tip = np.asarray(CENTER) + axes @ np.array([8.0, 18.0, 0.0])
        direction = (tip - entry) / np.linalg.norm(tip - entry)
        length = float(np.linalg.norm(tip - entry))
        screw = pv.Cylinder(center=(entry + tip) / 2, direction=direction, radius=2.5, height=length,
                            resolution=48).triangulate()
        screw.save(str(root / "screw_lps.stl"))
        template = pv.Cylinder(center=(-500.0, -500.0, 0.0), radius=2.4, height=35.0).triangulate()
        template.save(str(root / "tmpl_x9.vtk"))
        cls.root, cls.entry, cls.tip, cls.length = root, entry, tip, length

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, bone_name, out):
        argv = ["--ct", str(self.root / "ct.nii.gz"), "--bone-mesh", str(self.root / bone_name),
                "--screw-mesh", str(self.root / "screw_lps.stl"), "--screw-mesh", str(self.root / "tmpl_x9.vtk"),
                "--out", str(self.root / out)]
        self.assertEqual(self.script.main(argv), 0)
        with open(self.root / out / "prepare_report.json") as handle:
            report = json.load(handle)
        with open(self.root / out / "screws_from_mesh.csv") as handle:
            rows = list(csv.DictReader(handle))
        labels = sitk.ReadImage(str(self.root / out / "vertebrae_from_mesh.nii.gz"))
        return report, rows, labels

    def test_bone_mesh_becomes_separate_vertebra_labels(self):
        report, _, labels = self._run("bone_lps.stl", "out_lps")
        self.assertEqual(report["bone_meshes"][0]["coords"], "lps")
        self.assertEqual(len(report["labels"]), 2)
        array = sitk.GetArrayViewFromImage(labels)
        _, mask, _ = synthetic_vertebra_ct(15.0, 0.0, 0.0)
        reference = sitk.GetArrayViewFromImage(mask) > 0
        vertebra = array == 2  # labels run cranial to caudal; the extra block sits above the vertebra
        dice = 2 * np.count_nonzero(vertebra & reference) / (np.count_nonzero(vertebra) + np.count_nonzero(reference))
        self.assertGreater(dice, 0.95)

    def test_screw_entry_tip_diameter_and_label(self):
        report, rows, _ = self._run("bone_lps.stl", "out_screw")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["screw_id"], "screw1")
        self.assertEqual(row["label"], "2")
        entry = np.array([float(row[f"entry_{c}"]) for c in "xyz"])
        tip = np.array([float(row[f"tip_{c}"]) for c in "xyz"])
        self.assertLess(np.linalg.norm(entry - self.entry), 1.0)
        self.assertLess(np.linalg.norm(tip - self.tip), 1.0)
        self.assertAlmostEqual(float(row["diameter"]), 5.0, delta=0.3)
        self.assertAlmostEqual(float(row["length"]), self.length, delta=1.0)
        template = report["screws"][1]
        self.assertIn("skipped", template)

    def test_ras_meshes_are_detected(self):
        report, rows, labels = self._run("bone_ras.stl", "out_ras")
        self.assertEqual(report["bone_meshes"][0]["coords"], "ras")
        self.assertEqual(len(report["labels"]), 2)
        self.assertEqual(rows[0]["label"], "2")

    def test_screw_head_is_removed(self):
        direction = (self.tip - self.entry) / np.linalg.norm(self.tip - self.entry)
        head_end = self.entry - 12.0 * direction
        shank = pv.Line(self.entry, self.tip, resolution=60).tube(radius=2.5, n_sides=32, capping=True)
        head = pv.Line(head_end, self.entry, resolution=20).tube(radius=5.25, n_sides=32, capping=True)
        (shank + head).triangulate().save(str(self.root / "screw_head.stl"))
        argv = ["--ct", str(self.root / "ct.nii.gz"), "--bone-mesh", str(self.root / "bone_lps.stl"),
                "--screw-mesh", str(self.root / "screw_head.stl"), "--out", str(self.root / "out_head")]
        self.assertEqual(self.script.main(argv), 0)
        with open(self.root / "out_head" / "prepare_report.json") as handle:
            screw = json.load(handle)["screws"][0]
        self.assertTrue(screw["head_found"])
        self.assertAlmostEqual(screw["head_length_mm"], 12.0, delta=1.5)
        self.assertAlmostEqual(screw["length_mm"], self.length, delta=1.5)
        self.assertAlmostEqual(screw["diameter_mm"], 5.0, delta=0.3)
        self.assertLess(np.linalg.norm(np.asarray(screw["entry"]) - self.entry), 1.5)
        self.assertLess(np.linalg.norm(np.asarray(screw["tip"]) - self.tip), 1.5)

    def test_touching_vertebrae_are_split_cranial_first(self):
        _, mask, _ = synthetic_vertebra_ct(15.0, 0.0, 0.0)
        lower = sitk.GetArrayFromImage(mask) > 0
        upper = np.roll(lower, 32, axis=0)  # same vertebra 32 mm more cranial (z index = axis 0)
        bridge = np.zeros_like(lower)
        # 4 x 4 mm bony bridge from the lower body's top (z 72) to the upper body's bottom (z 80);
        # the body centre of the 15-degree-rotated phantom lies at about x 57, y 48
        bridge[70:82, 46:50, 55:59] = True
        self.assertEqual(_component_count(lower | upper | bridge), 1)
        fused = lower | upper | bridge
        _surface(fused).save(str(self.root / "fused.stl"))
        out = self.root / "out_fused"
        argv = ["--ct", str(self.root / "ct.nii.gz"), "--bone-mesh", str(self.root / "fused.stl"),
                "--vertebrae", "2", "--out", str(out)]
        self.assertEqual(self.script.main(argv), 0)
        with open(out / "prepare_report.json") as handle:
            report = json.load(handle)
        self.assertTrue(report["split_touching_vertebrae"])
        self.assertEqual(len(report["labels"]), 2)
        self.assertGreater(report["labels"]["1"]["centroid"][2], report["labels"]["2"]["centroid"][2])
        labels = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "vertebrae_from_mesh.nii.gz")))
        for value, reference in ((1, upper), (2, lower)):
            part = labels == value
            dice = 2 * np.count_nonzero(part & reference) / (np.count_nonzero(part) + np.count_nonzero(reference))
            self.assertGreater(dice, 0.9)
        self.assertTrue(bridge.any())

    def test_outputs_do_not_contain_input_names(self):
        self._run("bone_lps.stl", "out_names")
        for name in ("prepare_report.json", "screws_from_mesh.csv"):
            text = (self.root / "out_names" / name).read_text()
            for token in ("bone_lps", "screw_lps", "tmpl_x9", str(self.root)):
                self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
