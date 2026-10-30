"""Tests of the local CT/mask input checker."""

import importlib.util
import unittest
from pathlib import Path


HAS_IMAGING_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "SimpleITK"))
SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "clinical" / "inspect_inputs.py"

if HAS_IMAGING_DEPS:
    import numpy as np
    import SimpleITK as sitk


def _load_script():
    spec = importlib.util.spec_from_file_location("inspect_inputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ct(values, spacing=(0.7, 0.7, 0.8)):
    image = sitk.GetImageFromArray(values)
    image.SetSpacing(spacing)
    return image


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy and SimpleITK are required")
class InspectInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script()
        hu = np.full((20, 30, 30), -1000, dtype=np.int16)
        hu[5:15, 10:20, 10:20] = 250
        hu[5:15, 10:20, 10:12] = 1200
        cls.hu = hu

    def test_hu_ct_passes(self):
        lines, warnings = self.script.describe_ct(_ct(self.hu))
        self.assertEqual(warnings, [])
        self.assertTrue(any("orientation code" in line and "LPS" in line for line in lines))

    def test_raw_unsigned_values_are_flagged(self):
        raw = (self.hu.astype(np.int32) + 1024).astype(np.uint16)
        _, warnings = self.script.describe_ct(_ct(raw))
        self.assertTrue(any("not be HU" in w for w in warnings))
        self.assertTrue(any("unsigned" in w for w in warnings))

    def test_thick_slices_are_flagged(self):
        _, warnings = self.script.describe_ct(_ct(self.hu, spacing=(0.7, 0.7, 3.0)))
        self.assertTrue(any("slice spacing" in w for w in warnings))

    def test_mask_overlap_and_labels(self):
        ct = _ct(self.hu)
        labels = np.zeros(self.hu.shape, dtype=np.uint8)
        labels[5:15, 10:20, 10:20] = 28
        mask = sitk.GetImageFromArray(labels)
        mask.CopyInformation(ct)
        lines, warnings = self.script.describe_mask(mask, ct)
        self.assertEqual(warnings, [])
        self.assertTrue(any(line.startswith("labels") and "28:" in line for line in lines))
        self.assertTrue(any("same grid as CT      : True" in line for line in lines))
        mask.SetOrigin((500.0, 0.0, 0.0))
        _, warnings = self.script.describe_mask(mask, ct)
        self.assertTrue(any("outside the CT" in w for w in warnings))

    def test_main_exit_code(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ct.nii.gz")
            sitk.WriteImage(_ct(self.hu), path)
            self.assertEqual(self.script.main(["--ct", path]), 0)


if __name__ == "__main__":
    unittest.main()
