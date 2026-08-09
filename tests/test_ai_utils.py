"""Synthetic CT tests for HU processing and coordinate conversion."""

import importlib.util
import unittest


HAS_IMAGING_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "SimpleITK")
)

if HAS_IMAGING_DEPS:
    import numpy as np
    import SimpleITK as sitk

    from pedicle_planning.ai.utils import (
        clip_hu,
        get_ct_array,
        normalize_hu,
        voxel_to_world,
        world_to_voxel,
    )


@unittest.skipUnless(HAS_IMAGING_DEPS, "NumPy and SimpleITK are required")
class SyntheticImagingTests(unittest.TestCase):
    def test_hu_clipping(self):
        volume = np.array([-1000, -200, 400, 2500], dtype=np.float32)
        clipped = clip_hu(volume, hu_min=-200, hu_max=2000)
        np.testing.assert_array_equal(clipped, [-200, -200, 400, 2000])

    def test_hu_normalization(self):
        volume = np.array([-200, 900, 2000], dtype=np.float32)
        normalized = normalize_hu(
            volume,
            hu_min=-200,
            hu_max=2000,
            norm_min=0.0,
            norm_max=1.0,
        )
        np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0])

    def test_voxel_world_round_trip(self):
        voxel = (3, 4, 5)
        origin = (10.0, -2.0, 7.5)
        spacing = (0.5, 1.25, 2.0)
        world = voxel_to_world(voxel, origin, spacing)
        self.assertEqual(world_to_voxel(world, origin, spacing), voxel)

    def test_synthetic_simpleitk_volume(self):
        array = np.arange(4 * 5 * 6, dtype=np.int16).reshape((4, 5, 6))
        image = sitk.GetImageFromArray(array)
        image.SetOrigin((1.0, 2.0, 3.0))
        image.SetSpacing((0.5, 0.75, 1.25))
        np.testing.assert_array_equal(get_ct_array(image), array)
