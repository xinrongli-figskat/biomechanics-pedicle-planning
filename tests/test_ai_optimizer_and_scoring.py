"""Synthetic optimizer and scorer behavior tests."""

import importlib.util
import unittest


HAS_AI_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "SimpleITK", "pyvista", "vtk")
)

if HAS_AI_DEPS:
    import numpy as np
    import pyvista as pv
    import SimpleITK as sitk

    from pedicle_planning.ai.optimizer import EntryPointOptimizer
    from pedicle_planning.ai.scorer import EntryPointScorer


class _QuadraticScorer:
    def score(self, point):
        x, y, z = point
        return -(x * x + y * y + z * z)


@unittest.skipUnless(HAS_AI_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class SyntheticOptimizerAndScoringTests(unittest.TestCase):
    def test_local_optimizer_moves_toward_quadratic_optimum(self):
        initial = (1.5, 0.5, 0.0)
        optimizer = EntryPointOptimizer(_QuadraticScorer(), bone_mesh=None)
        optimized = optimizer.optimize(initial, radius=1.5, step=0.5, iterations=3)
        self.assertLess(np.linalg.norm(optimized), np.linalg.norm(initial))
        self.assertGreater(_QuadraticScorer().score(optimized), _QuadraticScorer().score(initial))

    def test_scoring_returns_finite_normalized_value(self):
        volume = np.full((9, 9, 9), 500, dtype=np.int16)
        image = sitk.GetImageFromArray(volume)
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetSpacing((1.0, 1.0, 1.0))
        mesh = pv.Sphere(radius=3.0, center=(4.0, 4.0, 4.0))
        score = EntryPointScorer(image, mesh).score((6.5, 4.0, 4.0))
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
