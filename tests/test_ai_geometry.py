"""Synthetic PyVista geometry tests."""

import importlib.util
import math
import unittest


HAS_GEOMETRY_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "SimpleITK", "pyvista", "vtk")
)

if HAS_GEOMETRY_DEPS:
    import numpy as np
    import pyvista as pv

    from pedicle_planning.ai.geometry import (
        distance_point_to_mesh,
        distance_point_to_point,
        get_mesh_normal_at_point,
    )


@unittest.skipUnless(HAS_GEOMETRY_DEPS, "NumPy, SimpleITK, PyVista, and VTK are required")
class SyntheticGeometryTests(unittest.TestCase):
    def test_two_point_distance(self):
        self.assertAlmostEqual(distance_point_to_point((0, 0, 0), (1, 2, 2)), 3.0)

    def test_point_to_sphere_mesh_distance(self):
        sphere = pv.Sphere(radius=1.0, center=(0.0, 0.0, 0.0))
        distance = distance_point_to_mesh((2.0, 0.0, 0.0), sphere)
        self.assertAlmostEqual(distance, 1.0, delta=0.08)

    def test_sphere_surface_normal_is_unit_and_outward(self):
        sphere = pv.Sphere(
            radius=1.0,
            center=(0.0, 0.0, 0.0),
            theta_resolution=60,
            phi_resolution=60,
        )
        normal = np.asarray(get_mesh_normal_at_point(sphere, (1.0, 0.0, 0.0)))
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=6)
        self.assertGreater(float(normal[0]), 0.8)
        self.assertTrue(math.isfinite(float(normal.sum())))
