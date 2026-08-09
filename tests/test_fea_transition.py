"""Pure-math tetrahedral transition template tests."""

import unittest

from pedicle_planning.fea.adaptive_tet_transition_templates import (
    get_transition_template,
)
from pedicle_planning.fea.adaptive_tet_transition_validator import (
    signed_volume6,
    validate_template,
)


class SyntheticTransitionTests(unittest.TestCase):
    def test_minimal_tetrahedron_has_positive_volume(self):
        nodes = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        self.assertEqual(signed_volume6(nodes, (0, 1, 2, 3)), 1)

    def test_transition_template_is_valid(self):
        template = get_transition_template(2, refined_faces={"+x", "-y"})
        report = validate_template(template)
        self.assertTrue(template.supported)
        self.assertEqual(report["template_status"], "pass")
        self.assertEqual(report["inverted_tet_count"], 0)
        self.assertEqual(report["zero_volume_tet_count"], 0)
        self.assertAlmostEqual(report["tet_volume_sum"], report["expected_volume"])
