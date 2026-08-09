"""Programmatically generated C3D4 and Abaqus writer tests."""

from pathlib import Path
import tempfile
import unittest

from pedicle_planning.fea.abaqus_mesh_assembly import build_abaqus_mesh_assembly
from pedicle_planning.fea.abaqus_part_writer import (
    parse_gmsh_abaqus_inp,
    write_part_from_gmsh_mesh,
)


MINIMAL_C3D4 = """*Heading
** synthetic tetrahedron
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*Element, type=C3D4
1, 1, 2, 3, 4
"""


class SyntheticAbaqusTests(unittest.TestCase):
    def test_minimal_c3d4_parse_and_part_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "synthetic_mesh.inp"
            output = Path(tmp) / "synthetic_part.inp"
            source.write_text(MINIMAL_C3D4, encoding="utf-8")

            parsed = parse_gmsh_abaqus_inp(str(source))
            self.assertEqual(len(parsed["nodes"]), 4)
            self.assertEqual(len(parsed["c3d4_elements"]), 1)

            report = write_part_from_gmsh_mesh(
                str(source),
                str(output),
                part_name="SyntheticPart",
                solid_elset="SyntheticSolid",
                material_name="SyntheticMaterial",
            )
            text = output.read_text(encoding="utf-8")
            self.assertEqual(report["validation_status"], "success")
            self.assertIn("*Element, type=C3D4", text)
            self.assertIn("*Solid Section", text)

    def test_abaqus_assembly_from_two_synthetic_meshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bone = root / "bone_mesh.inp"
            screw = root / "screw_mesh.inp"
            bone.write_text(MINIMAL_C3D4, encoding="utf-8")
            screw.write_text(MINIMAL_C3D4, encoding="utf-8")

            report = build_abaqus_mesh_assembly(
                bone_mesh_path=str(bone),
                screw_mesh_path=str(screw),
                template_dir=str(root),
            )
            self.assertEqual(report["status"], "generated")
            self.assertTrue(report["bone_include"])
            self.assertTrue(report["screw_include"])
            self.assertEqual(report["include_block"].count("*INCLUDE"), 2)
