"""Abaqus 网格装配 include 语句生成模块。

本模块检查原始 bone/screw mesh inp 文件，生成干净的 Part 级 INP 文件，
并输出主 Abaqus 模板中使用的 *INCLUDE 语句。不定义接触、边界条件或载荷。
"""

from __future__ import annotations

import os

try:
    from funcs.fea.abaqus_part_writer import generate_abaqus_part_files
except Exception:
    try:
        from .abaqus_part_writer import generate_abaqus_part_files
    except Exception:
        generate_abaqus_part_files = None


def _normalize_path_for_inp(path):
    """将路径转换为 Abaqus inp 中更稳定的斜杠格式。"""
    if path is None:
        return None
    return str(path).replace("\\", "/")


def _relative_or_basename(mesh_path, template_dir=None):
    """生成相对 template 输出目录的 include 路径。"""
    if not mesh_path:
        return None
    try:
        if template_dir:
            return _normalize_path_for_inp(os.path.relpath(mesh_path, template_dir))
    except Exception:
        pass
    return _normalize_path_for_inp(mesh_path)


def _part_output_dir_from_mesh(bone_mesh_path, screw_mesh_path, template_dir=None):
    for path in [bone_mesh_path, screw_mesh_path]:
        if path:
            try:
                return os.path.dirname(path) or template_dir or "result/fea_mesh"
            except Exception:
                pass
    if template_dir:
        return os.path.join(template_dir, "fea_mesh")
    return "result/fea_mesh"


def build_abaqus_mesh_assembly(
    bone_mesh_path=None,
    screw_mesh_path=None,
    template_dir=None,
):
    """检查 mesh inp 文件、生成 Part 文件并生成 Abaqus *INCLUDE 块。

    返回包含 Part 生成状态、include 路径和验证摘要的字典。
    """
    bone_raw_exists = bool(bone_mesh_path and os.path.exists(bone_mesh_path))
    screw_raw_exists = bool(screw_mesh_path and os.path.exists(screw_mesh_path))

    part_info = {
        "part_generation_status": "failed",
        "bone_part_path": None,
        "screw_part_path": None,
        "bone_part_node_count": 0,
        "bone_part_c3d4_count": 0,
        "screw_part_node_count": 0,
        "screw_part_c3d4_count": 0,
        "bone_section_assigned": False,
        "screw_section_assigned": False,
        "abaqus_part_assembly_status": "failed",
        "error_message": None,
    }

    if bone_raw_exists and screw_raw_exists and generate_abaqus_part_files is not None:
        part_output_dir = _part_output_dir_from_mesh(bone_mesh_path, screw_mesh_path, template_dir)
        part_info = generate_abaqus_part_files(
            bone_mesh_path=bone_mesh_path,
            screw_mesh_path=screw_mesh_path,
            output_dir=part_output_dir,
        )
    elif generate_abaqus_part_files is None:
        part_info["error_message"] = "Abaqus part writer module unavailable"
    else:
        part_info["error_message"] = "Raw mesh inp files are missing"

    bone_part_path = part_info.get("bone_part_path")
    screw_part_path = part_info.get("screw_part_path")
    bone_exists = bool(bone_part_path and os.path.exists(bone_part_path))
    screw_exists = bool(screw_part_path and os.path.exists(screw_part_path))

    bone_include_path = _relative_or_basename(bone_part_path, template_dir) if bone_exists else None
    screw_include_path = _relative_or_basename(screw_part_path, template_dir) if screw_exists else None

    include_lines = [
        "** ----------------------------------------------------------------",
        "** ABAQUS PART INCLUDE DEFINITIONS",
        "** Raw Gmsh mesh files are preserved for traceability; clean Part files are included below.",
        "** ----------------------------------------------------------------",
    ]
    if bone_exists:
        include_lines.append("*INCLUDE, INPUT={0}".format(bone_include_path))
    else:
        include_lines.append("** Bone Part include failed: {0}".format(bone_part_path))

    if screw_exists:
        include_lines.append("*INCLUDE, INPUT={0}".format(screw_include_path))
    else:
        include_lines.append("** Screw Part include failed: {0}".format(screw_part_path))

    if bone_exists and screw_exists:
        status = "generated"
    elif bone_exists or screw_exists:
        status = "partial"
    else:
        status = "failed"

    print("[ABAQUS-MESH-ASSEMBLY] bone mesh include:", "success" if bone_exists else "failed")
    print("[ABAQUS-MESH-ASSEMBLY] screw mesh include:", "success" if screw_exists else "failed")

    result = {
        "bone_include": bone_exists,
        "screw_include": screw_exists,
        "status": status,
        "bone_include_path": bone_include_path,
        "screw_include_path": screw_include_path,
        "bone_mesh_path": bone_mesh_path,
        "screw_mesh_path": screw_mesh_path,
        "bone_part_path": bone_part_path,
        "screw_part_path": screw_part_path,
        "include_block": "\n".join(include_lines),
    }
    result.update(part_info)
    return result
