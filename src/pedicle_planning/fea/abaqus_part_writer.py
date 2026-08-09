"""Abaqus Part-level INP writer for cleaned C3D4 solid meshes.

This module reads raw Gmsh Abaqus INP mesh files, filters out auxiliary
T3D2/CPS3 elements, and writes clean Abaqus Part definitions with Solid
Section assignments. It does not define contact, boundary conditions,
loads, reference points, or solver execution.
"""

from __future__ import annotations

import os


SUPPORTED_SOLID_TYPE = "C3D4"


def _normalize_keyword(line):
    return line.strip().lower().replace(" ", "")


def _get_keyword_option(keyword_line, option_name):
    """Read an Abaqus keyword option such as type=C3D4."""
    option_name = option_name.lower()
    parts = keyword_line.strip().split(",")
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip().lower() == option_name:
            return value.strip()
    return None


def _parse_int_list(line, source_path, line_number):
    values = []
    for item in line.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except Exception as exc:
            raise ValueError(
                "Cannot parse integer at {0}:{1}: {2}".format(
                    source_path, line_number, line
                )
            ) from exc
    return values


def _parse_float(value, source_path, line_number):
    try:
        return float(value.strip())
    except Exception as exc:
        raise ValueError(
            "Cannot parse coordinate at {0}:{1}: {2}".format(
                source_path, line_number, value
            )
        ) from exc


def parse_gmsh_abaqus_inp(source_path):
    """Parse nodes and C3D4 elements from a raw Gmsh Abaqus INP file."""
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError("Mesh inp not found: {0}".format(source_path))

    nodes = {}
    c3d4_elements = []
    skipped_counts = {}
    current_block = None
    current_element_type = None

    with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                keyword = _normalize_keyword(line)
                if keyword.startswith("*node"):
                    current_block = "node"
                    current_element_type = None
                elif keyword.startswith("*element"):
                    current_block = "element"
                    current_element_type = _get_keyword_option(line, "type")
                    if current_element_type:
                        current_element_type = current_element_type.upper()
                else:
                    current_block = None
                    current_element_type = None
                continue

            if current_block == "node":
                parts = [item.strip() for item in line.split(",")]
                if len(parts) < 4:
                    raise ValueError(
                        "Invalid node line at {0}:{1}: {2}".format(
                            source_path, line_number, line
                        )
                    )
                node_id = int(parts[0])
                if node_id in nodes:
                    raise ValueError(
                        "Duplicate node id {0} at {1}:{2}".format(
                            node_id, source_path, line_number
                        )
                    )
                nodes[node_id] = (
                    _parse_float(parts[1], source_path, line_number),
                    _parse_float(parts[2], source_path, line_number),
                    _parse_float(parts[3], source_path, line_number),
                )
            elif current_block == "element":
                values = _parse_int_list(line, source_path, line_number)
                if not values:
                    continue
                element_type = current_element_type or "UNKNOWN"
                if element_type == SUPPORTED_SOLID_TYPE:
                    if len(values) < 5:
                        raise ValueError(
                            "Invalid C3D4 line at {0}:{1}: {2}".format(
                                source_path, line_number, line
                            )
                        )
                    element_id = values[0]
                    connectivity = values[1:5]
                    c3d4_elements.append((element_id, connectivity))
                else:
                    skipped_counts[element_type] = skipped_counts.get(element_type, 0) + 1

    used_node_ids = set()
    for _, connectivity in c3d4_elements:
        for node_id in connectivity:
            if node_id not in nodes:
                raise ValueError(
                    "C3D4 element references missing node id {0} in {1}".format(
                        node_id, source_path
                    )
                )
            used_node_ids.add(node_id)

    return {
        "source_path": source_path,
        "nodes": nodes,
        "c3d4_elements": c3d4_elements,
        "used_node_ids": used_node_ids,
        "skipped_counts": skipped_counts,
    }


def _format_float(value):
    return "{0:.12g}".format(float(value))


def _write_atomic(output_path, text):
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, output_path)


def _validate_part_text(text, part_name, solid_elset):
    required = [
        "*Part, name={0}".format(part_name),
        "*Node",
        "*Element, type=C3D4, elset={0}".format(solid_elset),
        "*Solid Section, elset={0}".format(solid_elset),
        "*End Part",
    ]
    lower_text = text.lower()
    missing = [item for item in required if item.lower() not in lower_text]
    forbidden = ["t3d2", "cps3", "*heading", "*assembly", "*step"]
    found_forbidden = [item for item in forbidden if item in lower_text]
    return missing, found_forbidden


def write_part_from_gmsh_mesh(
    source_mesh_path,
    output_part_path,
    part_name,
    solid_elset,
    material_name,
):
    """Generate one clean Abaqus Part INP from a raw Gmsh mesh INP."""
    parsed = parse_gmsh_abaqus_inp(source_mesh_path)
    nodes = parsed["nodes"]
    c3d4_elements = parsed["c3d4_elements"]
    used_node_ids = parsed["used_node_ids"]
    skipped_counts = parsed["skipped_counts"]

    if not c3d4_elements:
        raise ValueError("No C3D4 solid elements found in {0}".format(source_mesh_path))

    sorted_old_node_ids = sorted(used_node_ids)
    old_to_new_node = {old_id: idx + 1 for idx, old_id in enumerate(sorted_old_node_ids)}

    lines = [
        "** Clean Abaqus Part generated from: {0}".format(source_mesh_path),
        "** Only C3D4 solid elements are included in this clean Part file.",
        "*Part, name={0}".format(part_name),
        "*Node",
    ]

    for old_id in sorted_old_node_ids:
        x, y, z = nodes[old_id]
        lines.append(
            "{0}, {1}, {2}, {3}".format(
                old_to_new_node[old_id],
                _format_float(x),
                _format_float(y),
                _format_float(z),
            )
        )

    lines.append("*Element, type=C3D4, elset={0}".format(solid_elset))
    for new_element_id, (_, connectivity) in enumerate(c3d4_elements, start=1):
        new_conn = [old_to_new_node[node_id] for node_id in connectivity]
        lines.append(
            "{0}, {1}, {2}, {3}, {4}".format(
                new_element_id, new_conn[0], new_conn[1], new_conn[2], new_conn[3]
            )
        )

    lines.extend([
        "*Solid Section, elset={0}, material={1}".format(solid_elset, material_name),
        ",",
        "*End Part",
        "",
    ])
    text = "\n".join(lines)

    missing, forbidden = _validate_part_text(text, part_name, solid_elset)
    if missing:
        raise ValueError("Part validation failed, missing: {0}".format(missing))
    if forbidden:
        raise ValueError("Part validation failed, forbidden keywords found: {0}".format(forbidden))

    referenced_nodes = set()
    for _, connectivity in c3d4_elements:
        for old_id in connectivity:
            referenced_nodes.add(old_to_new_node[old_id])
    output_node_ids = set(range(1, len(sorted_old_node_ids) + 1))
    if not referenced_nodes.issubset(output_node_ids):
        raise ValueError("C3D4 connectivity references missing output nodes")

    element_ids = list(range(1, len(c3d4_elements) + 1))
    if len(element_ids) != len(set(element_ids)):
        raise ValueError("Duplicate output element ids detected")

    os.makedirs(os.path.dirname(output_part_path), exist_ok=True)
    _write_atomic(output_part_path, text)

    return {
        "status": "generated",
        "source_mesh_path": source_mesh_path,
        "part_path": output_part_path,
        "part_name": part_name,
        "solid_elset": solid_elset,
        "material_name": material_name,
        "parsed_node_count": len(nodes),
        "part_node_count": len(sorted_old_node_ids),
        "c3d4_count": len(c3d4_elements),
        "node_id_min": 1,
        "node_id_max": len(sorted_old_node_ids),
        "element_id_min": 1,
        "element_id_max": len(c3d4_elements),
        "skipped_t3d2_count": int(skipped_counts.get("T3D2", 0)),
        "skipped_cps3_count": int(skipped_counts.get("CPS3", 0)),
        "skipped_element_counts": dict(skipped_counts),
        "section_assigned": True,
        "validation_status": "success",
    }


def generate_abaqus_part_files(
    bone_mesh_path=None,
    screw_mesh_path=None,
    output_dir="result/fea_mesh",
):
    """Generate clean Bone/Screw Part files from raw Gmsh mesh INP files."""
    print("[ABAQUS-PART] source bone mesh:", bone_mesh_path)
    print("[ABAQUS-PART] source screw mesh:", screw_mesh_path)

    os.makedirs(output_dir, exist_ok=True)
    bone_part_path = os.path.join(output_dir, "bone_part.inp")
    screw_part_path = os.path.join(output_dir, "screw_part.inp")

    result = {
        "part_generation_status": "not_started",
        "bone_part_path": None,
        "screw_part_path": None,
        "bone_part_node_count": 0,
        "bone_part_c3d4_count": 0,
        "screw_part_node_count": 0,
        "screw_part_c3d4_count": 0,
        "bone_section_assigned": False,
        "screw_section_assigned": False,
        "abaqus_part_assembly_status": "not_started",
        "bone_part": None,
        "screw_part": None,
        "error_message": None,
    }

    try:
        bone_info = write_part_from_gmsh_mesh(
            source_mesh_path=bone_mesh_path,
            output_part_path=bone_part_path,
            part_name="Bone",
            solid_elset="BoneSolid",
            material_name="Bone_CT_HU_Based",
        )
        screw_info = write_part_from_gmsh_mesh(
            source_mesh_path=screw_mesh_path,
            output_part_path=screw_part_path,
            part_name="Screw",
            solid_elset="ScrewSolid",
            material_name="Screw_Titanium",
        )

        result.update({
            "part_generation_status": "generated",
            "bone_part_path": bone_part_path,
            "screw_part_path": screw_part_path,
            "bone_part_node_count": bone_info["part_node_count"],
            "bone_part_c3d4_count": bone_info["c3d4_count"],
            "screw_part_node_count": screw_info["part_node_count"],
            "screw_part_c3d4_count": screw_info["c3d4_count"],
            "bone_section_assigned": bool(bone_info["section_assigned"]),
            "screw_section_assigned": bool(screw_info["section_assigned"]),
            "abaqus_part_assembly_status": "generated",
            "bone_part": bone_info,
            "screw_part": screw_info,
        })

        skipped_t3d2 = bone_info["skipped_t3d2_count"] + screw_info["skipped_t3d2_count"]
        skipped_cps3 = bone_info["skipped_cps3_count"] + screw_info["skipped_cps3_count"]
        print("[ABAQUS-PART] bone nodes parsed:", bone_info["parsed_node_count"])
        print("[ABAQUS-PART] bone C3D4 elements parsed:", bone_info["c3d4_count"])
        print("[ABAQUS-PART] screw nodes parsed:", screw_info["parsed_node_count"])
        print("[ABAQUS-PART] screw C3D4 elements parsed:", screw_info["c3d4_count"])
        print("[ABAQUS-PART] skipped T3D2 elements:", skipped_t3d2)
        print("[ABAQUS-PART] skipped CPS3 elements:", skipped_cps3)
        print("[ABAQUS-PART] bone part path:", bone_part_path)
        print("[ABAQUS-PART] screw part path:", screw_part_path)
        print("[ABAQUS-PART] bone section assigned:", result["bone_section_assigned"])
        print("[ABAQUS-PART] screw section assigned:", result["screw_section_assigned"])
        print("[ABAQUS-PART] assembly validation: success")
        print("[ABAQUS-PART] final status: generated")
        return result
    except Exception as exc:
        result["part_generation_status"] = "failed"
        result["abaqus_part_assembly_status"] = "failed"
        result["error_message"] = str(exc)
        print("[ABAQUS-PART] assembly validation: failed")
        print("[ABAQUS-PART] final status: failed")
        print("[ABAQUS-PART] error:", exc)
        return result
