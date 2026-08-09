from __future__ import annotations

from dataclasses import dataclass

try:
    from script.funcs.fea import adaptive_green_face_bucket_builder as fb
except Exception:
    import adaptive_green_face_bucket_builder as fb


@dataclass(frozen=True)
class SharedFaceTriangulation:
    shared_face_key: tuple
    shared_rectangle_bbox: tuple
    boundary_nodes: tuple
    subcell_partition: tuple
    triangle_signatures: frozenset
    parity_rule: str
    authority_source: str
    validation_status: str


def shared_face_key_from_bbox(bbox, face):
    axis = fb.FACE_AXIS[face]
    axes = fb.face_axes(face)
    plane = fb.face_const(bbox, face)
    axis_name = ("x", "y", "z")[axis]
    return (
        axis_name,
        int(plane),
        int(bbox[axes[0]]),
        int(bbox[axes[0] + 3]),
        int(bbox[axes[1]]),
        int(bbox[axes[1] + 3]),
    )


def build_required_shared_face_triangulation(shared_face_key, side_a_context=None, side_b_context=None):
    axis_name, plane, u0, u1, v0, v1 = shared_face_key
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]
    axes = tuple(i for i in (0, 1, 2) if i != axis)
    rect = {
        "axis_u": axes[0],
        "axis_v": axes[1],
        "fix_axis": axis,
        "fix_value": plane,
        "u0": u0,
        "u1": u1,
        "v0": v0,
        "v1": v1,
    }
    tris = []
    boundary_nodes = set()
    subcells = []
    for u in range(u0, u1):
        for v in range(v0, v1):
            subcells.append((u, v))
            p00 = fb.make_point(rect, u, v)
            p10 = fb.make_point(rect, u + 1, v)
            p01 = fb.make_point(rect, u, v + 1)
            p11 = fb.make_point(rect, u + 1, v + 1)
            boundary_nodes.update((p00, p10, p01, p11))
            if (u + v + plane) % 2 == 0:
                tris.extend((fb.tri_sig(p00, p10, p11), fb.tri_sig(p00, p11, p01)))
            else:
                tris.extend((fb.tri_sig(p00, p10, p01), fb.tri_sig(p10, p11, p01)))
    return SharedFaceTriangulation(
        shared_face_key=tuple(shared_face_key),
        shared_rectangle_bbox=(u0, u1, v0, v1),
        boundary_nodes=tuple(sorted(boundary_nodes)),
        subcell_partition=tuple(subcells),
        triangle_signatures=frozenset(tris),
        parity_rule="(u_min + v_min + plane_coordinate) mod 2",
        authority_source="global_finest_subcell_shared_face_authority",
        validation_status="pass" if len(tris) == 2 * len(subcells) else "failed",
    )


def face_triangles_for_cell_face(cell_dict, face):
    buckets = fb.build_face_bucket_from_finest_subcell_ownership(cell_dict, None, None)
    return frozenset(buckets[face]["triangles"])
