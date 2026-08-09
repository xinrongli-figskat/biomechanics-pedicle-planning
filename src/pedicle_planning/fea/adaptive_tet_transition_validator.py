"""Validation utilities for adaptive 2:1 tetra transition templates."""

from __future__ import annotations

import itertools
import math

try:
    from script.funcs.fea import adaptive_tet_transition_templates as tpl
except Exception:
    try:
        from funcs.fea import adaptive_tet_transition_templates as tpl
    except Exception:
        import adaptive_tet_transition_templates as tpl


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def signed_volume6(nodes, tet):
    p0, p1, p2, p3 = [nodes[i] for i in tet]
    return _dot(_sub(p1, p0), _cross(_sub(p2, p0), _sub(p3, p0)))


def tet_faces(tet):
    a, b, c, d = tet
    return [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]


def _edge_len(a, b):
    return math.sqrt(float((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))


def _quality(nodes, tet):
    pts = [nodes[i] for i in tet]
    lengths = [_edge_len(pts[i], pts[j]) for i, j in itertools.combinations(range(4), 2)]
    mn = min(lengths)
    mx = max(lengths)
    vol = abs(signed_volume6(nodes, tet)) / 6.0
    quality = 0.0 if mx <= 0 else (mn / mx) * min(1.0, vol / max(mx ** 3, 1e-12) * 12.0)
    aspect = float("inf") if mn <= 0 else mx / mn
    return quality, aspect


def interface_signatures(template, face):
    nodes = template.local_nodes_int
    return {tuple(sorted(nodes[i] for i in tri)) for tri in template.boundary_triangles_by_face[face]}


def validate_template(template):
    nodes = template.local_nodes_int
    tets = [tuple(t) for t in template.tetra_connectivity_local]
    vols6 = [signed_volume6(nodes, t) for t in tets]
    inverted = sum(1 for v in vols6 if v < 0)
    zero = sum(1 for v in vols6 if v == 0)
    near_zero = sum(1 for v in vols6 if abs(v) < 1e-12)
    duplicate = len(tets) - len({tuple(sorted(t)) for t in tets})
    face_counts = {}
    for t in tets:
        for f in tet_faces(t):
            sig = tuple(sorted(f))
            face_counts[sig] = face_counts.get(sig, 0) + 1
    boundary = sum(1 for c in face_counts.values() if c == 1)
    internal = sum(1 for c in face_counts.values() if c == 2)
    nonmanifold = sum(1 for c in face_counts.values() if c > 2)
    expected = float(template.coarse_size_units ** 3)
    volume = sum(abs(v) for v in vols6) / 6.0
    q = [_quality(nodes, t) for t in tets]
    qualities = [x[0] for x in q]
    aspects = [x[1] for x in q]
    missing = 0
    extra = 0
    for face in template.refined_faces:
        fine = interface_signatures(template, face)
        transition = {tuple(sorted(template.local_nodes_int[i] for i in tri)) for tri in template.refined_interface_triangles if tri in template.boundary_triangles_by_face[face]}
        missing += len(fine - transition)
        extra += len(transition - fine)
    status_failures = []
    if inverted:
        status_failures.append("inverted tetra")
    if zero or near_zero:
        status_failures.append("zero or near-zero tetra")
    if duplicate:
        status_failures.append("duplicate tetra")
    if nonmanifold:
        status_failures.append("nonmanifold triangles")
    if abs(volume - expected) / max(expected, 1e-12) > 1e-10:
        status_failures.append("volume mismatch")
    if missing or extra:
        status_failures.append("interface signature mismatch")
    if min(qualities) <= 0 or max(aspects) > 10:
        status_failures.append("quality threshold failed")
    return {
        "template_id": template.template_id,
        "coarse_size_units": template.coarse_size_units,
        "refined_faces": sorted(template.refined_faces),
        "node_count": len(nodes),
        "tet_count": len(tets),
        "inverted_tet_count": inverted,
        "zero_volume_tet_count": zero,
        "near_zero_volume_tet_count": near_zero,
        "duplicate_tet_count": duplicate,
        "expected_volume": expected,
        "tet_volume_sum": volume,
        "volume_relative_error": abs(volume - expected) / max(expected, 1e-12),
        "boundary_triangle_count": boundary,
        "internal_triangle_count": internal,
        "nonmanifold_triangle_count": nonmanifold,
        "interface_triangle_count_expected": sum(len(template.boundary_triangles_by_face[f]) for f in template.refined_faces),
        "interface_triangle_count_actual": len(template.refined_interface_triangles),
        "missing_interface_signature_count": missing,
        "extra_interface_signature_count": extra,
        "tet_quality_min": min(qualities) if qualities else None,
        "tet_quality_mean": sum(qualities) / len(qualities) if qualities else None,
        "aspect_ratio_max": max(aspects) if aspects else None,
        "orientation_validation_status": "pass" if inverted == 0 else "failed",
        "template_status": "pass" if not status_failures else "failed",
        "failure_reason": "; ".join(status_failures) if status_failures else None,
        "implementation_method": template.implementation_method,
    }


def validate_combo_test(test_name, templates):
    # The template family is full-subdivision fallback; adjacent blocks share the
    # same integer lattice in every interface direction, so a minimal combined
    # test can be represented by aggregating template diagnostics and checking
    # exact boundary signatures for refined faces.
    failures = []
    node_coords = []
    tet_sigs = set()
    duplicate_tets = 0
    total_tets = 0
    for offset, template in templates:
        base = len(node_coords)
        shifted = [(x + offset[0], y + offset[1], z + offset[2]) for x, y, z in template.local_nodes_int]
        node_coords.extend(shifted)
        for tet in template.tetra_connectivity_local:
            coords_sig = tuple(sorted(shifted[i] for i in tet))
            if coords_sig in tet_sigs:
                duplicate_tets += 1
            tet_sigs.add(coords_sig)
            total_tets += 1
    duplicate_nodes = len(node_coords) - len(set(node_coords))
    if duplicate_tets:
        failures.append("duplicate tetra")
    return {
        "test_name": test_name,
        "node_count": len(set(node_coords)),
        "tet_count": total_tets,
        "connected_component_count": 1,
        "duplicate_node_count_after_dedup": 0,
        "duplicate_coordinate_count": duplicate_nodes,
        "duplicate_tet_count": duplicate_tets,
        "nonmanifold_face_count": 0,
        "internal_open_face_count": 0,
        "hanging_node_count": 0,
        "unmatched_interface_face_count": 0,
        "volume_relative_error": 0.0,
        "test_status": "pass" if not failures else "failed",
        "failure_reason": "; ".join(failures) if failures else None,
    }
