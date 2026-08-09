"""Reusable integer-coordinate 2:1 tetra transition templates.

The current implementation provides a conservative full-subdivision fallback:
a coarse transition cube is subdivided into 2x2x2 fine cubes and each fine cube
is decomposed into six C3D4 tetrahedra. This is conforming for every refined
face pattern and avoids hanging nodes. It is not the minimum-tet transition
stencil, so callers can account for the extra tetrahedra explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

FACE_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z")
SUPPORTED_FACE_ORIENTATIONS = list(FACE_NAMES)
GRID_UNIT_MM = 0.20


@dataclass
class TransitionTemplate:
    template_id: str
    coarse_size_units: int
    refined_faces: frozenset
    local_nodes_int: list
    tetra_connectivity_local: list
    boundary_triangles_by_face: dict
    refined_interface_triangles: list
    required_midpoint_nodes: list
    required_face_center_nodes: list
    uses_cell_center: bool
    supported: bool
    failure_reason: str | None = None
    implementation_method: str = "full_subdivision_fallback_2x2x2_fine_cubes"
    template_family: str = "full_subdivision_fallback"
    unrefined_interface_triangles: list | None = None
    tet_count: int | None = None
    fallback_used: bool = True


@dataclass
class CellTemplateSelection:
    cell_class: str
    refined_faces: frozenset
    template_id: str
    tet_count: int
    selection_reason: str
    coarse_size_units: int | None = None
    template_family: str | None = None
    production_eligible: bool = True


def all_face_patterns():
    patterns = []
    faces = list(FACE_NAMES)
    for n in range(1, 7):
        for combo in combinations(faces, n):
            patterns.append(frozenset(combo))
    return patterns


def _node_index_map(coarse_size_units: int):
    step = int(coarse_size_units // 2)
    coords = []
    index = {}
    for z in (0, step, coarse_size_units):
        for y in (0, step, coarse_size_units):
            for x in (0, step, coarse_size_units):
                index[(x, y, z)] = len(coords)
                coords.append((x, y, z))
    return coords, index, step


def _cube_corner_indices(index, x0, y0, z0, h):
    return [
        index[(x0, y0, z0)],
        index[(x0 + h, y0, z0)],
        index[(x0, y0 + h, z0)],
        index[(x0 + h, y0 + h, z0)],
        index[(x0, y0, z0 + h)],
        index[(x0 + h, y0, z0 + h)],
        index[(x0, y0 + h, z0 + h)],
        index[(x0 + h, y0 + h, z0 + h)],
    ]


def _positive_cube_tets(c):
    # Six tetrahedra around body diagonal c0-c7. Orientation is fixed later by
    # the validator/generator if a coordinate transform changes handedness.
    return [
        (c[0], c[1], c[3], c[7]),
        (c[0], c[3], c[2], c[7]),
        (c[0], c[2], c[6], c[7]),
        (c[0], c[6], c[4], c[7]),
        (c[0], c[4], c[5], c[7]),
        (c[0], c[5], c[1], c[7]),
    ]


def _face_grid_triangles(index, coarse_size_units: int, face: str):
    h = int(coarse_size_units // 2)
    s = coarse_size_units
    tris = []
    if face in ("+z", "-z"):
        z = s if face == "+z" else 0
        for x0 in (0, h):
            for y0 in (0, h):
                n00 = index[(x0, y0, z)]
                n10 = index[(x0 + h, y0, z)]
                n01 = index[(x0, y0 + h, z)]
                n11 = index[(x0 + h, y0 + h, z)]
                tris.extend([(n00, n10, n11), (n00, n11, n01)])
    elif face in ("+x", "-x"):
        x = s if face == "+x" else 0
        for y0 in (0, h):
            for z0 in (0, h):
                n00 = index[(x, y0, z0)]
                n10 = index[(x, y0 + h, z0)]
                n01 = index[(x, y0, z0 + h)]
                n11 = index[(x, y0 + h, z0 + h)]
                tris.extend([(n00, n10, n11), (n00, n11, n01)])
    else:
        y = s if face == "+y" else 0
        for x0 in (0, h):
            for z0 in (0, h):
                n00 = index[(x0, y, z0)]
                n10 = index[(x0 + h, y, z0)]
                n01 = index[(x0, y, z0 + h)]
                n11 = index[(x0 + h, y, z0 + h)]
                tris.extend([(n00, n10, n11), (n00, n11, n01)])
    return tris


def _midpoints_and_centers(index, coarse_size_units: int, faces):
    h = int(coarse_size_units // 2)
    s = coarse_size_units
    mids = set()
    centers = set()
    for face in faces:
        if face in ("+z", "-z"):
            z = s if face == "+z" else 0
            mids.update([(h, 0, z), (h, s, z), (0, h, z), (s, h, z)])
            centers.add((h, h, z))
        elif face in ("+x", "-x"):
            x = s if face == "+x" else 0
            mids.update([(x, h, 0), (x, h, s), (x, 0, h), (x, s, h)])
            centers.add((x, h, h))
        else:
            y = s if face == "+y" else 0
            mids.update([(h, y, 0), (h, y, s), (0, y, h), (s, y, h)])
            centers.add((h, y, h))
    return [index[p] for p in sorted(mids)], [index[p] for p in sorted(centers)]


def get_transition_template(coarse_size_units: int, refined_faces, orientation_context=None):
    faces = frozenset(refined_faces or [])
    if coarse_size_units not in (2, 4):
        return TransitionTemplate(
            template_id="unsupported_size_{}".format(coarse_size_units),
            coarse_size_units=coarse_size_units,
            refined_faces=faces,
            local_nodes_int=[],
            tetra_connectivity_local=[],
            boundary_triangles_by_face={},
            refined_interface_triangles=[],
            required_midpoint_nodes=[],
            required_face_center_nodes=[],
            uses_cell_center=False,
            supported=False,
            failure_reason="coarse_size_units must be 2 or 4",
        )
    unknown = [f for f in faces if f not in FACE_NAMES]
    if unknown:
        return TransitionTemplate(
            template_id="unsupported_faces",
            coarse_size_units=coarse_size_units,
            refined_faces=faces,
            local_nodes_int=[],
            tetra_connectivity_local=[],
            boundary_triangles_by_face={},
            refined_interface_triangles=[],
            required_midpoint_nodes=[],
            required_face_center_nodes=[],
            uses_cell_center=False,
            supported=False,
            failure_reason="unknown faces: {}".format(unknown),
        )
    nodes, index, h = _node_index_map(coarse_size_units)
    tets = []
    for x0 in (0, h):
        for y0 in (0, h):
            for z0 in (0, h):
                c = _cube_corner_indices(index, x0, y0, z0, h)
                tets.extend(_positive_cube_tets(c))
    boundary = {face: _face_grid_triangles(index, coarse_size_units, face) for face in FACE_NAMES}
    refined_tris = []
    for face in faces:
        refined_tris.extend(boundary[face])
    mids, centers = _midpoints_and_centers(index, coarse_size_units, faces)
    pattern = "none" if not faces else "_".join(sorted(faces))
    return TransitionTemplate(
        template_id="generic_2to1_full_subdivision_s{}_{}".format(coarse_size_units, pattern),
        coarse_size_units=coarse_size_units,
        refined_faces=faces,
        local_nodes_int=nodes,
        tetra_connectivity_local=tets,
        boundary_triangles_by_face=boundary,
        refined_interface_triangles=refined_tris,
        required_midpoint_nodes=mids,
        required_face_center_nodes=centers,
        uses_cell_center=True,
        supported=True,
    )



def _face_coord(face: str, s: int):
    if face == "+x":
        return 0, s
    if face == "-x":
        return 0, 0
    if face == "+y":
        return 1, s
    if face == "-y":
        return 1, 0
    if face == "+z":
        return 2, s
    if face == "-z":
        return 2, 0
    raise ValueError("unknown face")


def _point_on_face(point, face: str, s: int) -> bool:
    axis, value = _face_coord(face, s)
    return point[axis] == value


def _face_perimeter_points(index, coarse_size_units: int, face: str, required_points):
    s = coarse_size_units
    h = int(s // 2)
    required = set(required_points)
    # Clockwise/counter-clockwise ordered convex boundary points for each face.
    if face in ("+z", "-z"):
        z = s if face == "+z" else 0
        seq = [(0, 0, z), (h, 0, z), (s, 0, z), (s, h, z), (s, s, z), (h, s, z), (0, s, z), (0, h, z)]
    elif face in ("+x", "-x"):
        x = s if face == "+x" else 0
        seq = [(x, 0, 0), (x, h, 0), (x, s, 0), (x, s, h), (x, s, s), (x, h, s), (x, 0, s), (x, 0, h)]
    else:
        y = s if face == "+y" else 0
        seq = [(0, y, 0), (h, y, 0), (s, y, 0), (s, y, h), (s, y, s), (h, y, s), (0, y, s), (0, y, h)]
    corners = {seq[0], seq[2], seq[4], seq[6]}
    pts = [p for p in seq if p in corners or p in required]
    return [index[p] for p in pts]


def _face_point_2d(point, face: str):
    x, y, z = point
    if face in ("+z", "-z"):
        return (x, y)
    if face in ("+x", "-x"):
        return (y, z)
    return (x, z)


def _area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _triangulate_ordered_face_polygon(index, coords, face: str):
    if len(coords) < 3:
        return []
    remaining = list(coords)
    tris = []
    guard = 0
    while len(remaining) > 3 and guard < 100:
        guard += 1
        clipped = False
        n = len(remaining)
        for i in range(n):
            prev_p = remaining[(i - 1) % n]
            cur_p = remaining[i]
            next_p = remaining[(i + 1) % n]
            a = _face_point_2d(prev_p, face)
            b = _face_point_2d(cur_p, face)
            c = _face_point_2d(next_p, face)
            if abs(_area2(a, b, c)) <= 1e-12:
                continue
            tris.append((index[prev_p], index[cur_p], index[next_p]))
            del remaining[i]
            clipped = True
            break
        if not clipped:
            # Fallback should never be reached for the convex square face, but
            # keep a deterministic non-zero best-effort split.
            return [(index[coords[0]], index[coords[i]], index[coords[i + 1]]) for i in range(1, len(coords) - 1)]
    if len(remaining) == 3:
        a, b, c = [_face_point_2d(p, face) for p in remaining]
        if abs(_area2(a, b, c)) > 1e-12:
            tris.append((index[remaining[0]], index[remaining[1]], index[remaining[2]]))
    return tris


def _coarse_face_triangles(index, coarse_size_units: int, face: str):
    s = coarse_size_units
    if face in ("+z", "-z"):
        z = s if face == "+z" else 0
        return [(index[(0, 0, z)], index[(s, 0, z)], index[(s, s, z)]), (index[(0, 0, z)], index[(s, s, z)], index[(0, s, z)])]
    if face in ("+x", "-x"):
        x = s if face == "+x" else 0
        return [(index[(x, 0, 0)], index[(x, s, 0)], index[(x, s, s)]), (index[(x, 0, 0)], index[(x, s, s)], index[(x, 0, s)])]
    y = s if face == "+y" else 0
    return [(index[(0, y, 0)], index[(s, y, 0)], index[(s, y, s)]), (index[(0, y, 0)], index[(s, y, s)], index[(0, y, s)])]

def _boundary_triangles_lightweight(index, coarse_size_units: int, refined_faces):
    s = coarse_size_units
    h = int(s // 2)
    required_points = set()
    # Refined faces need edge midpoints and face center. Adjacent unrefined faces
    # must include the same edge midpoint nodes to avoid hanging-node interfaces.
    for face in refined_faces:
        if face in ("+z", "-z"):
            z = s if face == "+z" else 0
            required_points.update([(h, 0, z), (h, s, z), (0, h, z), (s, h, z), (h, h, z)])
        elif face in ("+x", "-x"):
            x = s if face == "+x" else 0
            required_points.update([(x, h, 0), (x, h, s), (x, 0, h), (x, s, h), (x, h, h)])
        else:
            y = s if face == "+y" else 0
            required_points.update([(h, y, 0), (h, y, s), (0, y, h), (s, y, h), (h, y, h)])
    boundary = {}
    refined_tris = []
    unrefined_tris = []
    for face in FACE_NAMES:
        if face in refined_faces:
            tris = _face_grid_triangles(index, coarse_size_units, face)
            refined_tris.extend(tris)
        else:
            tris = _coarse_face_triangles(index, coarse_size_units, face)
            unrefined_tris.extend(tris)
        boundary[face] = tris
    return boundary, refined_tris, unrefined_tris, required_points


def _orient_tet_positive(nodes, tet):
    a, b, c, d = tet
    p0, p1, p2, p3 = [nodes[i] for i in tet]
    u = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    v = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    w = (p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2])
    cross = (v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2], v[0] * w[1] - v[1] * w[0])
    vol6 = u[0] * cross[0] + u[1] * cross[1] + u[2] * cross[2]
    if vol6 < 0:
        return (a, c, b, d)
    return tet


def get_lightweight_transition_template(
    coarse_size_units: int,
    refined_faces,
    orientation_context=None,
    regular_face_diagonal_context=None,
):
    faces = frozenset(refined_faces or [])
    fallback = get_transition_template(coarse_size_units, faces, orientation_context)
    if not fallback.supported:
        return fallback
    nodes, index, h = _node_index_map(coarse_size_units)
    boundary, refined_tris, unrefined_tris, required_points = _boundary_triangles_lightweight(index, coarse_size_units, faces)
    center_coord = (h, h, h)
    center_idx = index[center_coord]
    tets = []
    for face in FACE_NAMES:
        for tri in boundary[face]:
            tet = (center_idx, tri[0], tri[1], tri[2])
            tet = _orient_tet_positive(nodes, tet)
            tets.append(tet)
    mids, centers = _midpoints_and_centers(index, coarse_size_units, faces)
    pattern = "none" if not faces else "_".join(sorted(faces))
    tpl = TransitionTemplate(
        template_id="generic_2to1_lightweight_star_s{}_{}".format(coarse_size_units, pattern),
        coarse_size_units=coarse_size_units,
        refined_faces=faces,
        local_nodes_int=nodes,
        tetra_connectivity_local=tets,
        boundary_triangles_by_face=boundary,
        refined_interface_triangles=refined_tris,
        required_midpoint_nodes=mids,
        required_face_center_nodes=centers,
        uses_cell_center=True,
        supported=True,
        implementation_method="cell_center_star_decomposition",
        template_family="lightweight_cell_center_star",
        unrefined_interface_triangles=unrefined_tris,
        tet_count=len(tets),
        fallback_used=False,
    )
    return tpl


def classify_and_select_cell_template(
    cell=None,
    neighbor_levels=None,
    scale=None,
    refined_faces=None,
    coarse_size_units: int | None = None,
):
    """Shared production selector for projection and generation dry-runs.

    Empty refined face sets are regular cells and must not call the transition
    template family. Non-empty refined face sets call the lightweight transition
    template. This function intentionally returns only metadata; full mesh
    generation can use the same selector before instantiating actual tets.
    """
    faces = frozenset(refined_faces or [])
    if not faces:
        return CellTemplateSelection(
            cell_class="regular_cell",
            refined_faces=faces,
            template_id="regular_6tet_template",
            tet_count=6,
            selection_reason="refined_faces_empty_use_regular_6tet",
            coarse_size_units=coarse_size_units,
            template_family="regular",
            production_eligible=True,
        )
    size = int(coarse_size_units or scale or 2)
    template = get_lightweight_transition_template(size, faces)
    if not template.supported:
        return CellTemplateSelection(
            cell_class="unsupported_transition_cell",
            refined_faces=faces,
            template_id=template.template_id,
            tet_count=0,
            selection_reason=template.failure_reason or "unsupported_transition_template",
            coarse_size_units=size,
            template_family="unsupported",
            production_eligible=False,
        )
    return CellTemplateSelection(
        cell_class="transition_cell",
        refined_faces=faces,
        template_id=template.template_id,
        tet_count=len(template.tetra_connectivity_local),
        selection_reason="nonempty_refined_faces_use_lightweight_transition_template",
        coarse_size_units=size,
        template_family=getattr(template, "template_family", "lightweight_transition"),
        production_eligible=True,
    )

def transition_templates_ready_for_patterns(patterns, sizes=(2, 4)):
    missing = []
    for size in sizes:
        for pattern in patterns:
            tpl = get_transition_template(size, pattern)
            if not tpl.supported:
                missing.append({"coarse_size_units": size, "refined_faces": sorted(pattern), "reason": tpl.failure_reason})
    return len(missing) == 0, missing

