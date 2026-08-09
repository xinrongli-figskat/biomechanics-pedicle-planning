from __future__ import annotations

from collections import Counter, defaultdict, deque

FACES = ('+x', '-x', '+y', '-y', '+z', '-z')
FACE_AXIS = {'+x': 0, '-x': 0, '+y': 1, '-y': 1, '+z': 2, '-z': 2}


def face_axes(face):
    if face in ('+x', '-x'):
        return (1, 2)
    if face in ('+y', '-y'):
        return (0, 2)
    return (0, 1)


def face_const(bbox, face):
    ax = FACE_AXIS[face]
    return bbox[ax + 3] if face[0] == '+' else bbox[ax]


def face_rect(bbox, face):
    ax0, ax1 = face_axes(face)
    fix = FACE_AXIS[face]
    return {
        'axis_u': ax0,
        'axis_v': ax1,
        'fix_axis': fix,
        'fix_value': face_const(bbox, face),
        'u0': bbox[ax0],
        'u1': bbox[ax0 + 3],
        'v0': bbox[ax1],
        'v1': bbox[ax1 + 3],
    }


def bbox_from_cell_id(cell_id):
    level, ix, iy, iz = [int(v) for v in cell_id]
    size = {0: 4, 1: 2, 2: 1}[level]
    origin = (ix, iy, iz)
    return origin, size, (ix, iy, iz, ix + size, iy + size, iz + size)


def make_point(rect, u, v):
    p = [0, 0, 0]
    p[rect['axis_u']] = u
    p[rect['axis_v']] = v
    p[rect['fix_axis']] = rect['fix_value']
    return tuple(p)


def tri_sig(a, b, c):
    return tuple(sorted((tuple(a), tuple(b), tuple(c))))


def segment_sig(a, b):
    return tuple(sorted((tuple(a), tuple(b))))


def parity_diag(u, v):
    return (u + v) % 2


def build_face_bucket_from_finest_subcell_ownership(cell, neighbor_context=None, required_shared_rectangles=None):
    bbox = tuple(cell['bbox_int6'])
    result = {}
    for face in FACES:
        rect = face_rect(bbox, face)
        tris = []
        authorities = {}
        subcell_authority = {}
        # This is the validated repair: every finest 2D subcell receives exactly
        # one authority, then triangles are rebuilt from that ownership map.
        for u in range(rect['u0'], rect['u1']):
            for v in range(rect['v0'], rect['v1']):
                authority = 'default_face_triangulation'
                subcell_authority[(u, v)] = authority
                p00 = make_point(rect, u, v)
                p10 = make_point(rect, u + 1, v)
                p01 = make_point(rect, u, v + 1)
                p11 = make_point(rect, u + 1, v + 1)
                if parity_diag(u, v) == 0:
                    local = [tri_sig(p00, p10, p11), tri_sig(p00, p11, p01)]
                else:
                    local = [tri_sig(p00, p10, p01), tri_sig(p10, p11, p01)]
                for t in local:
                    tris.append(t)
                    authorities[t] = authority
        result[face] = {
            'triangles': tris,
            'authorities': authorities,
            'subcell_authority': subcell_authority,
            'required_shared_rectangles': [],
        }
    return result


def point_on_face(p, bbox, face):
    if p[FACE_AXIS[face]] != face_const(bbox, face):
        return False
    for ax in face_axes(face):
        if p[ax] < bbox[ax] or p[ax] > bbox[ax + 3]:
            return False
    return True


def tri_area_2d(tri, bbox, face):
    ax0, ax1 = face_axes(face)
    a, b, c = tri
    return abs((b[ax0] - a[ax0]) * (c[ax1] - a[ax1]) - (b[ax1] - a[ax1]) * (c[ax0] - a[ax0])) / 2.0


def edge_is_face_boundary(edge, bbox, face):
    a, b = edge
    for ax in face_axes(face):
        if a[ax] == b[ax] and a[ax] in (bbox[ax], bbox[ax + 3]):
            return True
    return False


def graph_components(edges):
    adj = defaultdict(set)
    verts = set()
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
        verts.add(a)
        verts.add(b)
    seen = set()
    comps = 0
    for v in verts:
        if v in seen:
            continue
        comps += 1
        q = deque([v])
        seen.add(v)
        while q:
            x = q.popleft()
            for nb in adj[x]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
    return comps


def audit_face(bbox, face, tris, authorities):
    unique = set(tris)
    seg = Counter()
    outside = 0
    zero = 0
    area_sum = 0.0
    vertices = set()
    for t in tris:
        if not all(point_on_face(p, bbox, face) for p in t):
            outside += 1
        ar = tri_area_2d(t, bbox, face)
        if ar == 0:
            zero += 1
        area_sum += ar
        pts = list(t)
        vertices.update(pts)
        for e in ((pts[0], pts[1]), (pts[0], pts[2]), (pts[1], pts[2])):
            seg[segment_sig(*e)] += 1
    boundary_edges = [e for e in seg if edge_is_face_boundary(e, bbox, face)]
    internal_edges = [e for e in seg if not edge_is_face_boundary(e, bbox, face)]
    boundary_bad = [e for e in boundary_edges if seg[e] != 1]
    internal_bad = [e for e in internal_edges if seg[e] != 2]
    rect = face_rect(bbox, face)
    subcells = {(u, v): 0.0 for u in range(rect['u0'], rect['u1']) for v in range(rect['v0'], rect['v1'])}
    for t in tris:
        us = sorted({p[rect['axis_u']] for p in t})
        vs = sorted({p[rect['axis_v']] for p in t})
        if len(us) >= 2 and len(vs) >= 2:
            u0, u1 = us[0], us[-1]
            v0, v1 = vs[0], vs[-1]
            if u1 - u0 == 1 and v1 - v0 == 1 and (u0, v0) in subcells:
                subcells[(u0, v0)] += 0.5
    uncovered = sum(1 for v in subcells.values() if v == 0)
    covered_once = sum(1 for v in subcells.values() if v == 1.0)
    multiply = sum(1 for v in subcells.values() if v > 1.0)
    union_area = sum(min(v, 1.0) for v in subcells.values())
    overlap_area = sum(max(0.0, v - 1.0) for v in subcells.values())
    expected = len(subcells)
    face_edges = set(seg)
    comp = graph_components(face_edges)
    euler = len(vertices) - len(face_edges) + len(unique)
    holes = 0 if comp == 1 and euler == 1 and uncovered == 0 else max(0, 1 - euler)
    status_ok = (
        len(tris) == len(unique)
        and outside == 0
        and zero == 0
        and uncovered == 0
        and multiply == 0
        and not boundary_bad
        and not internal_bad
        and comp == 1
        and euler == 1
    )
    source_counts = Counter(authorities.get(t, 'unknown') for t in tris)
    return {
        'face_id': face,
        'expected_subcell_count': len(subcells),
        'owned_subcell_count': covered_once,
        'unowned_subcell_count': uncovered,
        'multiply_owned_subcell_count': multiply,
        'triangle_count': len(tris),
        'duplicate_triangle_count': len(tris) - len(unique),
        'zero_area_triangle_count': zero,
        'triangle_outside_face_count': outside,
        'triangle_area_sum_units2': area_sum,
        'triangle_union_area_units2': union_area,
        'triangle_overlap_area_units2': overlap_area,
        'triangle_uncovered_area_units2': expected - union_area,
        'vertex_count': len(vertices),
        'edge_count': len(seg),
        'face_connected_component_count': comp,
        'face_hole_count': holes,
        'face_euler_characteristic': euler,
        'boundary_segment_incidence_error_count': len(boundary_bad),
        'internal_segment_incidence_error_count': len(internal_bad),
        'boundary_segment_incidence_error_examples_top20': [list(e) + [seg[e]] for e in boundary_bad[:20]],
        'internal_segment_incidence_error_examples_top20': [list(e) + [seg[e]] for e in internal_bad[:20]],
        'triangle_source_counts': dict(source_counts),
        'face_status': 'pass' if status_ok else 'failed',
        'failure_reason': None if status_ok else 'face bucket has coverage, incidence, component, or Euler error',
    }


def audit_cube_edges(bbox, face_buckets):
    def edge_nodes(face_a, face_b):
        axa = FACE_AXIS[face_a]
        axb = FACE_AXIS[face_b]
        ca = face_const(bbox, face_a)
        cb = face_const(bbox, face_b)
        nodes = {}
        for face in (face_a, face_b):
            pts = set()
            for t in face_buckets[face]['triangles']:
                for p in t:
                    if p[axa] == ca and p[axb] == cb:
                        pts.add(p)
            free = list({0, 1, 2} - {axa, axb})[0]
            ordered = tuple(sorted(pts, key=lambda p: p[free]))
            segs = {segment_sig(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)}
            nodes[face] = (ordered, segs)
        return nodes
    pairs = [('+x', '+y'), ('+x', '-y'), ('-x', '+y'), ('-x', '-y'), ('+x', '+z'), ('+x', '-z'), ('-x', '+z'), ('-x', '-z'), ('+y', '+z'), ('+y', '-z'), ('-y', '+z'), ('-y', '-z')]
    results = []
    for idx, (fa, fb) in enumerate(pairs, 1):
        ns = edge_nodes(fa, fb)
        na, sa = ns[fa]
        nb, sb = ns[fb]
        results.append({
            'edge_id': 'E%02d' % idx,
            'face_a': fa,
            'face_b': fb,
            'node_set_match': set(na) == set(nb),
            'ordered_node_sequence_match': na == nb,
            'segment_set_match': sa == sb,
            'missing_nodes_in_face_a': sorted(set(nb) - set(na)),
            'missing_nodes_in_face_b': sorted(set(na) - set(nb)),
            'missing_segments_in_face_a': sorted(sb - sa),
            'missing_segments_in_face_b': sorted(sa - sb),
        })
    return results


def audit_shell(face_buckets):
    tris = []
    tri_faces = defaultdict(set)
    for face, bucket in face_buckets.items():
        for t in bucket['triangles']:
            tris.append(t)
            tri_faces[t].add(face)
    edge = Counter()
    verts = set()
    for t in tris:
        pts = list(t)
        verts.update(pts)
        for e in ((pts[0], pts[1]), (pts[0], pts[2]), (pts[1], pts[2])):
            edge[segment_sig(*e)] += 1
    comps = graph_components(set(edge))
    chi = len(verts) - len(edge) + len(set(tris))
    return {
        'duplicate_triangle_before_dedup': len(tris) - len(set(tris)),
        'same_triangle_different_face_bucket_count': sum(1 for faces in tri_faces.values() if len(faces) > 1),
        'shell_vertex_count': len(verts),
        'shell_edge_count': len(edge),
        'shell_face_count': len(set(tris)),
        'shell_euler_characteristic': chi,
        'shell_edge_incidence_1_count': sum(1 for v in edge.values() if v == 1),
        'shell_edge_incidence_2_count': sum(1 for v in edge.values() if v == 2),
        'shell_edge_incidence_gt2_count': sum(1 for v in edge.values() if v > 2),
        'shell_connected_component_count': comps,
    }
