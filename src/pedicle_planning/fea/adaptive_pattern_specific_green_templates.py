from __future__ import annotations

from collections import Counter, defaultdict, deque
import itertools
import math

FACES = ("+x", "-x", "+y", "-y", "+z", "-z")
FACE_AXIS = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2}


def face_axes(face):
    if face in ("+x", "-x"):
        return (1, 2)
    if face in ("+y", "-y"):
        return (0, 2)
    return (0, 1)


def face_const(face, size):
    return size if face[0] == "+" else 0


def make_point(face, size, u, v):
    axes = face_axes(face)
    p = [0, 0, 0]
    p[FACE_AXIS[face]] = face_const(face, size)
    p[axes[0]] = u
    p[axes[1]] = v
    return tuple(p)


def tri_sig(a, b, c):
    return tuple(sorted((tuple(a), tuple(b), tuple(c))))


def seg_sig(a, b):
    return tuple(sorted((tuple(a), tuple(b))))


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def vol6(a, b, c, d):
    return dot(vsub(b, a), cross(vsub(c, a), vsub(d, a)))


def norm(a):
    return math.sqrt(dot(a, a))


def tet_quality(coords):
    v = abs(vol6(*coords)) / 6.0
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    lens = [norm(vsub(coords[i], coords[j])) for i, j in pairs]
    q = 0.0 if v <= 0 else 12.0 * ((3.0 * v) ** (2.0 / 3.0)) / sum(x * x for x in lens)
    ar = max(lens) / min(lens) if min(lens) > 0 else float("inf")
    return q, ar, v


def cube_edges():
    return [
        ("+x", "+y"), ("+x", "-y"), ("-x", "+y"), ("-x", "-y"),
        ("+x", "+z"), ("+x", "-z"), ("-x", "+z"), ("-x", "-z"),
        ("+y", "+z"), ("+y", "-z"), ("-y", "+z"), ("-y", "-z"),
    ]


def edge_free_axis(face_a, face_b):
    return list({0, 1, 2} - {FACE_AXIS[face_a], FACE_AXIS[face_b]})[0]


def edge_coord_values_for_face(face, size, coords_u, coords_v, edge_face):
    free = edge_free_axis(face, edge_face)
    axes = face_axes(face)
    if free == axes[0]:
        return set(coords_u[face])
    if free == axes[1]:
        return set(coords_v[face])
    return {0, size}


def add_edge_values_to_face(face, edge_face, values, coords_u, coords_v):
    axes = face_axes(face)
    free = edge_free_axis(face, edge_face)
    if free == axes[0]:
        coords_u[face].update(values)
    elif free == axes[1]:
        coords_v[face].update(values)


def build_boundary_triangles(size, refined_faces, origin=(0, 0, 0)):
    refined_faces = frozenset(refined_faces or ())
    mid = size // 2
    coords_u = {f: {0, size} for f in FACES}
    coords_v = {f: {0, size} for f in FACES}
    for f in refined_faces:
        coords_u[f].add(mid)
        coords_v[f].add(mid)
    changed = True
    while changed:
        changed = False
        for fa, fb in cube_edges():
            vals = edge_coord_values_for_face(fa, size, coords_u, coords_v, fb) | edge_coord_values_for_face(fb, size, coords_u, coords_v, fa)
            before = (len(coords_u[fa]), len(coords_v[fa]), len(coords_u[fb]), len(coords_v[fb]))
            add_edge_values_to_face(fa, fb, vals, coords_u, coords_v)
            add_edge_values_to_face(fb, fa, vals, coords_u, coords_v)
            after = (len(coords_u[fa]), len(coords_v[fa]), len(coords_u[fb]), len(coords_v[fb]))
            changed = changed or before != after
    ox, oy, oz = origin
    out = {}
    for face in FACES:
        tris = []
        us = sorted(coords_u[face])
        vs = sorted(coords_v[face])
        for u0, u1 in zip(us, us[1:]):
            for v0, v1 in zip(vs, vs[1:]):
                p00 = make_point(face, size, u0, v0)
                p10 = make_point(face, size, u1, v0)
                p01 = make_point(face, size, u0, v1)
                p11 = make_point(face, size, u1, v1)
                # Use global integer parity if an origin is supplied.
                gu = p00[face_axes(face)[0]] + origin[face_axes(face)[0]]
                gv = p00[face_axes(face)[1]] + origin[face_axes(face)[1]]
                gp = face_const(face, size) + origin[FACE_AXIS[face]]
                if (gu + gv + gp) % 2 == 0:
                    local = [tri_sig(p00, p10, p11), tri_sig(p00, p11, p01)]
                else:
                    local = [tri_sig(p00, p10, p01), tri_sig(p10, p11, p01)]
                shifted = []
                for t in local:
                    shifted.append(tuple(sorted((tuple((p[0] + ox, p[1] + oy, p[2] + oz)) for p in t))))
                tris.extend(shifted)
        out[face] = tris
    return out


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
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    q.append(y)
    return comps


def validate_template(size, refined_faces):
    boundary = build_boundary_triangles(size, refined_faces)
    tris = []
    for f in FACES:
        tris.extend(boundary[f])
    edge = Counter()
    verts = set()
    for t in tris:
        verts.update(t)
        pts = list(t)
        for a, b in ((pts[0], pts[1]), (pts[0], pts[2]), (pts[1], pts[2])):
            edge[seg_sig(a, b)] += 1
    shell_ok = (
        len(tris) == len(set(tris))
        and sum(1 for v in edge.values() if v == 1) == 0
        and sum(1 for v in edge.values() if v > 2) == 0
        and graph_components(edge) == 1
        and len(verts) - len(edge) + len(set(tris)) == 2
    )
    center = (size / 2.0, size / 2.0, size / 2.0)
    tets = []
    zero = inv = near = 0
    q = []
    ar = []
    vols = []
    for tri in tris:
        coords = [center] + [tuple(float(x) for x in p) for p in tri]
        v6 = vol6(*coords)
        tet = ("center",) + tuple(tri)
        if v6 < 0:
            inv += 1
            coords[2], coords[3] = coords[3], coords[2]
            tet = ("center", tri[0], tri[2], tri[1])
            v6 = -v6
        vol = abs(v6) / 6.0
        if vol == 0:
            zero += 1
        if vol <= max(1e-14, (size ** 3) * 1e-12):
            near += 1
        qq, aa, vv = tet_quality(coords)
        q.append(qq)
        ar.append(aa)
        vols.append(vv)
        tets.append(tet)
    face_owner = defaultdict(list)
    for ti, tet in enumerate(tets):
        for f in itertools.combinations(tet, 3):
            face_owner[tuple(sorted(f, key=repr))].append(ti)
    nonmanifold = sum(1 for v in face_owner.values() if len(v) > 2)
    adj = [set() for _ in tets]
    for owners in face_owner.values():
        if len(owners) == 2:
            a, b = owners
            adj[a].add(b)
            adj[b].add(a)
    seen = set()
    comps = 0
    for i in range(len(tets)):
        if i in seen:
            continue
        comps += 1
        qd = deque([i])
        seen.add(i)
        while qd:
            x = qd.popleft()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    qd.append(y)
    rel = abs(sum(vols) - size ** 3) / (size ** 3)
    ok = shell_ok and zero == 0 and near == 0 and nonmanifold == 0 and rel <= 1e-10 and min(q) > 0 and max(ar) <= 10
    return {
        "template_id": "pattern_green_s%s_%s" % (size, "none" if not refined_faces else "_".join(sorted(refined_faces))),
        "size_units": size,
        "refined_faces": sorted(refined_faces),
        "tet_count": len(tets),
        "boundary_triangle_count": len(tris),
        "duplicate_tet_count": len(tets) - len(set(tuple(sorted(t, key=repr)) for t in tets)),
        "zero_volume_tet_count": zero,
        "near_zero_volume_tet_count": near,
        "inverted_before_fix_count": inv,
        "inverted_after_fix_count": 0,
        "nonmanifold_face_count": nonmanifold,
        "volume_relative_error": rel,
        "quality_min": min(q),
        "quality_mean": sum(q) / len(q),
        "aspect_ratio_max": max(ar),
        "shell_valid": shell_ok,
        "tet_graph_component_count": comps,
        "template_status": "pass" if ok else "failed",
    }
