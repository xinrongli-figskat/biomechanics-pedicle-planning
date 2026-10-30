"""Turn planning-system meshes (bone STL, positioned screw models) into pipeline inputs.

Some planning systems store a segmented bone surface (e.g. STL) and the
planned screws as positioned surface models instead of a voxel mask and
entry/tip coordinates. This script converts them for ``run_case_pipeline.py``:

* bone meshes -> a label map on the CT grid; separate vertebrae become
  separate labels (connected components, e.g. L3 and L4 in one STL). When
  vertebrae touch (fused surface at a facet joint or disc), ``--vertebrae N``
  splits the mask into N labels by erosion and marker-based watershed.
  Labels are numbered cranial to caudal (1 = most cranial);
* screw meshes -> entry point, tip and diameter from the mesh shape
  (principal axis and the radius profile along it). A wider end segment
  (screw head / tulip) is detected and excluded, so length and diameter
  describe the threaded shank and the entry is the shank end next to the
  head; without a head the entry is the end that lies less deep in bone.
  Each screw gets the label of the vertebra its axis runs through.

Mesh coordinates are matched to the CT automatically: each mesh is tested in
LPS and RAS (x and y negated) and the convention whose bounds overlap the CT
is used (override with ``--mesh-coords``). Meshes that overlap the CT in
neither convention (e.g. unpositioned screw templates) are skipped.

Outputs use neutral names (``screw1``, ``label 1``) and never contain input
file names, which may include patient names.

Example
-------
    python examples/clinical/prepare_from_meshes.py --ct cases/case01/ct.nii \\
        --bone-mesh cases/case01/bone.stl --screw-mesh cases/case01/screw_a.stl \\
        --out cases/case01/prepared
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from pedicle_planning.ai.trajectory import points_inside_mesh, sample_mask_nearest


RAS_FLIP = np.array([-1.0, -1.0, 1.0])


def read_mesh(path: str):
    import pyvista as pv

    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh.triangulate().clean()


def ct_bounds(ct: sitk.Image) -> tuple:
    """Axis-aligned physical bounds (min, max) of the CT volume."""
    size = np.asarray(ct.GetSize()) - 1
    corners = [ct.TransformContinuousIndexToPhysicalPoint(tuple(float(c) for c in (i, j, k)))
               for i in (0, size[0]) for j in (0, size[1]) for k in (0, size[2])]
    corners = np.asarray(corners)
    return corners.min(axis=0), corners.max(axis=0)


def overlap_fraction(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    return float(np.mean(inside))


def to_lps(mesh, convention: str, ct: sitk.Image):
    """Return (mesh in LPS, chosen convention, overlap) choosing LPS/RAS by CT overlap if 'auto'."""
    lo, hi = ct_bounds(ct)
    points = np.asarray(mesh.points, dtype=np.float64)
    options = {"lps": points, "ras": points * RAS_FLIP}
    if convention == "auto":
        scores = {name: overlap_fraction(p, lo, hi) for name, p in options.items()}
        convention = max(scores, key=scores.get)
        overlap = scores[convention]
    else:
        overlap = overlap_fraction(options[convention], lo, hi)
    out = mesh.copy()
    out.points = options[convention]
    return out, convention, overlap


def rasterize(mesh, ct: sitk.Image, margin_vox: int = 2) -> np.ndarray:
    """Boolean array (z, y, x) on the CT grid marking voxel centres inside the closed mesh."""
    size = np.asarray(ct.GetSize())
    lo, hi = np.asarray(mesh.bounds[0::2]), np.asarray(mesh.bounds[1::2])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    idx = np.array([ct.TransformPhysicalPointToContinuousIndex(tuple(c)) for c in corners])
    i_lo = np.clip(np.floor(idx.min(axis=0)).astype(int) - margin_vox, 0, size - 1)
    i_hi = np.clip(np.ceil(idx.max(axis=0)).astype(int) + margin_vox, 0, size - 1)
    out = np.zeros(size[::-1], dtype=bool)
    if np.any(i_hi < i_lo):
        return out
    ii, jj, kk = np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(i_lo, i_hi)), indexing="ij")
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    origin = np.asarray(ct.GetOrigin())
    matrix = np.asarray(ct.GetDirection()).reshape(3, 3) * np.asarray(ct.GetSpacing())
    points = origin + ijk.astype(np.float64) @ matrix.T
    inside = points_inside_mesh(mesh, points)
    sel = ijk[inside]
    out[sel[:, 2], sel[:, 1], sel[:, 0]] = True
    return out


def _components(binary: sitk.Image, min_volume_ml: float) -> np.ndarray:
    """Connected components (largest first) with components below min_volume_ml removed."""
    labels = sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(binary), sortByObjectSize=True))
    voxel_ml = float(np.prod(binary.GetSpacing())) / 1000.0
    counts = np.bincount(labels.ravel())
    keep = [v for v in range(1, len(counts)) if counts[v] * voxel_ml >= min_volume_ml]
    out = np.zeros_like(labels)
    for new, old in enumerate(keep, start=1):
        out[labels == old] = new
    return out


def split_touching(binary: sitk.Image, n_parts: int, seed_min_ml: float = 2.0, max_radius_mm: float = 10.0) -> np.ndarray:
    """
    Split a mask of touching vertebrae into n_parts labels.

    The mask is eroded with growing radius until it falls apart into n_parts
    components of at least seed_min_ml; these seeds are grown back over the
    original mask with a watershed on the distance map, so the cut runs along
    the narrow connection (facet contact, disc bridge).
    """
    spacing = np.asarray(binary.GetSpacing())
    seeds = None
    radius_mm = float(spacing.min())
    while radius_mm <= max_radius_mm + 1e-9:
        radius = [max(1, int(round(radius_mm / sp))) for sp in spacing]
        parts = _components(sitk.BinaryErode(binary, radius), seed_min_ml)
        if parts.max() >= n_parts:
            seeds = np.where(parts <= n_parts, parts, 0)
            break
        radius_mm += float(spacing.min())
    if seeds is None:
        raise ValueError(f"could not split the bone mask into {n_parts} parts by erosion up to {max_radius_mm} mm")
    markers = sitk.GetImageFromArray(seeds.astype(np.uint32))
    markers.CopyInformation(binary)
    distance = sitk.SignedMaurerDistanceMap(binary, insideIsPositive=False, squaredDistance=False,
                                            useImageSpacing=True)
    grown = sitk.MorphologicalWatershedFromMarkers(distance, markers, markWatershedLine=False, fullyConnected=True)
    return sitk.GetArrayFromImage(grown) * (sitk.GetArrayViewFromImage(binary) > 0)


def label_components(bone: np.ndarray, reference: sitk.Image, min_volume_ml: float,
                     n_vertebrae: int = None, cranial=(0.0, 0.0, 1.0)) -> tuple:
    """
    Vertebra labels from the bone mask, numbered cranial to caudal.

    Returns (label image, {label: {"volume_ml", "centroid"}}, split_applied).
    """
    binary = sitk.GetImageFromArray(bone.astype(np.uint8))
    binary.CopyInformation(reference)
    labels = _components(binary, min_volume_ml)
    split = False
    if n_vertebrae is not None and labels.max() < n_vertebrae:
        labels = split_touching(_as_image(labels > 0, reference), n_vertebrae)
        split = True

    voxel_ml = float(np.prod(reference.GetSpacing())) / 1000.0
    origin = np.asarray(reference.GetOrigin())
    matrix = np.asarray(reference.GetDirection()).reshape(3, 3) * np.asarray(reference.GetSpacing())
    info = []
    for value in range(1, int(labels.max()) + 1):
        kji = np.argwhere(labels == value)
        if kji.shape[0] == 0:
            continue
        centroid = origin + kji[:, ::-1].mean(axis=0) @ matrix.T
        info.append((value, kji.shape[0] * voxel_ml, centroid))
    info.sort(key=lambda item: -float(np.dot(item[2], cranial)))  # most cranial first
    ordered = np.zeros(labels.shape, dtype=np.uint8)
    details = {}
    for new, (old, volume, centroid) in enumerate(info, start=1):
        ordered[labels == old] = new
        details[new] = {"volume_ml": volume, "centroid": centroid}
    return _as_image(ordered, reference), details, split


def _as_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.CopyInformation(reference)
    return image


def screw_from_mesh(mesh, bin_mm: float = 1.0, head_ratio: float = 1.3) -> dict:
    """
    Axis, shank end points, shank length and outer diameter of a screw-shaped mesh.

    The radius profile along the principal axis (95th percentile per 1 mm bin)
    gives the shank radius (median over bins); a run of bins at one end that is
    wider than head_ratio x shank radius is treated as the screw head / tulip
    and excluded. end_a is the head side when a head is found.
    """
    points = np.asarray(mesh.points, dtype=np.float64)
    centre = points.mean(axis=0)
    axis = np.linalg.svd(points - centre, full_matrices=False)[2][0]
    t = (points - centre) @ axis
    radial = np.linalg.norm((points - centre) - np.outer(t, axis), axis=1)

    t0, t1 = float(t.min()), float(t.max())
    n_bins = max(1, int(np.ceil((t1 - t0) / bin_mm)))
    index = np.minimum(((t - t0) / bin_mm).astype(int), n_bins - 1)
    bins_r = {int(b): float(np.percentile(radial[index == b], 95)) for b in np.unique(index)}
    shank_radius = float(np.median(list(bins_r.values())))
    threshold = head_ratio * shank_radius

    # head at the low end: bins 0..k ; at the high end: bins n-1..n-1-k
    low = [b for b in range(n_bins)]
    high = list(reversed(low))
    low_last = None
    for b in low:
        if b in bins_r:
            if bins_r[b] > threshold:
                low_last = b
            else:
                break
    high_first = None
    for b in high:
        if b in bins_r:
            if bins_r[b] > threshold:
                high_first = b
            else:
                break
    head_low = 0.0 if low_last is None else (low_last + 1) * bin_mm
    head_high = 0.0 if high_first is None else (n_bins - high_first) * bin_mm
    if head_low and head_high:  # both ends wide: not a recognisable head, keep the full length
        head_low = head_high = 0.0

    s0, s1 = t0 + head_low, t1 - head_high
    if head_high > head_low:  # head on the high end: make end_a the head side
        end_a, end_b = centre + s1 * axis, centre + s0 * axis
    else:
        end_a, end_b = centre + s0 * axis, centre + s1 * axis
    in_shank = (t >= s0) & (t <= s1)
    return {
        "end_a": end_a,
        "end_b": end_b,
        "length": float(s1 - s0),
        "diameter": float(2.0 * np.percentile(radial[in_shank], 95)) if np.any(in_shank) else 2.0 * shank_radius,
        "head_found": bool(head_low or head_high),
        "head_length": float(max(head_low, head_high)),
        "total_length": float(t1 - t0),
    }


def _bone_depth(point, mask: sitk.Image, radius: float = 3.0) -> float:
    """Fraction of a small sphere around `point` that lies in bone."""
    offsets = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)], dtype=float)
    offsets = offsets / np.maximum(np.linalg.norm(offsets, axis=1, keepdims=True), 1e-9) * radius
    offsets[13] = 0.0
    return float(np.mean(sample_mask_nearest(mask, np.asarray(point) + offsets)))


def orient_and_label(screw: dict, labels: sitk.Image) -> dict:
    """
    Pick the entry end and the vertebra label the axis runs through.

    With a detected head the entry is the shank end next to it (end_a); otherwise
    the end with less bone around it.
    """
    binary = sitk.Cast(labels > 0, sitk.sitkUInt8)
    depth_a = _bone_depth(screw["end_a"], binary)
    depth_b = _bone_depth(screw["end_b"], binary)
    if screw.get("head_found") or depth_a <= depth_b:
        entry, tip = screw["end_a"], screw["end_b"]
    else:
        entry, tip = screw["end_b"], screw["end_a"]
    samples = entry + np.outer(np.linspace(0.0, 1.0, 81), tip - entry)
    array = sitk.GetArrayViewFromImage(labels)
    size = np.asarray(labels.GetSize())
    hits = []
    for p in samples:
        idx = np.rint(labels.TransformPhysicalPointToContinuousIndex(tuple(p))).astype(int)
        if np.all(idx >= 0) and np.all(idx < size):
            value = int(array[idx[2], idx[1], idx[0]])
            if value:
                hits.append(value)
    label = int(np.bincount(hits).argmax()) if hits else None
    return {
        "entry": entry, "tip": tip, "label": label,
        "axis_in_bone_fraction": len(hits) / len(samples),
        "entry_bone_fraction": depth_a if entry is screw["end_a"] else depth_b,
        "tip_bone_fraction": depth_b if entry is screw["end_a"] else depth_a,
    }


def prepare(args) -> dict:
    ct = sitk.ReadImage(str(args.ct))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"bone_meshes": [], "screws": [], "labels": {}}

    bone = np.zeros(ct.GetSize()[::-1], dtype=bool)
    for n, path in enumerate(args.bone_mesh, start=1):
        mesh, convention, overlap = to_lps(read_mesh(path), args.mesh_coords, ct)
        entry = {"mesh": f"bone mesh {n}", "coords": convention, "overlap_with_ct": overlap,
                 "closed": bool(mesh.n_open_edges == 0), "mesh_volume_ml": float(mesh.volume) / 1000.0}
        if overlap < 0.5:
            entry["skipped"] = "does not overlap the CT in LPS or RAS"
        else:
            part = rasterize(mesh, ct)
            entry["rasterized_volume_ml"] = float(np.count_nonzero(part)) * float(np.prod(ct.GetSpacing())) / 1000.0
            bone |= part
        report["bone_meshes"].append(entry)
    if not bone.any():
        raise ValueError("no bone mesh could be placed in the CT")

    labels, details, split = label_components(bone, ct, args.min_volume_ml, args.vertebrae)
    report["labels"] = {str(k): v for k, v in details.items()}
    report["split_touching_vertebrae"] = split
    if args.vertebrae is not None and len(details) != args.vertebrae:
        print(f"[warn] expected {args.vertebrae} vertebrae, found {len(details)}", file=sys.stderr)
    sitk.WriteImage(labels, str(out_dir / "vertebrae_from_mesh.nii.gz"))

    rows = []
    for n, path in enumerate(args.screw_mesh or [], start=1):
        sid = f"screw{n}"
        mesh, convention, overlap = to_lps(read_mesh(path), args.mesh_coords, ct)
        entry = {"screw_id": sid, "coords": convention, "overlap_with_ct": overlap}
        if overlap < 0.5:
            entry["skipped"] = "not positioned in the CT (template or other space)"
            report["screws"].append(entry)
            continue
        geometry = screw_from_mesh(mesh)
        placed = orient_and_label(geometry, labels)
        entry.update({"length_mm": geometry["length"], "diameter_mm": geometry["diameter"],
                      "head_found": geometry["head_found"], "head_length_mm": geometry["head_length"],
                      "total_length_mm": geometry["total_length"], **placed})
        report["screws"].append(entry)
        if placed["label"] is None:
            entry["skipped"] = "screw axis does not run through any bone label"
            continue
        rows.append([sid, placed["label"], *placed["entry"], *placed["tip"],
                     round(geometry["diameter"], 2), round(geometry["length"], 2)])

    with open(out_dir / "screws_from_mesh.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["screw_id", "label", "entry_x", "entry_y", "entry_z", "tip_x", "tip_y", "tip_z",
                         "diameter", "length"])
        writer.writerows(rows)
    with open(out_dir / "prepare_report.json", "w", encoding="utf-8") as handle:
        json.dump(_jsonable(report), handle, indent=2)
    return report


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in list(value)]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 3)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def print_report(report: dict, out: str) -> None:
    for b in report["bone_meshes"]:
        status = b.get("skipped", f"rasterized {b.get('rasterized_volume_ml', 0):.1f} ml "
                                  f"(mesh volume {b['mesh_volume_ml']:.1f} ml, closed={b['closed']})")
        print(f"[{b['mesh']}] coords={b['coords']} overlap={b['overlap_with_ct']:.2f}: {status}")
    if report.get("split_touching_vertebrae"):
        print("[info] touching vertebrae were split by erosion + watershed")
    for label, info in report["labels"].items():
        print(f"[label {label}] {info['volume_ml']:.1f} ml, centroid z {info['centroid'][2]:.1f} mm"
              + ("  (most cranial)" if label == "1" else ""))
    for s in report["screws"]:
        if "skipped" in s and "length_mm" not in s:
            print(f"[{s['screw_id']}] coords={s['coords']} overlap={s['overlap_with_ct']:.2f}: skipped - {s['skipped']}")
            continue
        head = (f" (head {s['head_length_mm']:.1f} mm removed, total {s['total_length_mm']:.1f} mm)"
                if s.get("head_found") else " (no head detected)")
        print(f"[{s['screw_id']}] coords={s['coords']} shank length={s['length_mm']:.1f} mm "
              f"diameter={s['diameter_mm']:.2f} mm{head} label={s['label']} "
              f"axis-in-bone={s['axis_in_bone_fraction']:.2f} "
              f"entry/tip bone fraction={s['entry_bone_fraction']:.2f}/{s['tip_bone_fraction']:.2f}"
              + (f"  SKIPPED: {s['skipped']}" if "skipped" in s else ""))
    print(f"outputs written to {out} (vertebrae_from_mesh.nii.gz, screws_from_mesh.csv, prepare_report.json)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ct", required=True, help="CT volume the meshes were planned on")
    parser.add_argument("--bone-mesh", action="append", required=True, help="bone surface mesh (repeatable)")
    parser.add_argument("--screw-mesh", action="append", default=[], help="positioned screw mesh (repeatable)")
    parser.add_argument("--mesh-coords", choices=("auto", "lps", "ras"), default="auto")
    parser.add_argument("--vertebrae", type=int, default=None,
                        help="expected number of vertebrae in the bone meshes; touching ones are split to this count")
    parser.add_argument("--min-volume-ml", type=float, default=5.0,
                        help="drop bone components smaller than this (fragments)")
    parser.add_argument("--out", required=True, help="output folder")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare(args)
    print_report(report, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
