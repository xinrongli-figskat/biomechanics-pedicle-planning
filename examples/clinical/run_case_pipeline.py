"""Run the screw-planning feature pipeline on one locally stored CT case.

Pipeline per screw:
    CT + vertebra mask + screw entry/tip
    -> per-vertebra anatomical frame (fallback: image axes)
    -> whole vertebral-body bone density (patient-intrinsic features)
    -> pedicle isthmus detection and trajectory match
    -> trajectory bone-density features of the baseline screw
    -> candidate generation, safety filtering and (optional) risk scoring

Everything runs locally; nothing is uploaded. Input paths and image metadata
other than geometry are not written to the outputs. Keep inputs and outputs
outside the repository (or in the git-ignored ``cases/`` and ``case_output/``
folders) so that patient-derived files are never committed.

This is a research prototype, not a medical device. The built-in ``demo`` risk
function is a placeholder with no clinical validity.

Example
-------
    python examples/clinical/run_case_pipeline.py \\
        --ct cases/case01/ct.nii.gz \\
        --mask cases/case01/vertebrae.nii.gz --label 27 \\
        --screws cases/case01/screws.csv --coords ras \\
        --out case_output/case01

See ``examples/clinical/README.md`` for the input formats.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from pedicle_planning.ai.candidates import (
    CandidateGridConfig,
    ConstraintContext,
    Candidate,
    candidate_features,
    optimize_screw_plan,
)
from pedicle_planning.ai.pedicle import detect_pedicle_isthmus
from pedicle_planning.ai.trajectory import LinearCalibration, ScrewPlan
from pedicle_planning.ai.vertebra_frame import estimate_vertebra_frame, image_axes_frame
from pedicle_planning.ai.vertebral_body import compute_vertebral_body_bmd


KEY_FEATURES = (
    "pedicle_interface_median",
    "pedicle_interface_p10",
    "pedicle_interface_medial_median",
    "pedicle_interface_lateral_median",
    "pedicle_interface_cortical_fraction",
    "transition_interface_median",
    "transition_interface_p10",
    "body_all_median",
    "tip_all_median",
    "tip_sphere_mean",
    "whole_interface_low_density_fraction",
    "in_bone_length",
    "match_angle_deg",
    "match_isthmus_offset_mm",
    "match_outer_clearance_mm",
    "match_inner_clearance_mm",
    "match_diameter_ratio",
)

CANDIDATE_COLUMNS = (
    "screw_id", "is_baseline", "pivot", "medial_offset_mm", "cranial_offset_mm",
    "medial_angle_deg", "cranial_angle_deg", "length_mm", "diameter_mm",
    "entry_x", "entry_y", "entry_z", "tip_x", "tip_y", "tip_z",
    "entry_snap_status", "entry_snap_shift_mm", "feasible", "violations", "risk",
)

REQUIRED_SCREW_COLUMNS = ("screw_id", "entry_x", "entry_y", "entry_z", "tip_x", "tip_y", "tip_z", "diameter")


# ----------------------------------------------------------------------------- inputs

def read_image(path: str) -> sitk.Image:
    """Read a NIfTI/NRRD/MHA file, or the first DICOM series in a directory."""
    p = Path(path)
    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(p))
        if not series:
            raise ValueError(f"no DICOM series found in {p.name}/")
        if len(series) > 1:
            print(f"[warn] {len(series)} DICOM series found; using the first one", file=sys.stderr)
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(p), series[0]))
        return reader.Execute()
    return sitk.ReadImage(str(p))


def binary_mask(mask: sitk.Image, label=None) -> sitk.Image:
    """Single-vertebra binary mask from a binary mask or a label map."""
    array = sitk.GetArrayViewFromImage(mask)
    selected = array != 0 if label is None else array == int(label)
    if not np.any(selected):
        raise ValueError(f"mask contains no voxels for label {label!r}")
    out = sitk.GetImageFromArray(selected.astype(np.uint8))
    out.CopyInformation(mask)
    return out


def to_lps(point, coords: str, ct: sitk.Image) -> tuple:
    """Convert a point given in the CSV coordinate convention to LPS physical coordinates."""
    x, y, z = (float(v) for v in point)
    if coords == "lps":
        return (x, y, z)
    if coords == "ras":
        return (-x, -y, z)
    if coords == "voxel":
        return tuple(ct.TransformContinuousIndexToPhysicalPoint((x, y, z)))
    raise ValueError(f"unknown coordinate convention {coords!r}")


def read_screws(path: str, coords: str, ct: sitk.Image, default_label=None) -> list:
    """Read baseline screws from CSV (see README for columns)."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("screw file has no rows")
    missing = [c for c in REQUIRED_SCREW_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"screw file is missing columns: {missing}")

    screws = []
    for row in rows:
        entry = to_lps((row["entry_x"], row["entry_y"], row["entry_z"]), coords, ct)
        tip = to_lps((row["tip_x"], row["tip_y"], row["tip_z"]), coords, ct)
        diameter = float(row["diameter"])
        length = row.get("length")
        if length not in (None, ""):
            direction = np.subtract(tip, entry)
            plan = ScrewPlan(entry, tuple(direction), float(length), diameter)
        else:
            plan = ScrewPlan.from_entry_tip(entry, tip, diameter)
        label = row.get("label")
        label = int(label) if label not in (None, "") else default_label
        screws.append({"screw_id": row["screw_id"], "plan": plan, "label": label})
    ids = [s["screw_id"] for s in screws]
    if len(set(ids)) != len(ids):
        raise ValueError("screw_id values must be unique")
    return screws


def load_risk_function(spec: str):
    """None, 'demo', 'package.module:function' or 'path/to/file.py:function'."""
    if spec in (None, "", "none"):
        return None
    if spec == "demo":
        return demo_risk
    if ":" not in spec:
        raise ValueError("--risk must be 'none', 'demo', 'module:function' or 'file.py:function'")
    target, func_name = spec.rsplit(":", 1)
    if target.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location("user_risk_model", target)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, func_name)


def demo_risk(features: dict) -> float:
    """
    PLACEHOLDER risk score for exercising the pipeline only.

    Not trained, not validated, and not a loosening probability: lower
    interface density, larger pedicle misalignment and shorter purchase
    increase the score.
    """
    density = features.get("whole_interface_median", float("nan"))
    angle = features.get("match_angle_deg", 0.0)
    purchase = features.get("in_bone_length", 0.0)
    if not np.isfinite(density):
        return float("nan")
    return float(1.0 / (1.0 + math.exp((density - 250.0) / 75.0))
                 + 0.01 * (angle if np.isfinite(angle) else 0.0) - 0.002 * purchase)


def load_grid(path: str) -> CandidateGridConfig:
    if not path:
        return CandidateGridConfig()
    with open(path, encoding="utf-8") as handle:
        overrides = json.load(handle)
    valid = {f.name for f in dataclasses.fields(CandidateGridConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"unknown grid fields: {sorted(unknown)}")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
    return CandidateGridConfig(**values)


# ----------------------------------------------------------------------------- outputs

def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _frame_dict(vframe) -> dict:
    return {
        "method": vframe.method,
        "reliable": vframe.reliable,
        "symmetry_score": vframe.symmetry_score,
        "origin": vframe.origin,
        "left": vframe.left,
        "anterior": vframe.anterior,
        "cranial": vframe.cranial,
        "angles_to_image_axes": vframe.angles_to(),
    }


def _pedicle_dict(pedicle, plan) -> dict:
    isthmus = pedicle.isthmus
    return {
        "isthmus_point": pedicle.isthmus_point,
        "axis": pedicle.axis,
        "axis_status": pedicle.axis_status,
        "axis_fit_tilt_deg": pedicle.axis_fit_tilt_deg,
        "outer_width_mm": pedicle.outer_width,
        "outer_height_mm": pedicle.outer_height,
        "inner_width_mm": pedicle.inner_width,
        "inner_height_mm": pedicle.inner_height,
        "outer_min_width_mm": pedicle.outer_min_width,
        "inner_min_width_mm": pedicle.inner_min_width,
        "cortex_resolved": isthmus.cortex_resolved,
        "cancellous_hu": isthmus.cancellous_hu,
        "cortex_contrast_hu": isthmus.cortex_contrast_hu,
        "cortex_threshold_hu": isthmus.cortex_threshold_hu,
        "width_profile": pedicle.profile,
        "segment_bounds_mm": pedicle.segment_bounds_for(plan),
        "baseline_match": pedicle.trajectory_match(plan),
    }


def write_candidates_csv(path: Path, screw_id: str, result, all_features: bool) -> None:
    feature_names = list(KEY_FEATURES)
    if all_features:
        feature_names = sorted({k for e in result.evaluations for k in (e.features or {})})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(CANDIDATE_COLUMNS) + feature_names)
        for e in result.evaluations:
            c, plan = e.candidate, e.candidate.plan
            row = [screw_id, c.is_baseline, c.pivot, c.medial_offset_mm, c.cranial_offset_mm,
                   c.medial_angle_deg, c.cranial_angle_deg, plan.length, plan.diameter,
                   *plan.entry, *plan.tip, c.entry_snap_status, c.entry_snap_shift_mm,
                   e.feasible, "; ".join(e.violations), e.risk]
            feats = e.features or {}
            row += [feats.get(name, "") for name in feature_names]
            writer.writerow(["" if isinstance(v, float) and not np.isfinite(v) else v for v in row])


def write_qc_figure(path: Path, ct: sitk.Image, mask: sitk.Image, baseline: ScrewPlan,
                    best: ScrewPlan, focus_point) -> bool:
    """Axial and sagittal QC slices (image index planes) through the focus point."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    array = sitk.GetArrayViewFromImage(ct)
    mask_array = sitk.GetArrayViewFromImage(mask) if mask.GetSize() == ct.GetSize() else None
    spacing = ct.GetSpacing()
    fi, fj, fk = (int(round(v)) for v in ct.TransformPhysicalPointToContinuousIndex(tuple(focus_point)))
    size = ct.GetSize()
    fi, fj, fk = (min(max(v, 0), n - 1) for v, n in zip((fi, fj, fk), size))

    def to_index(point):
        return np.asarray(ct.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in point)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    # For LPS images with near-identity direction: axial shows anterior up, sagittal shows cranial up.
    panels = (
        ("axial (k = %d)" % fk, array[fk, :, :], None if mask_array is None else mask_array[fk, :, :],
         (0, 1), spacing[1] / spacing[0], "upper"),
        ("sagittal (i = %d)" % fi, array[:, :, fi], None if mask_array is None else mask_array[:, :, fi],
         (1, 2), spacing[2] / spacing[1], "lower"),
    )
    for ax, (title, image2d, mask2d, (a, b), aspect, origin) in zip(axes, panels):
        ax.imshow(image2d, cmap="gray", vmin=-200, vmax=1200, origin=origin, aspect=aspect)
        if mask2d is not None and np.any(mask2d):
            ax.contour(mask2d != 0, levels=[0.5], colors="deepskyblue", linewidths=0.6)
        for plan, color, label in ((baseline, "white", "baseline"), (best, "orange", "best")):
            if plan is None:
                continue
            e, t = to_index(plan.entry), to_index(plan.tip)
            ax.plot([e[a], t[a]], [e[b], t[b]], color=color, linewidth=1.5, label=label)
            ax.plot([e[a]], [e[b]], "o", color=color, markersize=3)
        ax.set_title(title + " - projected screw axes")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("QC only - image index planes, screws projected; research prototype")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- pipeline

def process_case(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct = read_image(args.ct)
    mask_image = read_image(args.mask)
    screws = read_screws(args.screws, args.coords, ct, args.label)
    risk_fn = load_risk_function(args.risk)
    grid = load_grid(args.grid)
    if args.pivot:
        grid = dataclasses.replace(grid, PIVOT=args.pivot)
    if args.no_snap:
        grid = dataclasses.replace(grid, SNAP_ENTRY=False)
    calibration = None
    if args.calibration:
        slope, intercept = (float(v) for v in args.calibration.split(","))
        calibration = LinearCalibration(slope, intercept)

    summary = {
        "note": "Research prototype output; not for clinical decision making. Input paths are not recorded.",
        "ct_geometry": {"size": ct.GetSize(), "spacing": ct.GetSpacing(), "direction": ct.GetDirection()},
        "options": {"coords": args.coords, "risk": "none" if risk_fn is None else
                    ("demo (placeholder)" if risk_fn is demo_risk else "user"),
                    "grid": dataclasses.asdict(grid),
                    "calibration": None if calibration is None else dataclasses.asdict(calibration)},
        "vertebrae": {},
        "screws": {},
    }

    vertebrae = {}
    for label in sorted({s["label"] for s in screws}, key=lambda v: (v is None, v)):
        key = "mask" if label is None else f"label_{label}"
        print(f"[vertebra {key}] building frame and body density")
        mask = binary_mask(mask_image, label)
        vframe = estimate_vertebra_frame(mask)
        if not vframe.reliable and not args.keep_unreliable_frame:
            print(f"[warn] vertebra {key}: symmetry score {vframe.symmetry_score:.2f} below threshold; "
                  "falling back to image axes", file=sys.stderr)
            estimated = vframe
            vframe = image_axes_frame(mask)
        else:
            estimated = None
        body = compute_vertebral_body_bmd(ct, mask, vframe, calibration=calibration)
        vertebrae[label] = (mask, vframe, body)
        summary["vertebrae"][key] = {
            "frame": _frame_dict(vframe),
            "estimated_frame_rejected": None if estimated is None else _frame_dict(estimated),
            "body_bmd": body.to_flat_dict(),
        }

    for screw in screws:
        sid, plan = screw["screw_id"], screw["plan"]
        mask, vframe, body = vertebrae[screw["label"]]
        record = {"side": vframe.side_of(plan.entry), "baseline": {
            "entry": plan.entry, "tip": plan.tip, "length_mm": plan.length, "diameter_mm": plan.diameter}}
        summary["screws"][sid] = record
        print(f"[screw {sid}] side={record['side']}")
        try:
            frame = vframe.anatomical_frame_for(plan.entry)
            try:
                pedicle = detect_pedicle_isthmus(mask, plan, image=ct, frame=frame)
                record["pedicle"] = _pedicle_dict(pedicle, plan)
            except ValueError as exc:
                pedicle = None
                record["pedicle_error"] = str(exc)
                print(f"[warn] screw {sid}: isthmus detection failed: {exc}", file=sys.stderr)
            if grid.PIVOT == "isthmus" and pedicle is None:
                raise ValueError('PIVOT="isthmus" needs a detected isthmus')

            ctx = ConstraintContext(image=ct, bone_mask=mask, pedicle=pedicle, frame=frame)
            baseline_features = candidate_features(ct, Candidate(plan, is_baseline=True), ctx, calibration)
            with open(out_dir / f"features_{sid}_baseline.json", "w", encoding="utf-8") as handle:
                json.dump(_jsonable(baseline_features), handle, indent=2)
            record["baseline_key_features"] = {k: baseline_features.get(k) for k in KEY_FEATURES}

            result = optimize_screw_plan(
                ct, plan, frame, risk_fn or (lambda f: float("nan")), bone_mask=mask, pedicle=pedicle,
                grid=grid, calibration=calibration, patient_features=body.to_flat_dict())
            record["optimization"] = result.summary()
            write_candidates_csv(out_dir / f"candidates_{sid}.csv", sid, result, args.all_features)
            if not args.no_qc:
                best = result.best.candidate.plan if (result.best is not None and risk_fn is not None) else None
                focus = pedicle.isthmus_point if pedicle is not None else plan.entry
                qc_path = out_dir / f"qc_{sid}.png"
                try:
                    record["qc_figure"] = write_qc_figure(qc_path, ct, mask, plan, best, focus)
                except OSError as exc:  # e.g. the old figure is open in a viewer on Windows
                    record["qc_figure"] = False
                    record["qc_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[warn] screw {sid}: QC figure not written ({exc}); close any program showing "
                          f"{qc_path.name} and rerun, or use another --out folder", file=sys.stderr)
        except Exception as exc:  # keep processing the other screws
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            print(f"[error] screw {sid}: {record['error']}", file=sys.stderr)

    with open(out_dir / "case_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ct", required=True, help="CT volume (NIfTI/NRRD/MHA) or DICOM series directory, in HU")
    parser.add_argument("--mask", required=True, help="vertebra mask: binary single vertebra or label map")
    parser.add_argument("--label", type=int, default=None,
                        help="label value of the vertebra in a label map (overridden by a 'label' CSV column)")
    parser.add_argument("--screws", required=True, help="CSV with baseline screws (see README)")
    parser.add_argument("--coords", choices=("lps", "ras", "voxel"), default="lps",
                        help="coordinate convention of the CSV points (3D Slicer uses RAS)")
    parser.add_argument("--out", default="case_output", help="output directory")
    parser.add_argument("--risk", default="none",
                        help="'none' (features and feasibility only), 'demo' (placeholder), "
                             "'module:function' or 'file.py:function' taking a feature dict")
    parser.add_argument("--pivot", choices=("entry", "isthmus"), default=None, help="override grid PIVOT")
    parser.add_argument("--no-snap", action="store_true", help="do not snap candidate entries to bone")
    parser.add_argument("--grid", default=None, help="JSON file with CandidateGridConfig overrides")
    parser.add_argument("--calibration", default=None, help="linear HU->BMD calibration 'slope,intercept'")
    parser.add_argument("--keep-unreliable-frame", action="store_true",
                        help="use the estimated vertebra frame even when flagged unreliable")
    parser.add_argument("--all-features", action="store_true", help="write all features to candidate CSVs")
    parser.add_argument("--no-qc", action="store_true", help="skip QC figures")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = process_case(args)
    failed = [sid for sid, rec in summary["screws"].items() if "error" in rec]
    for sid, rec in summary["screws"].items():
        opt = rec.get("optimization", {})
        print(f"[done] {sid}: feasible {opt.get('n_feasible')}/{opt.get('n_candidates')}, "
              f"baseline risk {opt.get('baseline_risk')}, best risk {opt.get('best_risk')}")
    print(f"outputs written to {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
