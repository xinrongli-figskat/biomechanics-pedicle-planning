"""Generate reference (baseline) pedicle screws from vertebra segmentations.

For data without a surgeon's plan (e.g. public datasets such as VerSe), this
writes a screw CSV for ``run_case_pipeline.py``: one standard straight-forward
trajectory per pedicle, through the pedicle isthmus centre, parallel to the
endplates, with the medial angle that keeps the axis farthest from the
pedicle wall. These are reproducible reference trajectories, not surgical
plans.

Example
-------
    python examples/clinical/make_baseline_screws.py --mask cases/verse01/seg.nii.gz \\
        --labels 22 23 24 --out cases/verse01/baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from pedicle_planning.ai.baseline import BaselineConfig, generate_baseline_screws
from pedicle_planning.ai.vertebra_frame import estimate_vertebra_frame, image_axes_frame


def _binary(mask: sitk.Image, label=None) -> sitk.Image:
    array = sitk.GetArrayViewFromImage(mask)
    selected = array != 0 if label is None else array == int(label)
    if not np.any(selected):
        raise ValueError(f"mask contains no voxels for label {label!r}")
    out = sitk.GetImageFromArray(selected.astype(np.uint8))
    out.CopyInformation(mask)
    return out


def _round(value, digits=3):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_round(v, digits) for v in value]
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def make_baselines(mask_path: str, labels, sides, out_dir: str, config: BaselineConfig = None) -> dict:
    mask_image = sitk.ReadImage(str(mask_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report, rows = {}, []
    for label in labels:
        key = "mask" if label is None else f"label_{label}"
        try:
            mask = _binary(mask_image, label)
            frame = estimate_vertebra_frame(mask)
            if not frame.reliable:
                print(f"[warn] {key}: vertebra frame unreliable (symmetry {frame.symmetry_score:.2f}); "
                      "using image axes", file=sys.stderr)
                frame = image_axes_frame(mask)
            results = generate_baseline_screws(mask, frame, sides, config)
        except ValueError as exc:
            report[key] = {"error": str(exc)}
            print(f"[error] {key}: {exc}", file=sys.stderr)
            continue
        report[key] = {"frame_method": frame.method, "symmetry_score": _round(frame.symmetry_score), "screws": {}}
        for side, result in results.items():
            if isinstance(result, str):
                report[key]["screws"][side] = {"error": result}
                print(f"[error] {key} {side}: {result}", file=sys.stderr)
                continue
            sid = f"{'v' if label is None else label}_{side}"
            plan = result.plan
            rows.append([sid, "" if label is None else int(label), *plan.entry, *plan.tip,
                         plan.diameter, plan.length])
            report[key]["screws"][side] = {
                "screw_id": sid,
                "length_mm": plan.length,
                "diameter_mm": plan.diameter,
                "medial_angle_deg": _round(result.medial_angle_deg, 1),
                "isthmus_width_mm": _round(result.isthmus_width_mm, 2),
                "wall_clearance_mm": _round(result.wall_clearance_mm, 2),
                "in_bone_length_mm": _round(result.in_bone_length_mm, 1),
                "entry": _round(plan.entry),
                "tip": _round(plan.tip),
            }
            print(f"[{key} {side}] {sid}: {plan.length:.0f} x {plan.diameter:.1f} mm, medial "
                  f"{result.medial_angle_deg:.0f} deg, isthmus {result.isthmus_width_mm:.1f} mm, "
                  f"min wall distance {result.wall_clearance_mm:.2f} mm")

    with open(out / "baseline_screws.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["screw_id", "label", "entry_x", "entry_y", "entry_z", "tip_x", "tip_y", "tip_z",
                         "diameter", "length"])
        writer.writerows(rows)
    with open(out / "baseline_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mask", required=True, help="binary single-vertebra mask or label map")
    parser.add_argument("--labels", type=int, nargs="*", default=None,
                        help="vertebra labels in a label map (default: treat the mask as one vertebra)")
    parser.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    parser.add_argument("--out", required=True, help="output folder (baseline_screws.csv, baseline_report.json)")
    args = parser.parse_args(argv)
    labels = args.labels if args.labels else [None]
    report = make_baselines(args.mask, labels, tuple(args.sides), args.out)
    n_ok = sum(1 for v in report.values() for s in v.get("screws", {}).values() if "error" not in s)
    print(f"{n_ok} screws written to {Path(args.out) / 'baseline_screws.csv'}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
