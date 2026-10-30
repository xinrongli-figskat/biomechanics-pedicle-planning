"""Check a locally stored CT (and optionally a mask) before running the case pipeline.

Prints geometry, orientation and intensity statistics, and flags common
problems: intensities that are not in HU (e.g. raw unsigned values from a
converter that ignored the DICOM rescale), very thick slices, and masks
that do not overlap the CT. Nothing is written or uploaded; header text
fields that could identify a patient are not printed.

Example
-------
    python examples/clinical/inspect_inputs.py --ct D:\\cases\\case01\\ct.nii
    python examples/clinical/inspect_inputs.py --ct ct.nii --mask seg\\vertebrae_L4.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _read(path: str) -> sitk.Image:
    p = Path(path)
    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(p))
        if not series:
            raise ValueError(f"no DICOM series found in {p.name}/")
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(p), series[0]))
        return reader.Execute()
    return sitk.ReadImage(str(p))


def _orientation(image: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())


def describe_ct(ct: sitk.Image, max_samples: int = 2_000_000) -> tuple:
    """Return (lines, warnings) describing the CT."""
    array = sitk.GetArrayViewFromImage(ct)
    flat = array.ravel()
    if flat.size > max_samples:
        flat = flat[:: flat.size // max_samples + 1]
    flat = flat.astype(np.float64)
    p = np.percentile(flat, [0.5, 1, 50, 99, 99.9])
    spacing = ct.GetSpacing()
    size = ct.GetSize()
    lines = [
        f"size (x, y, z)       : {size}",
        f"spacing (mm)         : {tuple(round(s, 4) for s in spacing)}",
        f"extent (mm)          : {tuple(round(n * s, 1) for n, s in zip(size, spacing))}",
        f"orientation code     : {_orientation(ct)}   (direction {tuple(round(d, 3) for d in ct.GetDirection())})",
        f"pixel type           : {ct.GetPixelIDTypeAsString()}",
        f"min / max            : {float(np.min(flat)):.0f} / {float(np.max(flat)):.0f}",
        f"percentiles 1/50/99/99.9 : {p[1]:.0f} / {p[2]:.0f} / {p[3]:.0f} / {p[4]:.0f}",
    ]
    warnings = []
    if p[0] >= -500:
        warnings.append("lowest intensities are above -500: values may not be HU (air should be about -1000). "
                        "Check the DICOM rescale slope/intercept used by the converter.")
    if p[0] >= 0 and np.max(flat) > 1500:
        warnings.append("all values non-negative with a large maximum: looks like raw unsigned data; "
                        "HU = value * slope + intercept is probably missing (often intercept -1024).")
    if p[4] < 700:
        warnings.append("almost no voxels above 700 HU: cortical bone is not visible at the expected level.")
    if spacing[2] > 1.5:
        warnings.append(f"slice spacing {spacing[2]:.2f} mm is thick; pedicle and interface features will be "
                        "blurred (<= 1 mm recommended).")
    if max(spacing[0], spacing[1]) > 1.0:
        warnings.append("in-plane pixel size above 1 mm; pedicle widths will be coarse.")
    return lines, warnings


def describe_mask(mask: sitk.Image, ct: sitk.Image = None, max_labels: int = 60) -> tuple:
    array = sitk.GetArrayViewFromImage(mask)
    labels, counts = np.unique(array, return_counts=True)
    voxel_ml = float(np.prod(mask.GetSpacing())) / 1000.0
    lines = [
        f"size / spacing       : {mask.GetSize()} / {tuple(round(s, 4) for s in mask.GetSpacing())}",
        f"orientation code     : {_orientation(mask)}",
        f"labels (value: ml)   : " + ", ".join(
            f"{int(v)}: {c * voxel_ml:.1f}" for v, c in list(zip(labels, counts))[:max_labels] if v != 0),
    ]
    warnings = []
    if len(labels) == 1:
        warnings.append("mask is empty.")
    if ct is not None:
        same_grid = (mask.GetSize() == ct.GetSize()
                     and np.allclose(mask.GetSpacing(), ct.GetSpacing(), atol=1e-4)
                     and np.allclose(mask.GetOrigin(), ct.GetOrigin(), atol=1e-3)
                     and np.allclose(mask.GetDirection(), ct.GetDirection(), atol=1e-4))
        lines.append(f"same grid as CT      : {same_grid}")
        nz = np.argwhere(array != 0)
        if nz.size:
            centre = nz.mean(axis=0)[::-1]
            point = mask.TransformContinuousIndexToPhysicalPoint(tuple(float(c) for c in centre))
            index = ct.TransformPhysicalPointToContinuousIndex(point)
            inside = all(0 <= i <= n - 1 for i, n in zip(index, ct.GetSize()))
            lines.append(f"mask centre inside CT: {inside}")
            if not inside:
                warnings.append("mask centre lies outside the CT volume: the two files are not in the same space.")
    return lines, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ct", required=True, help="CT file or DICOM series folder")
    parser.add_argument("--mask", default=None, help="optional vertebra mask or label map")
    args = parser.parse_args(argv)

    ct = _read(args.ct)
    lines, warnings = describe_ct(ct)
    print("== CT")
    print("\n".join(lines))
    if args.mask:
        mask_lines, mask_warnings = describe_mask(_read(args.mask), ct)
        print("== mask")
        print("\n".join(mask_lines))
        warnings += mask_warnings
    print("== checks")
    if warnings:
        for w in warnings:
            print(f"[warn] {w}")
    else:
        print("no problems detected")
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
