"""Synthetic, bilaterally symmetric vertebra phantom shared by the tests."""

import math

import numpy as np
import SimpleITK as sitk


CENTER = (60.0, 60.0, 60.0)


def _true_axes(axial=0.0, coronal=0.0, sagittal=0.0):
    """Columns are the true (left, anterior, cranial) axes in LPS physical coordinates."""
    left, anterior, cranial = np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0])
    t = math.radians(axial)
    anterior = math.cos(t) * anterior - math.sin(t) * left
    left = np.cross(cranial, anterior)
    t = math.radians(coronal)
    cranial = math.cos(t) * cranial - math.sin(t) * left
    left = np.cross(cranial, anterior)
    t = math.radians(sagittal)
    cranial = math.cos(t) * cranial + math.sin(t) * anterior
    anterior = np.cross(left, cranial)
    return np.stack([left, anterior, cranial], axis=1)


def _local_coords(axes, size=120):
    k, j, i = np.meshgrid(*(np.arange(size),) * 3, indexing="ij")
    p = np.stack([i, j, k], -1).reshape(-1, 3).astype(np.float64) - np.asarray(CENTER)
    q = p @ axes
    return q[:, 0], q[:, 1], q[:, 2]


def _synthetic_vertebra(axial=0.0, coronal=0.0, sagittal=0.0, spur=False, remove=None):
    """
    Bilaterally symmetric vertebra in local (left, anterior, cranial) coordinates:
    elliptical body, pedicles, lamina, caudally sloping spinous process and
    transverse processes; rotated by the given angles about CENTER (1 mm voxels).
    """
    axes = _true_axes(axial, coronal, sagittal)
    l, a, s = _local_coords(axes)
    body = ((l / 22) ** 2 + ((a - 12) / 16) ** 2 <= 1) & (np.abs(s) <= 12)
    # hourglass pedicles: narrowest (half-width 4 mm) at a = -8, widening toward lamina and body
    pedicle_half_width = 4.0 + 0.04 * (a + 8.0) ** 2
    pedicles = ((((np.abs(l) - 12) / pedicle_half_width) ** 2 + (s / (1.5 * pedicle_half_width)) ** 2 <= 1)
                & (a >= -14) & (a <= 0.5))  # overlaps the body: no gap between pedicle and body
    lamina = (np.abs(l) <= 14) & (a >= -22) & (a <= -14) & (np.abs(s) <= 8)
    drop = 0.3 * np.clip(-a - 22, 0, None)
    spinous = (np.abs(l) <= 3) & (a >= -45) & (a <= -22) & (s >= -12 - drop) & (s <= 2 - drop)
    transverse = (np.abs(l) >= 14) & (np.abs(l) <= 35) & (a >= -12) & (a <= -6) & (np.abs(s) <= 4)
    bone = body | pedicles | lamina | spinous | transverse
    if spur:
        bone |= (np.abs(l) <= 3) & (a >= 22) & (a <= 26) & (s >= 12) & (s <= 16)
    if remove == "right_posterior":
        bone &= ~((l < -10) & (a < -1))
    elif remove == "right_half":
        bone &= ~(l < 0)
    mask = sitk.GetImageFromArray(bone.reshape(120, 120, 120).astype(np.uint8))
    return mask, axes


def synthetic_vertebra_ct(axial=0.0, coronal=0.0, sagittal=0.0, cortex_mm=1.0):
    """
    CT companion of _synthetic_vertebra: 1000 HU cortical shell, vertebral-body
    cancellous bone of 120 + 8 * |height| HU (denser toward the endplates),
    300 HU cancellous bone in the posterior elements.
    Returns (ct, mask, axes).
    """
    mask, axes = _synthetic_vertebra(axial, coronal, sagittal)
    l, a, s = _local_coords(axes)
    bone = sitk.GetArrayFromImage(mask).ravel() > 0
    body = (((l / 22) ** 2 + ((a - 12) / 16) ** 2 <= 1) & (np.abs(s) <= 12))
    eroded = sitk.GetArrayFromImage(sitk.BinaryErode(mask, [int(round(cortex_mm))] * 3)).ravel() > 0
    hu = np.zeros(bone.shape, dtype=np.float32)
    hu[bone] = 1000.0
    interior = bone & eroded
    hu[interior] = 300.0
    hu[interior & body] = (120.0 + 8.0 * np.abs(s))[interior & body]
    ct = sitk.GetImageFromArray(hu.reshape(120, 120, 120))
    ct.CopyInformation(mask)
    return ct, mask, axes
