"""
baseline.py
------------------------------------
由椎体分割自动生成基准（参考）置钉方案

当没有医生经验方案时（如公开数据集），为每侧椎弓根生成一条标准直线型轨迹，
作为特征提取与候选优化的初始解。它是可复现的参考轨迹，不代表医生方案。

步骤（均在椎体解剖坐标系 left / anterior / cranial 下进行）：
1. 椎管：沿正中矢状线自椎体中心向后，离开骨处为椎体后壁，再次进入骨处为椎板；
2. 椎弓根中心：在椎体后壁与椎板前缘（留 LAMINA_MARGIN_MM）之间逐层取冠状面，
   在中线一侧的骨区域内取内切圆最大的点（距离变换极大值）。横突等细长附属
   结构不改变内切圆位置；
3. 峡部 = 内切圆半径最小的冠状面；轨迹过峡部中心，矢状面平行于终板（头倾 0°），
   横断面内倾角在 MEDIAL_ANGLE_RANGE_DEG 内逐度搜索，取使峡部前后
   CLEARANCE_HALF_SPAN_MM 内轨迹到骨壁的最小距离最大者（“离皮质最远”），
   同样安全的角度取安全锥的中间。
   （由几层椎弓根中心拟合方向在椎弓根很短时不稳定，已弃用。）
4. 进钉点 = 自峡部中心沿轨迹向后到达骨表面处；长度取骨内全长的 LENGTH_FRACTION
   且钉尖距前皮质不少于 TIP_MARGIN_MM，向下取到规格；直径取不超过
   DIAMETER_FRACTION × 峡部宽度（2 × 内切圆半径）的规格。
"""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from .pedicle import IsthmusDetectionConfig, measure_section
from .trajectory import ScrewPlan, sample_mask_nearest


@dataclass
class BaselineConfig:
    """自动基准方案配置参数（长度单位为毫米）。"""

    GRID_STEP: float = 0.5
    """冠状面采样网格间距。"""

    CORONAL_STEP: float = 0.5
    """冠状面之间的前后间距。"""

    LAMINA_MARGIN_MM: float = 2.0
    """冠状面不取到椎板前缘以内该距离（椎弓根-椎板交界处内切圆会落入椎板）。"""


    LATERAL_RANGE: tuple = (2.0, 35.0)
    """在中线一侧搜索椎弓根的内外范围（距正中矢状面）。"""

    CRANIAL_HALF_RANGE: float = 20.0
    """冠状面在头尾方向的搜索半范围（相对椎体中心）。"""

    MAX_WALK_MM: float = 80.0
    """沿直线寻找椎管、进钉点与前皮质的最大距离。"""

    LENGTH_FRACTION: float = 0.8
    """螺钉长度占骨内全长的比例上限。"""

    TIP_MARGIN_MM: float = 3.0
    """钉尖距前皮质的最小距离。"""

    DIAMETER_FRACTION: float = 0.8
    """螺钉直径占峡部宽度的比例上限。"""

    EXIT_RUN_MM: float = 2.5
    """沿轨迹连续离开骨掩膜达到该长度才算到达骨表面（跨过分割中的细小缝隙）。"""

    MEDIAL_ANGLE_RANGE_DEG: tuple = (-10.0, 35.0)
    """横断面内倾角的搜索范围（向中线为正）。"""

    ANGLE_STEP_DEG: float = 1.0

    CLEARANCE_HALF_SPAN_MM: float = 8.0
    """在峡部前后该范围内评估轨迹到椎弓根壁的最小距离。"""

    CLEARANCE_TOLERANCE_MM: float = 0.25
    """最小骨壁距离与最优值相差不超过该值的角度视为同样安全（约一个射线采样步长）。"""

    LENGTHS_MM: tuple = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0)
    DIAMETERS_MM: tuple = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5)

    def validate(self) -> None:
        for name in ("GRID_STEP", "CORONAL_STEP", "CRANIAL_HALF_RANGE", "MAX_WALK_MM", "TIP_MARGIN_MM",
                     "LAMINA_MARGIN_MM"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} ({getattr(self, name)}) must be positive")
        lo_a, hi_a = self.MEDIAL_ANGLE_RANGE_DEG
        if not -90.0 < lo_a < hi_a < 90.0:
            raise ValueError("MEDIAL_ANGLE_RANGE_DEG must satisfy -90 < lo < hi < 90")
        if self.CLEARANCE_TOLERANCE_MM < 0:
            raise ValueError("CLEARANCE_TOLERANCE_MM must be non-negative")
        if not self.ANGLE_STEP_DEG > 0 or not self.CLEARANCE_HALF_SPAN_MM > 0:
            raise ValueError("ANGLE_STEP_DEG and CLEARANCE_HALF_SPAN_MM must be positive")
        if self.EXIT_RUN_MM < 0:
            raise ValueError("EXIT_RUN_MM must be non-negative")
        lo, hi = self.LATERAL_RANGE
        if not 0.0 <= lo < hi:
            raise ValueError(f"LATERAL_RANGE ({self.LATERAL_RANGE}) must satisfy 0 <= lo < hi")
        for name in ("LENGTH_FRACTION", "DIAMETER_FRACTION"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} ({getattr(self, name)}) must be in (0, 1]")
        if not self.LENGTHS_MM or not self.DIAMETERS_MM:
            raise ValueError("LENGTHS_MM and DIAMETERS_MM must not be empty")


@dataclass
class BaselineScrew:
    """自动生成的一侧基准方案及其依据。"""

    side: str
    plan: ScrewPlan
    isthmus_center: np.ndarray
    isthmus_width_mm: float
    medial_angle_deg: float
    wall_clearance_mm: float
    in_bone_length_mm: float
    pedicle_centers: np.ndarray


def _local_points(frame, l, a, s) -> np.ndarray:
    """椎体坐标 (left, anterior, cranial) → 物理坐标；参数可广播。"""
    l, a, s = np.broadcast_arrays(np.asarray(l, float), np.asarray(a, float), np.asarray(s, float))
    return (np.asarray(frame.origin)
            + l[..., None] * np.asarray(frame.left)
            + a[..., None] * np.asarray(frame.anterior)
            + s[..., None] * np.asarray(frame.cranial))


def _walk(mask, start, direction, max_mm, step) -> np.ndarray:
    radii = np.arange(0.0, max_mm + 1e-9, step)
    points = np.asarray(start) + np.outer(radii, direction)
    return radii, sample_mask_nearest(mask, points)


def _first(values: np.ndarray, start: int = 0) -> int:
    hits = np.flatnonzero(values[start:])
    return int(hits[0]) + start if hits.size else -1


def _first_run(values: np.ndarray, run: int) -> int:
    """第一个连续 run 个 True 的起点；没有则 -1。"""
    if run <= 1:
        return _first(values)
    window = np.convolve(values.astype(int), np.ones(run, dtype=int), mode="valid")
    hits = np.flatnonzero(window == run)
    return int(hits[0]) if hits.size else -1


def find_canal(mask: sitk.Image, frame, config: BaselineConfig) -> tuple:
    """沿正中矢状线向后：返回 (椎体后壁 a, 椎板前缘 a)，以椎体坐标的前后坐标表示。"""
    direction = -np.asarray(frame.anterior)
    radii, inside = _walk(mask, frame.origin, direction, config.MAX_WALK_MM, config.GRID_STEP)
    if not inside[0]:
        raise ValueError("vertebra frame origin is not inside bone")
    exit_idx = _first(~inside)
    if exit_idx < 0:
        raise ValueError("no spinal canal found behind the vertebral body")
    reenter = _first(inside, exit_idx)
    if reenter < 0:
        raise ValueError("no posterior arch found behind the spinal canal")
    return -float(radii[exit_idx]), -float(radii[reenter])


def _pedicle_center_in_plane(mask, frame, a, side_sign, config) -> tuple:
    """冠状面 a 上，side_sign 侧骨区域的最大内切圆中心 (l, s)（近最大值平台的重心）与半径。"""
    lo, hi = config.LATERAL_RANGE
    ls = side_sign * np.arange(lo, hi + 1e-9, config.GRID_STEP)
    ss = np.arange(-config.CRANIAL_HALF_RANGE, config.CRANIAL_HALF_RANGE + 1e-9, config.GRID_STEP)
    gl, gs = np.meshgrid(ls, ss, indexing="ij")
    inside = sample_mask_nearest(mask, _local_points(frame, gl, a, gs).reshape(-1, 3)).reshape(gl.shape)
    if not inside.any():
        return None
    padded = np.pad(inside.astype(np.uint8), 1)
    image = sitk.GetImageFromArray(padded)
    image.SetSpacing((config.GRID_STEP, config.GRID_STEP))
    distance = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        image, insideIsPositive=True, squaredDistance=False, useImageSpacing=True))[1:-1, 1:-1]
    distance[~inside] = -np.inf
    peak = float(distance.max())
    # the distance map is flat near its maximum: use the centroid of the near-maximal plateau
    # (connected to the maximum) instead of a single, grid-quantized argmax
    i, j = np.unravel_index(int(np.argmax(distance)), distance.shape)
    plateau = (distance >= 0.9 * peak).astype(np.uint8)
    labels = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(plateau)))
    region = labels == labels[i, j]
    return float(gl[region].mean()), float(gs[region].mean()), peak


def _min_wall_distance(mask, center, direction, half_span, step, ray_config) -> float:
    """沿直线在 center 前后 half_span 内逐层测量到骨壁的最小射线距离（未闭合方向不计）。"""
    worst = float("inf")
    for t in np.arange(-half_span, half_span + 1e-9, step):
        section = measure_section(mask, center + t * direction, direction, t, None, None, ray_config)
        if not section.center_in_bone:
            return -float("inf")
        worst = min(worst, float(np.min(section.outer)))
    return worst


def safest_medial_angle(mask, frame, center, side_sign, config) -> tuple:
    """
    过峡部中心、头倾 0° 的直线中，使沿椎弓根到骨壁的最小距离最大的内倾角；
    多个角度同样安全（相差不超过 CLEARANCE_TOLERANCE_MM）时取这一安全锥的中间。

    Returns
    -------
    tuple
        (内倾角（度）, 该角度下的最小骨壁距离（毫米）)
    """
    ray_config = IsthmusDetectionConfig()
    medial = -side_sign * np.asarray(frame.left)
    anterior = np.asarray(frame.anterior)
    lo, hi = config.MEDIAL_ANGLE_RANGE_DEG
    angles = np.arange(lo, hi + 1e-9, config.ANGLE_STEP_DEG)
    clearances = np.array([
        _min_wall_distance(mask, center, np.cos(np.radians(a)) * anterior + np.sin(np.radians(a)) * medial,
                           config.CLEARANCE_HALF_SPAN_MM, 1.0, ray_config)
        for a in angles
    ])
    best = int(np.argmax(clearances))
    if not np.isfinite(clearances[best]):
        return float(angles[best]), float(clearances[best])
    # a short, wide pedicle is safe over a cone of angles: take the middle of the contiguous run of
    # near-best angles around the optimum instead of a noisy single maximum
    ok = clearances >= clearances[best] - config.CLEARANCE_TOLERANCE_MM
    lo_i = hi_i = best
    while lo_i - 1 >= 0 and ok[lo_i - 1]:
        lo_i -= 1
    while hi_i + 1 < len(angles) and ok[hi_i + 1]:
        hi_i += 1
    middle = (lo_i + hi_i) // 2
    return float(angles[middle]), float(clearances[middle])


def _floor_to(values, limit) -> float:
    allowed = [v for v in sorted(values) if v <= limit + 1e-9]
    return float(allowed[-1]) if allowed else float("nan")


def generate_baseline_screw(mask: sitk.Image, frame, side: str,
                            config: BaselineConfig = None) -> BaselineScrew:
    """
    为一侧椎弓根生成基准方案

    Parameters
    ----------
    mask : sitk.Image
        单个椎体（含后部结构）二值掩膜
    frame : VertebraFrame
        椎体坐标系（estimate_vertebra_frame 的结果）
    side : str
        "left" 或 "right"（患者左右）
    """
    if config is None:
        config = BaselineConfig()
    config.validate()
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    side_sign = 1.0 if side == "left" else -1.0

    a_body, a_lamina = find_canal(mask, frame, config)
    a_values = np.arange(a_body - config.CORONAL_STEP, a_lamina + config.LAMINA_MARGIN_MM + 1e-9,
                         -config.CORONAL_STEP)
    centers = []
    for a in a_values:
        found = _pedicle_center_in_plane(mask, frame, a, side_sign, config)
        if found is not None and found[2] > 0:
            centers.append((a, *found))
    if len(centers) < 2:
        raise ValueError(f"could not locate the {side} pedicle between the body and the lamina")
    centers = np.asarray(centers)  # columns: a, l, s, radius
    isthmus = centers[int(np.argmin(centers[:, 3]))]

    center = _local_points(frame, isthmus[1], isthmus[0], isthmus[2])
    medial_angle, wall_clearance = safest_medial_angle(mask, frame, center, side_sign, config)
    rad = np.radians(medial_angle)
    direction = np.cos(rad) * np.asarray(frame.anterior) - side_sign * np.sin(rad) * np.asarray(frame.left)

    # entry: walk back from the isthmus centre to the posterior bone surface
    run = max(1, int(round(config.EXIT_RUN_MM / config.GRID_STEP)))
    radii, inside = _walk(mask, center, -direction, config.MAX_WALK_MM, config.GRID_STEP)
    back_exit = _first_run(~inside, run)
    if back_exit < 0:
        raise ValueError(f"{side} trajectory does not leave bone posteriorly within {config.MAX_WALK_MM} mm")
    entry = center - radii[back_exit] * direction

    # anterior cortex: walk forward from the isthmus centre to where the bone ends
    radii, inside = _walk(mask, center, direction, config.MAX_WALK_MM, config.GRID_STEP)
    exit_idx = _first_run(~inside, run)
    if exit_idx < 0:
        raise ValueError(f"{side} trajectory does not reach the anterior cortex within {config.MAX_WALK_MM} mm")
    in_bone = float(radii[back_exit] + radii[exit_idx])
    length = _floor_to(config.LENGTHS_MM, min(config.LENGTH_FRACTION * in_bone, in_bone - config.TIP_MARGIN_MM))
    width = 2.0 * float(isthmus[3])
    diameter = _floor_to(config.DIAMETERS_MM, config.DIAMETER_FRACTION * width)
    if not (np.isfinite(length) and np.isfinite(diameter)):
        raise ValueError(f"{side} pedicle too small for the screw catalogue "
                         f"(in-bone length {in_bone:.1f} mm, isthmus width {width:.1f} mm)")
    return BaselineScrew(
        side=side,
        plan=ScrewPlan(tuple(entry), tuple(direction), length, diameter),
        isthmus_center=center,
        isthmus_width_mm=width,
        medial_angle_deg=medial_angle,
        wall_clearance_mm=wall_clearance,
        in_bone_length_mm=in_bone,
        pedicle_centers=centers,
    )


def generate_baseline_screws(mask: sitk.Image, frame, sides=("left", "right"),
                             config: BaselineConfig = None) -> dict:
    """双侧（或指定侧）基准方案；单侧失败时该侧值为错误说明字符串。"""
    out = {}
    for side in sides:
        try:
            out[side] = generate_baseline_screw(mask, frame, side, config)
        except ValueError as exc:
            out[side] = str(exc)
    return out
