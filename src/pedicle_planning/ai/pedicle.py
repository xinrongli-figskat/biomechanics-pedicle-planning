"""
pedicle.py
------------------------------------
椎弓根峡部检测与钉道匹配度

以一枚位于椎弓根内的基准螺钉（通常为医生经验方案）为引导，沿钉轴做一系列
垂直切片，在每个切片上提取包含钉轴的骨截面连通域，取截面积最小处为椎弓根
峡部；再由峡部附近截面的质心拟合椎弓根轴线，并沿该轴线重新切片以减小斜切
带来的面积高估。

输出的 PedicleGeometry 以解剖位置（垂直于椎弓根轴线的平面）描述峡部与
椎弓根段，可用于任意候选螺钉：计算钉道分段边界、轨迹匹配度与峡部处的
壁距，从而避免对每个候选方案重复检测。

输入要求：二值骨（椎体）掩膜，基准钉轴需穿过椎弓根。CT图像可选，用于以
皮质阈值区分松质骨通道，得到椎弓根“内径”。
"""

from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk

from .trajectory import (
    ScrewPlan,
    _orthonormal_frame,
    sample_image_trilinear,
    sample_mask_nearest,
)


@dataclass
class IsthmusDetectionConfig:
    """椎弓根峡部检测配置参数（长度单位均为毫米）。"""

    AXIAL_STEP: float = 1.0
    """切片间距。"""

    CROSS_STEP: float = 0.25
    """切片内采样网格间距。"""

    WINDOW_RADIUS: float = 12.0
    """切片采样窗口半宽；截面连通域触及窗口边界时视为已进入椎体/椎板，不参与峡部搜索。"""

    SEARCH_RANGE: tuple = (0.05, 0.7)
    """在基准钉轴上搜索峡部的范围（相对螺钉长度的比例）。"""

    CENTER_SNAP: float = 2.0
    """钉轴点不在骨内时，向最近骨像素吸附的最大距离。"""

    CORTICAL_HU: float = 800.0
    """皮质阈值；截面内低于该值且与中心连通的区域视为松质骨通道（内径）。"""

    PEDICLE_AREA_FACTOR: float = 2.0
    """截面积不超过峡部面积该倍数的连续区间视为椎弓根段。"""

    AXIS_FIT_AREA_FACTOR: float = 1.5
    """用于拟合椎弓根轴线的截面面积上限（峡部面积的倍数）。"""

    ANGLE_STEP_DEG: float = 5.0
    """计算最小宽度时的方向步长。"""

    REFINE: bool = True
    """是否沿拟合的椎弓根轴线重新切片。"""

    MIN_INNER_DOMINANCE: float = 0.7
    """
    提供CT时，截面内含钉轴的松质骨连通块占全部松质骨的最小比例。低于该值说明
    切面横跨了皮质壁（如椎弓根-椎体交界处的椎体后壁），内径无意义，该截面不参与
    峡部选择；若所有截面都不满足，则退回仅按窗口条件选择。
    """

    def validate(self) -> None:
        for name in ("AXIAL_STEP", "CROSS_STEP", "WINDOW_RADIUS", "ANGLE_STEP_DEG"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} ({getattr(self, name)}) must be positive")
        if self.CENTER_SNAP < 0:
            raise ValueError(f"CENTER_SNAP ({self.CENTER_SNAP}) must be non-negative")
        lo, hi = self.SEARCH_RANGE
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"SEARCH_RANGE ({self.SEARCH_RANGE}) must satisfy 0 <= lo < hi <= 1")
        if not 0.0 <= self.MIN_INNER_DOMINANCE <= 1.0:
            raise ValueError(f"MIN_INNER_DOMINANCE ({self.MIN_INNER_DOMINANCE}) must be in [0, 1]")
        for name in ("PEDICLE_AREA_FACTOR", "AXIS_FIT_AREA_FACTOR"):
            if not getattr(self, name) >= 1.0:
                raise ValueError(f"{name} ({getattr(self, name)}) must be >= 1")


@dataclass
class CrossSection:
    """
    单个垂直切片上的骨截面

    grid_u/grid_v 为切片平面内各像素的二维坐标（毫米，相对切片中心，沿 u/v 轴），
    outer 为包含钉轴的骨连通域，inner 为其中的松质骨通道（无CT时为 None）；
    inner_dominance 为 inner 占截面内全部松质骨的比例（无CT时为 NaN）；
    contains_center 表示切片中心（钉轴点）本身位于 outer 内（未经吸附）。
    """

    t: float
    center: np.ndarray
    normal: np.ndarray
    u: np.ndarray
    v: np.ndarray
    step: float
    grid_u: np.ndarray
    grid_v: np.ndarray
    outer: np.ndarray
    inner: np.ndarray
    touches_window: bool
    contains_center: bool
    area_outer: float
    area_inner: float
    inner_dominance: float
    centroid: np.ndarray
    min_width_outer: float
    min_width_inner: float
    min_width_direction: np.ndarray

    def to_plane_coords(self, point) -> np.ndarray:
        d = np.asarray(point, dtype=np.float64) - self.center
        return np.array([np.dot(d, self.u), np.dot(d, self.v)])

    def clearance(self, point, region: str = "outer") -> float:
        """平面内点到所选区域边界（区域外最近像素）的距离；点不在区域内时为负值。"""
        mask = self.outer if region == "outer" else self.inner
        if mask is None:
            raise ValueError(f"cross-section has no {region} region")
        pu, pv = self.to_plane_coords(point)
        dist = np.hypot(self.grid_u - pu, self.grid_v - pv)
        outside = ~mask
        nearest_outside = float(dist[outside].min()) if np.any(outside) else float("inf")
        if not np.any(mask):
            return -float("inf")
        nearest_inside = float(dist[mask].min())
        if nearest_inside <= self.step * 0.5 + 1e-9:
            return nearest_outside - 0.5 * self.step
        return -nearest_inside


@dataclass
class PedicleGeometry:
    """
    椎弓根几何

    isthmus 为峡部截面，axis 为拟合的椎弓根轴线方向（与基准螺钉同向），
    section_normal 为椎弓根段起止平面的法向（第一次切片所用的基准钉轴方向），
    start_point/end_point 为椎弓根段起止平面（法向 section_normal）上的点。
    profile 记录每个切片的 (沿切片线的位置, 外截面积, 是否触及窗口)。
    """

    isthmus: CrossSection
    axis: np.ndarray
    section_normal: np.ndarray
    start_point: np.ndarray
    end_point: np.ndarray
    profile: list = field(default_factory=list)

    @property
    def isthmus_point(self) -> np.ndarray:
        return self.isthmus.centroid

    @property
    def outer_min_width(self) -> float:
        return self.isthmus.min_width_outer

    @property
    def inner_min_width(self) -> float:
        """松质骨通道最小宽度；未提供CT时退化为外径最小宽度。"""
        if np.isfinite(self.isthmus.min_width_inner):
            return self.isthmus.min_width_inner
        return self.isthmus.min_width_outer

    def axis_plane_t(self, plan: ScrewPlan, plane_point, plane_normal=None) -> float:
        """螺钉轴线与平面交点的轴向位置（距进钉点，毫米）；轴线与平面平行时为 NaN。"""
        normal = self.section_normal if plane_normal is None else np.asarray(plane_normal, dtype=np.float64)
        direction = np.asarray(plan.direction)
        denom = float(np.dot(direction, normal))
        if abs(denom) < 1e-9:
            return float("nan")
        return float(np.dot(np.asarray(plane_point) - np.asarray(plan.entry), normal) / denom)

    def isthmus_crossing(self, plan: ScrewPlan) -> np.ndarray:
        """螺钉轴线穿过峡部截面平面的点。"""
        t = self.axis_plane_t(plan, self.isthmus.center, self.isthmus.normal)
        return np.asarray(plan.entry) + t * np.asarray(plan.direction)

    def segment_bounds_for(self, plan: ScrewPlan, transition_mm: float = 5.0,
                           tip_mm: float = 8.0) -> tuple:
        """
        按解剖位置给出该螺钉的分段边界（毫米），可直接传给
        compute_trajectory_features 的 segment_bounds_mm。

        entry: 进钉点到椎弓根起始平面；pedicle: 椎弓根段；transition: 其后
        transition_mm；body: 至钉尖前 tip_mm；tip: 最后 tip_mm。
        """
        length = plan.length
        t_start = self.axis_plane_t(plan, self.start_point)
        t_end = self.axis_plane_t(plan, self.end_point)
        if not (np.isfinite(t_start) and np.isfinite(t_end)):
            raise ValueError("screw axis is parallel to the pedicle planes")
        t_start, t_end = sorted((t_start, t_end))
        raw = [0.0, t_start, t_end, t_end + transition_mm, length - tip_mm, length]
        bounds = np.clip(np.maximum.accumulate(np.clip(raw, 0.0, length)), 0.0, length)
        bounds[-1] = length
        return tuple(float(b) for b in bounds)

    def trajectory_match(self, plan: ScrewPlan) -> dict:
        """
        轨迹匹配度特征

        angle_deg: 螺钉轴与椎弓根轴夹角；isthmus_offset_mm: 峡部平面内螺钉轴
        与峡部质心的距离；outer/inner_clearance_mm: 峡部处螺钉表面到外轮廓/
        松质骨通道边界的最小距离（负值表示越出）；diameter_ratio: 螺钉直径与
        峡部内径之比。
        """
        direction = np.asarray(plan.direction)
        cos_angle = float(np.clip(abs(np.dot(direction, self.axis)), -1.0, 1.0))
        crossing = self.isthmus_crossing(plan)
        radius = plan.diameter / 2.0
        outer_clearance = self.isthmus.clearance(crossing, "outer") - radius
        if self.isthmus.inner is not None:
            inner_clearance = self.isthmus.clearance(crossing, "inner") - radius
        else:
            inner_clearance = float("nan")
        return {
            "angle_deg": float(np.degrees(np.arccos(cos_angle))),
            "isthmus_offset_mm": float(np.linalg.norm(crossing - self.isthmus.centroid)),
            "outer_clearance_mm": float(outer_clearance),
            "inner_clearance_mm": float(inner_clearance),
            "diameter_ratio": float(plan.diameter / self.inner_min_width),
            "isthmus_t_mm": self.axis_plane_t(plan, self.isthmus.center, self.isthmus.normal),
        }


def _component_at(mask2d: np.ndarray, grid_u: np.ndarray, grid_v: np.ndarray,
                  pu: float, pv: float, snap: float) -> np.ndarray:
    """返回包含平面点 (pu, pv) 的连通域；点不在区域内时吸附到 snap 距离内最近的像素。"""
    if not np.any(mask2d):
        return None
    labels = sitk.GetArrayFromImage(
        sitk.ConnectedComponent(sitk.GetImageFromArray(mask2d.astype(np.uint8)))
    )
    dist = np.hypot(grid_u - pu, grid_v - pv)
    candidates = np.where(mask2d, dist, np.inf)
    flat = int(np.argmin(candidates))
    if not np.isfinite(candidates.flat[flat]) or candidates.flat[flat] > snap + 1e-9:
        return None
    return labels == labels.flat[flat]


def _min_width(grid_u, grid_v, region, step, angle_step_deg) -> tuple:
    if region is None or not np.any(region):
        return float("nan"), np.array([np.nan, np.nan])
    pu, pv = grid_u[region], grid_v[region]
    best_width, best_dir = np.inf, (1.0, 0.0)
    for theta in np.radians(np.arange(0.0, 180.0, angle_step_deg)):
        c, s = np.cos(theta), np.sin(theta)
        proj = pu * c + pv * s
        width = float(proj.max() - proj.min() + step)
        if width < best_width:
            best_width, best_dir = width, (c, s)
    return best_width, np.asarray(best_dir)


def extract_cross_section(bone_mask: sitk.Image, center, normal, t: float = 0.0,
                          image: sitk.Image = None,
                          config: IsthmusDetectionConfig = None) -> CrossSection:
    """
    在过 center、法向为 normal 的平面上提取包含 center 的骨截面

    Returns
    -------
    CrossSection or None
        center 附近（CENTER_SNAP 内）没有骨时返回 None
    """
    if config is None:
        config = IsthmusDetectionConfig()
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    center = np.asarray(center, dtype=np.float64)
    u, v = _orthonormal_frame(normal)

    half = int(np.ceil(config.WINDOW_RADIUS / config.CROSS_STEP))
    coords = np.arange(-half, half + 1) * config.CROSS_STEP
    grid_u, grid_v = np.meshgrid(coords, coords, indexing="ij")
    points = center + np.outer(grid_u.ravel(), u) + np.outer(grid_v.ravel(), v)

    in_bone = sample_mask_nearest(bone_mask, points).reshape(grid_u.shape)
    outer = _component_at(in_bone, grid_u, grid_v, 0.0, 0.0, config.CENTER_SNAP)
    if outer is None:
        return None

    inner = None
    inner_dominance = float("nan")
    if image is not None:
        hu = sample_image_trilinear(image, points).reshape(grid_u.shape)
        cancellous = outer & np.isfinite(hu) & (hu < config.CORTICAL_HU)
        inner = _component_at(cancellous, grid_u, grid_v, 0.0, 0.0, config.CENTER_SNAP)
        if inner is None:
            inner = np.zeros_like(outer)
        total = np.count_nonzero(cancellous)
        inner_dominance = float(np.count_nonzero(inner) / total) if total else 0.0

    border = np.zeros_like(outer)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    pixel_area = config.CROSS_STEP ** 2

    cu, cv = float(grid_u[outer].mean()), float(grid_v[outer].mean())
    width_outer, dir_outer = _min_width(grid_u, grid_v, outer, config.CROSS_STEP, config.ANGLE_STEP_DEG)
    width_inner, _ = _min_width(grid_u, grid_v, inner, config.CROSS_STEP, config.ANGLE_STEP_DEG)

    return CrossSection(
        t=float(t),
        center=center,
        normal=normal,
        u=np.asarray(u),
        v=np.asarray(v),
        step=config.CROSS_STEP,
        grid_u=grid_u,
        grid_v=grid_v,
        outer=outer,
        inner=inner,
        touches_window=bool(np.any(outer & border)),
        contains_center=bool(outer[half, half]),
        area_outer=float(np.count_nonzero(outer) * pixel_area),
        area_inner=float(np.count_nonzero(inner) * pixel_area) if inner is not None else float("nan"),
        inner_dominance=inner_dominance,
        centroid=center + cu * np.asarray(u) + cv * np.asarray(v),
        min_width_outer=width_outer,
        min_width_inner=width_inner,
        min_width_direction=dir_outer[0] * np.asarray(u) + dir_outer[1] * np.asarray(v),
    )


def _profile(bone_mask, image, origin, direction, t_values, config) -> list:
    return [extract_cross_section(bone_mask, origin + t * direction, direction, t, image, config)
            for t in t_values]


def _enclosed(section) -> bool:
    return section is not None and not section.touches_window and section.contains_center


def _single_channel(section, min_inner_dominance: float) -> bool:
    return (not np.isfinite(section.inner_dominance)) or section.inner_dominance >= min_inner_dominance


def _eligible_sections(sections, min_inner_dominance: float) -> list:
    """可参与峡部选择的截面：封闭、钉轴点在骨内，且（有CT时）松质骨以单一通道为主；
    若没有截面满足通道条件，退回仅按封闭条件。"""
    enclosed = [_enclosed(s) for s in sections]
    eligible = [e and _single_channel(s, min_inner_dominance) for e, s in zip(enclosed, sections)]
    return eligible if any(eligible) else enclosed


def _isthmus_index(sections, min_inner_dominance: float = 0.0) -> int:
    eligible = _eligible_sections(sections, min_inner_dominance)
    valid = [i for i, ok in enumerate(eligible) if ok]
    if not valid:
        return None
    return min(valid, key=lambda i: sections[i].area_outer)


def _contiguous_range(sections, center_idx, max_area, min_inner_dominance: float = 0.0) -> tuple:
    eligible = _eligible_sections(sections, min_inner_dominance)

    def ok(i):
        return eligible[i] and sections[i].area_outer <= max_area

    lo = hi = center_idx
    while lo - 1 >= 0 and ok(lo - 1):
        lo -= 1
    while hi + 1 < len(sections) and ok(hi + 1):
        hi += 1
    return lo, hi


def _fit_axis(sections, lo, hi, reference) -> np.ndarray:
    centroids = np.array([sections[i].centroid for i in range(lo, hi + 1)])
    if centroids.shape[0] < 3:
        return np.asarray(reference, dtype=np.float64)
    centered = centroids - centroids.mean(axis=0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    if np.dot(axis, reference) < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def detect_pedicle_isthmus(bone_mask: sitk.Image, plan: ScrewPlan, image: sitk.Image = None,
                           config: IsthmusDetectionConfig = None) -> PedicleGeometry:
    """
    沿基准螺钉检测椎弓根峡部

    Parameters
    ----------
    bone_mask : sitk.Image
        椎体（含椎弓根）二值掩膜
    plan : ScrewPlan
        穿过椎弓根的基准螺钉方案
    image : sitk.Image, optional
        CT图像（HU），用于提取松质骨通道（内径）
    config : IsthmusDetectionConfig, optional

    Raises
    ------
    ValueError
        搜索范围内找不到未触及窗口边界的骨截面（钉轴不在椎弓根内或窗口过小）
    """
    if config is None:
        config = IsthmusDetectionConfig()
    config.validate()

    direction = np.asarray(plan.direction)
    entry = np.asarray(plan.entry)
    lo_t, hi_t = (f * plan.length for f in config.SEARCH_RANGE)
    t_values = np.arange(lo_t, hi_t + 1e-9, config.AXIAL_STEP)

    sections = _profile(bone_mask, image, entry, direction, t_values, config)
    idx = _isthmus_index(sections, config.MIN_INNER_DOMINANCE)
    if idx is None:
        raise ValueError("no enclosed bone cross-section found along the screw axis; "
                         "check that the baseline screw passes through the pedicle")

    fit_lo, fit_hi = _contiguous_range(sections, idx, config.AXIS_FIT_AREA_FACTOR * sections[idx].area_outer,
                                       config.MIN_INNER_DOMINANCE)
    axis = _fit_axis(sections, fit_lo, fit_hi, direction)

    # The pedicle segment is delimited on the first-pass profile (planes normal to the baseline axis).
    ped_lo, ped_hi = _contiguous_range(sections, idx, config.PEDICLE_AREA_FACTOR * sections[idx].area_outer,
                                       config.MIN_INNER_DOMINANCE)
    half_step = 0.5 * config.AXIAL_STEP
    start_point = entry + (t_values[ped_lo] - half_step) * direction
    end_point = entry + (t_values[ped_hi] + half_step) * direction
    profile = [
        (float(t), float(s.area_outer) if s is not None else float("nan"),
         bool(s.touches_window) if s is not None else False)
        for t, s in zip(t_values, sections)
    ]
    isthmus = sections[idx]

    # Refinement only corrects the obliquity of the isthmus section: re-slice perpendicular to the
    # fitted axis close to the first-pass isthmus instead of searching the whole range again.
    if config.REFINE and np.degrees(np.arccos(np.clip(np.dot(axis, direction), -1.0, 1.0))) > 0.5:
        origin = isthmus.centroid
        reach = max(t_values[idx] - t_values[fit_lo], t_values[fit_hi] - t_values[idx]) + config.AXIAL_STEP
        refined_t = np.arange(-reach, reach + 1e-9, 0.5 * config.AXIAL_STEP)
        refined = _profile(bone_mask, image, origin, axis, refined_t, config)
        refined_idx = _isthmus_index(refined, config.MIN_INNER_DOMINANCE)
        # a much smaller refined section means the oblique re-slice slipped off the pedicle
        if refined_idx is not None and refined[refined_idx].area_outer >= 0.6 * isthmus.area_outer:
            isthmus = refined[refined_idx]
    return PedicleGeometry(
        isthmus=isthmus,
        axis=np.asarray(axis),
        section_normal=np.asarray(direction),
        start_point=start_point,
        end_point=end_point,
        profile=profile,
    )
