"""
pedicle.py
------------------------------------
椎弓根峡部检测与钉道匹配度（射线测量法）

以一枚穿过椎弓根的基准螺钉（通常为医生经验方案）为引导，沿钉轴每隔
AXIAL_STEP 取一个垂直截面。在截面内从钉轴点向各方向（0° 为内侧、90° 为头侧）
发出射线，射线连续离开骨掩膜处即为外壁；有CT时，沿射线在外壁附近寻找皮质，
内壁取 HU 越过“松质骨水平与皮质峰值的中点”处（每个截面自适应，而非固定阈值）。

* 横径（内外宽度）= 内侧 + 外侧射线长度，高度 = 头侧 + 尾侧射线长度；
* 峡部 = 内外两侧射线都在 MAX_RAY_MM 内闭合、横径最小的截面。与横突、上关节突
  相连只会让相应方向的射线变长，不会使整个截面失效；
* 椎弓根段 = 峡部前后横径不超过 PEDICLE_WIDTH_FACTOR 倍的连续区间；
* 椎弓根轴 = 椎弓根段内各截面“内外中点、头尾中点”的拟合直线；峡部再沿该轴
  垂直重测一次，以消除斜切造成的宽度高估。

输出的 PedicleGeometry 以解剖位置描述峡部与椎弓根段，可对任意候选螺钉计算
分段边界、轨迹匹配度与峡部处到内/外侧壁的距离。

输入要求：二值骨（椎体）掩膜；建议提供解剖方向 frame（内侧、头侧），否则
内外/头尾方向取截面内任意正交方向，横径与高度失去解剖含义。
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
    """截面间距。"""

    SEARCH_RANGE: tuple = (0.05, 0.7)
    """在基准钉轴上搜索峡部的范围（相对螺钉长度的比例）。"""

    RAY_ANGLE_STEP_DEG: float = 15.0
    """截面内射线的角度间隔；须整除 90，以包含内侧、头侧、外侧、尾侧四个方向。"""

    RAY_STEP_MM: float = 0.25
    """沿射线的采样间距。"""

    MAX_RAY_MM: float = 15.0
    """射线最大长度；超过仍在骨内视为该方向未闭合（连到横突、椎板或椎体）。"""

    EXIT_RUN_MM: float = 0.75
    """连续离开骨掩膜达到该长度才算到达外壁，避免分割小孔造成的误判。"""

    CORTEX_SEARCH_MM: float = 4.0
    """在外壁以内该范围内寻找皮质峰值与内壁。"""

    MIN_CORTEX_CONTRAST_HU: float = 100.0
    """皮质峰值与松质骨水平之差的下限；低于该值视为皮质未分辨，内壁取外壁。"""

    PEDICLE_WIDTH_FACTOR: float = 1.3
    """横径不超过峡部横径该倍数的连续区间视为椎弓根段。"""

    REFINE: bool = True
    """是否在峡部沿拟合的椎弓根轴垂直重测。"""

    MIN_AXIS_SECTIONS: int = 4
    """拟合椎弓根轴所需的最少截面数；不足时沿用钉轴方向。"""

    AXIS_CORE_WIDTH_FACTOR: float = 1.2
    """
    只用横径不超过峡部横径该倍数的截面拟合椎弓根轴（峡部核心段）；不足
    MIN_AXIS_SECTIONS 时放宽到整个椎弓根段。椎弓根两端呈漏斗形张开，
    张开段的截面中点会把轴线拉偏。
    """

    MAX_AXIS_ANGLE_DEG: float = 20.0
    """
    椎弓根轴与钉轴夹角的上限；回归结果超过该值视为不可靠，沿用钉轴方向
    （angle_deg 报 NaN）。经验规划的螺钉与椎弓根轴相差超过 20° 少见，而首例
    临床数据中漏斗形张开造成的假倾角为 28–34°。
    """

    def validate(self) -> None:
        for name in ("AXIAL_STEP", "RAY_ANGLE_STEP_DEG", "RAY_STEP_MM", "MAX_RAY_MM", "EXIT_RUN_MM",
                     "CORTEX_SEARCH_MM"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} ({getattr(self, name)}) must be positive")
        if abs(90.0 / self.RAY_ANGLE_STEP_DEG - round(90.0 / self.RAY_ANGLE_STEP_DEG)) > 1e-9:
            raise ValueError("RAY_ANGLE_STEP_DEG must divide 90")
        lo, hi = self.SEARCH_RANGE
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"SEARCH_RANGE ({self.SEARCH_RANGE}) must satisfy 0 <= lo < hi <= 1")
        if not 1.0 <= self.AXIS_CORE_WIDTH_FACTOR <= self.PEDICLE_WIDTH_FACTOR:
            raise ValueError("AXIS_CORE_WIDTH_FACTOR must be within [1, PEDICLE_WIDTH_FACTOR]")
        if not self.PEDICLE_WIDTH_FACTOR >= 1.0:
            raise ValueError(f"PEDICLE_WIDTH_FACTOR ({self.PEDICLE_WIDTH_FACTOR}) must be >= 1")
        if self.MIN_CORTEX_CONTRAST_HU < 0:
            raise ValueError("MIN_CORTEX_CONTRAST_HU must be non-negative")


@dataclass
class RaySection:
    """
    单个截面的射线测量

    angles_deg 从内侧（0°）经头侧（90°）、外侧（180°）到尾侧（270°）；outer/inner
    为各方向到外壁/内壁的距离（未闭合为 inf）。宽度取对向两侧各自三条相邻射线
    距离的中位数之和。
    """

    t: float
    center: np.ndarray
    normal: np.ndarray
    medial: np.ndarray
    cranial: np.ndarray
    angles_deg: np.ndarray
    outer: np.ndarray
    inner: np.ndarray
    center_in_bone: bool
    cancellous_hu: float
    cortex_contrast_hu: float
    cortex_threshold_hu: float
    cortex_resolved: bool

    def _side(self, values: np.ndarray, angle: float) -> float:
        step = float(self.angles_deg[1] - self.angles_deg[0])
        idx = int(round(angle / step)) % len(values)
        neighbours = values[[(idx - 1) % len(values), idx, (idx + 1) % len(values)]]
        return float(np.median(neighbours))

    def distance(self, angle: float, region: str = "outer") -> float:
        return self._side(self.outer if region == "outer" else self.inner, angle)

    def width(self, region: str = "outer") -> float:
        """横径（内侧 + 外侧）。"""
        return self.distance(0.0, region) + self.distance(180.0, region)

    def height(self, region: str = "outer") -> float:
        """高度（头侧 + 尾侧）。"""
        return self.distance(90.0, region) + self.distance(270.0, region)

    @property
    def ml_bounded(self) -> bool:
        return self.center_in_bone and np.isfinite(self.width("outer"))

    @property
    def chord_center(self) -> np.ndarray:
        """内外中点与头尾中点（未闭合方向不修正）。"""
        point = np.asarray(self.center, dtype=np.float64).copy()
        dm, dl = self.distance(0.0), self.distance(180.0)
        if np.isfinite(dm) and np.isfinite(dl):
            point += 0.5 * (dm - dl) * self.medial
        dc, dd = self.distance(90.0), self.distance(270.0)
        if np.isfinite(dc) and np.isfinite(dd):
            point += 0.5 * (dc - dd) * self.cranial
        return point

    def area(self, region: str = "outer") -> float:
        """射线多边形面积（含未闭合方向时为 NaN）。"""
        d = self.outer if region == "outer" else self.inner
        if not np.all(np.isfinite(d)):
            return float("nan")
        step = np.radians(float(self.angles_deg[1] - self.angles_deg[0]))
        return float(0.5 * np.sum(d * np.roll(d, -1)) * np.sin(step))


def _in_plane_axes(normal: np.ndarray, frame) -> tuple:
    """截面内的内侧、头侧方向（垂直于 normal）；无 frame 时取任意正交方向。"""
    if frame is not None:
        medial = np.asarray(frame.medial, dtype=np.float64)
        medial = medial - np.dot(medial, normal) * normal
        if np.linalg.norm(medial) > 1e-6:
            medial /= np.linalg.norm(medial)
            cranial = np.cross(normal, medial)
            if np.dot(cranial, np.asarray(frame.cranial, dtype=np.float64)) < 0:
                cranial = -cranial
            return medial, cranial
    u, v = _orthonormal_frame(normal)
    return np.asarray(u), np.asarray(v)


def _exit_distance(inside_row: np.ndarray, radii: np.ndarray, run: int) -> float:
    """第一个连续 run 个骨外样本的起点；没有则 inf。"""
    outside = ~inside_row
    if run <= 1:
        hits = np.flatnonzero(outside)
        return float(radii[hits[0]]) if hits.size else float("inf")
    window = np.convolve(outside.astype(int), np.ones(run, dtype=int), mode="valid")
    hits = np.flatnonzero(window == run)
    return float(radii[hits[0]]) if hits.size else float("inf")


def measure_section(bone_mask: sitk.Image, center, normal, t: float = 0.0, image: sitk.Image = None,
                    frame=None, config: IsthmusDetectionConfig = None) -> RaySection:
    """在过 center、法向为 normal 的截面上做射线测量。"""
    if config is None:
        config = IsthmusDetectionConfig()
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    center = np.asarray(center, dtype=np.float64)
    medial, cranial = _in_plane_axes(normal, frame)

    angles = np.arange(0.0, 360.0, config.RAY_ANGLE_STEP_DEG)
    rad = np.radians(angles)
    directions = np.outer(np.cos(rad), medial) + np.outer(np.sin(rad), cranial)
    radii = np.arange(0.0, config.MAX_RAY_MM + 1e-9, config.RAY_STEP_MM)
    points = center + directions[:, None, :] * radii[None, :, None]
    flat = points.reshape(-1, 3)
    inside = sample_mask_nearest(bone_mask, flat).reshape(len(angles), len(radii))
    run = max(1, int(round(config.EXIT_RUN_MM / config.RAY_STEP_MM)))
    outer = np.array([_exit_distance(inside[i], radii, run) for i in range(len(angles))])
    center_in_bone = bool(inside[:, 0].all())

    inner = outer.copy()
    cancellous = contrast = threshold = float("nan")
    resolved = False
    if image is not None and center_in_bone:
        hu = sample_image_trilinear(image, flat).reshape(len(angles), len(radii))
        bounded = np.isfinite(outer)
        core = [hu[i, radii <= 0.5 * outer[i]] for i in range(len(angles)) if bounded[i]]
        core = np.concatenate(core) if core else np.zeros(0)
        core = core[np.isfinite(core)]
        if core.size:
            cancellous = float(np.median(core))
            peaks = []
            for i in np.flatnonzero(bounded):
                window = (radii >= outer[i] - config.CORTEX_SEARCH_MM) & (radii <= outer[i])
                values = hu[i, window]
                if np.any(np.isfinite(values)):
                    peaks.append(float(np.nanmax(values)))
            if peaks:
                contrast = float(np.median(peaks)) - cancellous
                if contrast >= config.MIN_CORTEX_CONTRAST_HU:
                    resolved = True
                    threshold = cancellous + 0.5 * contrast
                    for i in np.flatnonzero(bounded):
                        # the inner wall lies in the outer half of the ray: a narrow pedicle must not
                        # let the search window reach the axis point
                        inner_start = max(outer[i] - config.CORTEX_SEARCH_MM, 0.5 * outer[i])
                        window = np.flatnonzero((radii >= inner_start) & (radii <= outer[i]))
                        above = window[hu[i, window] >= threshold]
                        inner[i] = float(radii[above[0]]) if above.size else outer[i]

    return RaySection(
        t=float(t), center=center, normal=normal, medial=medial, cranial=cranial, angles_deg=angles,
        outer=outer, inner=inner, center_in_bone=center_in_bone, cancellous_hu=cancellous,
        cortex_contrast_hu=contrast, cortex_threshold_hu=threshold, cortex_resolved=resolved,
    )


@dataclass
class PedicleGeometry:
    """
    椎弓根几何

    isthmus 为峡部射线截面；axis 为拟合的椎弓根轴线方向（与基准螺钉同向），
    axis_status / axis_fit_tilt_deg 说明拟合是否被采用（见 _fit_axis）；未采用时
    axis 为钉轴方向，trajectory_match 的 angle_deg 为 NaN（而不是 0）；
    section_normal 为椎弓根段起止平面的法向（基准钉轴方向），start_point/end_point
    为起止平面上的点；profile 为各截面的诊断字典（t、横径、高度、四个方向的
    外壁距离、是否属于椎弓根段，未闭合为 inf）。
    """

    isthmus: RaySection
    axis: np.ndarray
    section_normal: np.ndarray
    start_point: np.ndarray
    end_point: np.ndarray
    bone_mask: object = None
    image: object = None
    frame: object = None
    config: IsthmusDetectionConfig = None
    profile: list = field(default_factory=list)
    axis_status: str = "fitted"
    axis_fit_tilt_deg: float = float("nan")

    @property
    def isthmus_point(self) -> np.ndarray:
        return self.isthmus.chord_center

    @property
    def outer_width(self) -> float:
        return self.isthmus.width("outer")

    @property
    def outer_height(self) -> float:
        return self.isthmus.height("outer")

    @property
    def inner_width(self) -> float:
        return self.isthmus.width("inner")

    @property
    def inner_height(self) -> float:
        return self.isthmus.height("inner")

    @property
    def outer_min_width(self) -> float:
        return float(np.nanmin([self.outer_width, self.outer_height]))

    @property
    def inner_min_width(self) -> float:
        """内径（横径与高度中较小者）；皮质未分辨时等于外径。"""
        return float(np.nanmin([self.inner_width, self.inner_height]))

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

    def _clearances(self, plan: ScrewPlan) -> dict:
        crossing = self.isthmus_crossing(plan)
        radius = plan.diameter / 2.0
        section = measure_section(self.bone_mask, crossing, self.isthmus.normal, image=self.image,
                                  frame=self.frame, config=self.config)
        if not section.center_in_bone:
            return {key: -radius - _distance_to_bone(self.bone_mask, crossing, self.isthmus.normal, self.frame,
                                                     self.config)
                    for key in ("outer", "inner", "medial", "lateral")}
        return {
            "outer": float(np.min(section.outer)) - radius,
            "inner": float(np.min(section.inner)) - radius if section.cortex_resolved else float("nan"),
            "medial": section.distance(0.0) - radius,
            "lateral": section.distance(180.0) - radius,
        }

    def trajectory_match(self, plan: ScrewPlan) -> dict:
        """
        轨迹匹配度特征

        angle_deg: 螺钉轴与椎弓根轴夹角（椎弓根轴未拟合成功时为 NaN）；isthmus_offset_mm: 峡部平面内螺钉轴与峡部
        中心的距离；outer/inner_clearance_mm: 峡部处螺钉表面到外壁/内壁（皮质内缘）
        的最小距离（负值表示越出）；medial/lateral_clearance_mm: 到内侧壁/外侧壁的
        距离；diameter_ratio: 螺钉直径与峡部内径之比。
        """
        direction = np.asarray(plan.direction)
        cos_angle = float(np.clip(abs(np.dot(direction, self.axis)), -1.0, 1.0))
        crossing = self.isthmus_crossing(plan)
        clear = self._clearances(plan)
        return {
            "angle_deg": float(np.degrees(np.arccos(cos_angle))) if self.axis_status == "fitted" else float("nan"),
            "isthmus_offset_mm": float(np.linalg.norm(crossing - self.isthmus_point)),
            "outer_clearance_mm": clear["outer"],
            "inner_clearance_mm": clear["inner"],
            "medial_clearance_mm": clear["medial"],
            "lateral_clearance_mm": clear["lateral"],
            "diameter_ratio": (float(plan.diameter / self.inner_min_width) if self.inner_min_width > 0
                               else float("nan")),
            "isthmus_t_mm": self.axis_plane_t(plan, self.isthmus.center, self.isthmus.normal),
        }


def _distance_to_bone(bone_mask, point, normal, frame, config) -> float:
    """截面内从骨外的点到最近骨像素的射线距离（找不到为 MAX_RAY_MM）。"""
    config = config or IsthmusDetectionConfig()
    normal = np.asarray(normal, dtype=np.float64)
    medial, cranial = _in_plane_axes(normal / np.linalg.norm(normal), frame)
    angles = np.radians(np.arange(0.0, 360.0, config.RAY_ANGLE_STEP_DEG))
    radii = np.arange(0.0, config.MAX_RAY_MM + 1e-9, config.RAY_STEP_MM)
    directions = np.outer(np.cos(angles), medial) + np.outer(np.sin(angles), cranial)
    pts = np.asarray(point) + directions[:, None, :] * radii[None, :, None]
    inside = sample_mask_nearest(bone_mask, pts.reshape(-1, 3)).reshape(len(angles), len(radii))
    hits = [radii[np.flatnonzero(row)[0]] for row in inside if row.any()]
    return float(min(hits)) if hits else float(config.MAX_RAY_MM)


def _smoothed_widths(sections) -> np.ndarray:
    widths = np.array([s.width("outer") if s.ml_bounded else np.inf for s in sections])
    out = widths.copy()
    for i in range(len(widths)):
        window = widths[max(0, i - 1):i + 2]
        finite = window[np.isfinite(window)]
        out[i] = float(np.median(finite)) if (np.isfinite(widths[i]) and finite.size) else np.inf
    return out


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """两两配对斜率的中位数，对个别异常截面稳健。"""
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    valid = np.abs(dx) > 1e-9
    return float(np.median((y[j] - y[i])[valid] / dx[valid])) if np.any(valid) else 0.0


def _fit_axis(sections, direction: np.ndarray, min_sections: int, max_angle_deg: float) -> tuple:
    """
    椎弓根轴：各截面中点相对钉轴的内外、头尾偏移对轴向位置 t 做 Theil–Sen 回归，
    用斜率修正钉轴方向。截面过少或斜率超过 max_angle_deg 时返回钉轴方向。

    （短区间内的中点云做主成分分析对噪声很敏感，回归只把误差放在偏移上。）

    Returns
    -------
    tuple
        (axis, status, fitted_tilt_deg)；status 为 "fitted"、"too_few_sections" 或
        "tilt_above_limit"，fitted_tilt_deg 为回归得到的倾角（被拒绝时也给出）。
    """
    direction = np.asarray(direction, dtype=np.float64)
    if len(sections) < min_sections:
        return direction, "too_few_sections", float("nan")
    t = np.array([s.t for s in sections])
    offsets = np.array([s.chord_center - s.center for s in sections])
    medial, cranial = sections[0].medial, sections[0].cranial
    slope_m = _theil_sen_slope(t, offsets @ medial)
    slope_c = _theil_sen_slope(t, offsets @ cranial)
    tilt = float(np.degrees(np.arctan(np.hypot(slope_m, slope_c))))
    if tilt > max_angle_deg:
        return direction, "tilt_above_limit", tilt
    axis = direction + slope_m * medial + slope_c * cranial
    return axis / np.linalg.norm(axis), "fitted", tilt


def detect_pedicle_isthmus(bone_mask: sitk.Image, plan: ScrewPlan, image: sitk.Image = None,
                           config: IsthmusDetectionConfig = None, frame=None) -> PedicleGeometry:
    """
    沿基准螺钉检测椎弓根峡部（射线测量法）

    Parameters
    ----------
    bone_mask : sitk.Image
        椎体（含椎弓根）二值掩膜
    plan : ScrewPlan
        穿过椎弓根的基准螺钉方案
    image : sitk.Image, optional
        CT图像（HU），用于皮质内缘（内径）
    config : IsthmusDetectionConfig, optional
    frame : object, optional
        具有 medial、cranial 属性的解剖方向（如 AnatomicalFrame）；强烈建议提供

    Raises
    ------
    ValueError
        搜索范围内没有内外两侧都闭合的截面（钉轴不在椎弓根内）
    """
    if config is None:
        config = IsthmusDetectionConfig()
    config.validate()

    direction = np.asarray(plan.direction)
    entry = np.asarray(plan.entry)
    lo_t, hi_t = (f * plan.length for f in config.SEARCH_RANGE)
    t_values = np.arange(lo_t, hi_t + 1e-9, config.AXIAL_STEP)
    sections = [measure_section(bone_mask, entry + t * direction, direction, t, image, frame, config)
                for t in t_values]
    widths = _smoothed_widths(sections)
    if not np.any(np.isfinite(widths)):
        raise ValueError("no section with closed medial and lateral walls found along the screw axis; "
                         "check that the baseline screw passes through the pedicle")
    idx = int(np.argmin(widths))
    limit = config.PEDICLE_WIDTH_FACTOR * widths[idx]
    lo = hi = idx
    while lo - 1 >= 0 and widths[lo - 1] <= limit:
        lo -= 1
    while hi + 1 < len(sections) and widths[hi + 1] <= limit:
        hi += 1

    core = [s for s, w in zip(sections[lo:hi + 1], widths[lo:hi + 1])
            if w <= config.AXIS_CORE_WIDTH_FACTOR * widths[idx]]
    if len(core) < config.MIN_AXIS_SECTIONS:
        core = sections[lo:hi + 1]
    axis, axis_status, axis_tilt = _fit_axis(core, direction, config.MIN_AXIS_SECTIONS,
                                             config.MAX_AXIS_ANGLE_DEG)
    half_step = 0.5 * config.AXIAL_STEP
    start_point = entry + (t_values[lo] - half_step) * direction
    end_point = entry + (t_values[hi] + half_step) * direction
    profile = [{
        "t_mm": float(s.t),
        "width_mm": s.width("outer") if s.ml_bounded else float("inf"),
        "height_mm": s.height("outer") if s.center_in_bone else float("inf"),
        "medial_mm": s.distance(0.0), "lateral_mm": s.distance(180.0),
        "cranial_mm": s.distance(90.0), "caudal_mm": s.distance(270.0),
        "center_in_bone": bool(s.center_in_bone),
        "in_pedicle": bool(lo <= i <= hi),
    } for i, s in enumerate(sections)]

    isthmus = sections[idx]
    if config.REFINE and np.degrees(np.arccos(np.clip(np.dot(axis, direction), -1.0, 1.0))) > 1.0:
        origin = isthmus.chord_center
        candidates = [measure_section(bone_mask, origin + dt * axis, axis, isthmus.t + dt, image, frame, config)
                      for dt in (-config.AXIAL_STEP, 0.0, config.AXIAL_STEP)]
        candidates = [c for c in candidates if c.ml_bounded]
        if candidates:
            refined = min(candidates, key=lambda c: c.width("outer"))
            # an oblique cut only overestimates width; a much smaller value means the re-cut slipped
            if 0.6 * isthmus.width("outer") <= refined.width("outer") <= isthmus.width("outer") + 1e-9:
                isthmus = refined

    return PedicleGeometry(
        isthmus=isthmus,
        axis=np.asarray(axis),
        section_normal=np.asarray(direction),
        start_point=start_point,
        end_point=end_point,
        bone_mask=bone_mask,
        image=image,
        frame=frame,
        config=config,
        profile=profile,
        axis_status=axis_status,
        axis_fit_tilt_deg=axis_tilt,
    )
