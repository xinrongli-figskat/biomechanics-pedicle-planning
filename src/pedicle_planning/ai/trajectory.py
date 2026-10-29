"""
trajectory.py
------------------------------------
椎弓根螺钉钉道采样与骨密度特征

给定一枚三维植入的螺钉（进钉点、方向、长度、直径），在CT图像中沿钉道
体积采样HU值，并按钉道分段（进钉区、椎弓根段、过渡区、椎体段、钉尖段）
以及径向区域（钉芯区、骨-螺钉界面环带）计算骨密度统计特征。

坐标约定：所有点与方向均位于CT图像的物理坐标系（SimpleITK物理空间，
单位毫米）。物理坐标到体素坐标的转换使用图像的原点、间距与方向矩阵，
因此支持非单位方向矩阵的图像。

本模块只新增功能，不修改现有评分器与优化器的行为。
"""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


SEGMENT_NAMES = ("entry", "pedicle", "transition", "body", "tip")
REGION_NAMES = ("core", "interface", "all")
SECTOR_REGION_NAMES = ("interface_medial", "interface_lateral", "interface_cranial", "interface_caudal")


@dataclass(frozen=True)
class ScrewPlan:
    """
    螺钉置钉方案

    Parameters
    ----------
    entry : tuple
        进钉点物理坐标 (x, y, z)，单位毫米
    direction : tuple
        由进钉点指向钉尖的方向向量，内部会归一化
    length : float
        螺钉长度，单位毫米
    diameter : float
        螺钉外径，单位毫米
    """

    entry: tuple
    direction: tuple
    length: float
    diameter: float

    def __post_init__(self):
        entry = np.asarray(self.entry, dtype=np.float64)
        direction = np.asarray(self.direction, dtype=np.float64)
        if entry.shape != (3,) or not np.all(np.isfinite(entry)):
            raise ValueError("entry must be 3 finite coordinates")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be 3 finite components")
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            raise ValueError("direction must be a non-zero vector")
        if not self.length > 0:
            raise ValueError(f"length ({self.length}) must be positive")
        if not self.diameter > 0:
            raise ValueError(f"diameter ({self.diameter}) must be positive")
        object.__setattr__(self, "entry", tuple(float(v) for v in entry))
        object.__setattr__(self, "direction", tuple(float(v) for v in direction / norm))
        object.__setattr__(self, "length", float(self.length))
        object.__setattr__(self, "diameter", float(self.diameter))

    @classmethod
    def from_entry_tip(cls, entry: tuple, tip: tuple, diameter: float) -> "ScrewPlan":
        """由进钉点与钉尖点构造方案，长度取两点距离。"""
        entry_arr = np.asarray(entry, dtype=np.float64)
        tip_arr = np.asarray(tip, dtype=np.float64)
        axis = tip_arr - entry_arr
        return cls(tuple(entry_arr), tuple(axis), float(np.linalg.norm(axis)), diameter)

    @property
    def tip(self) -> tuple:
        """钉尖物理坐标。"""
        tip = np.asarray(self.entry) + self.length * np.asarray(self.direction)
        return tuple(float(v) for v in tip)


@dataclass(frozen=True)
class LinearCalibration:
    """
    HU到骨密度的线性校准：BMD = slope * HU + intercept

    用于体模校准或内部校准（如空气/脂肪/肌肉/血液）。
    未提供校准时，特征直接以HU表示。
    """

    slope: float
    intercept: float
    unit: str = "mg/cm3"

    @classmethod
    def from_two_points(cls, hu_a: float, value_a: float, hu_b: float, value_b: float,
                        unit: str = "mg/cm3") -> "LinearCalibration":
        """由两个参考点（如两种体模插件）求校准直线。"""
        if hu_a == hu_b:
            raise ValueError("calibration reference HU values must differ")
        slope = (value_b - value_a) / (hu_b - hu_a)
        return cls(float(slope), float(value_a - slope * hu_a), unit)

    def apply(self, hu: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(hu, dtype=np.float64) + self.intercept


@dataclass
class TrajectorySamplingConfig:
    """
    钉道采样配置参数

    SEGMENT_FRACTIONS 给出各段沿钉轴的分界（相对螺钉长度的比例，
    从进钉点0到钉尖1），依次对应 entry, pedicle, transition, body, tip。
    已知椎弓根峡部位置时，可通过 compute_trajectory_features 的
    segment_bounds_mm 参数以毫米直接指定分界。
    """

    AXIAL_STEP: float = 0.5
    """沿钉轴的采样间距，单位毫米。"""

    CROSS_STEP: float = 0.5
    """横截面内的采样网格间距，单位毫米。"""

    INTERFACE_THICKNESS: float = 1.0
    """骨-螺钉界面环带厚度（螺钉外径之外），单位毫米。"""

    TIP_SPHERE_RADIUS: float = 4.0
    """钉尖周围球形区域半径，单位毫米。"""

    SEGMENT_FRACTIONS: tuple = (0.0, 0.2, 0.45, 0.6, 0.8, 1.0)

    LOW_DENSITY_HU: float = 110.0
    """低密度阈值（HU），用于计算低密度体素比例。"""

    CORTICAL_HU: float = 800.0
    """皮质骨阈值（HU），用于计算高密度（皮质接触）体素比例。"""

    def validate(self) -> None:
        for name in ("AXIAL_STEP", "CROSS_STEP", "TIP_SPHERE_RADIUS"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} ({getattr(self, name)}) must be positive")
        if self.INTERFACE_THICKNESS < 0:
            raise ValueError(f"INTERFACE_THICKNESS ({self.INTERFACE_THICKNESS}) must be non-negative")
        _validate_bounds(self.SEGMENT_FRACTIONS, 0.0, 1.0)


@dataclass
class RegionStats:
    """
    单个区域的骨密度统计

    mean/median/std/p10/p90 在提供校准时为校准后单位，否则为HU；
    low_density_fraction 与 cortical_fraction 始终按HU阈值计算；
    bone_fraction 为图像内采样点中位于骨内的比例（未提供骨掩膜/网格时为1）。
    """

    n_samples: int
    n_valid: int
    bone_fraction: float
    mean: float
    median: float
    std: float
    p10: float
    p90: float
    cv: float
    low_density_fraction: float
    cortical_fraction: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TrajectoryFeatures:
    """
    钉道骨密度特征

    segments[段名][区域名] -> RegionStats，段名见 SEGMENT_NAMES，
    区域名见 REGION_NAMES（core: 钉芯，interface: 界面环带，all: 二者合并）；
    提供解剖方向 frame 时另含 SECTOR_REGION_NAMES：界面环带的内侧/外侧半环与
    头侧/尾侧半环（以垂直于钉轴的内侧、头侧方向划分，各占180°）。
    in_bone_length 为钉轴中心线位于骨内（且在图像内）的长度；未提供骨掩膜/网格时
    仅反映位于图像内的长度。
    """

    plan: ScrewPlan
    unit: str
    segment_bounds_mm: tuple
    segments: dict
    whole: dict
    tip_sphere: RegionStats
    in_bone_length: float
    outside_image_fraction: float

    def to_flat_dict(self) -> dict:
        """展平为 {特征名: 数值}，便于作为机器学习模型的输入。"""
        flat = {}
        for seg_name, regions in self.segments.items():
            for region_name, stats in regions.items():
                for key, value in stats.as_dict().items():
                    flat[f"{seg_name}_{region_name}_{key}"] = value
        for region_name, stats in self.whole.items():
            for key, value in stats.as_dict().items():
                flat[f"whole_{region_name}_{key}"] = value
        for key, value in self.tip_sphere.as_dict().items():
            flat[f"tip_sphere_{key}"] = value
        flat["in_bone_length"] = self.in_bone_length
        flat["in_bone_fraction"] = self.in_bone_length / self.plan.length
        flat["outside_image_fraction"] = self.outside_image_fraction
        flat["screw_length"] = self.plan.length
        flat["screw_diameter"] = self.plan.diameter
        return flat


def _validate_bounds(bounds, low, high) -> None:
    if len(bounds) != len(SEGMENT_NAMES) + 1:
        raise ValueError(f"segment bounds must have {len(SEGMENT_NAMES) + 1} values, got {len(bounds)}")
    values = np.asarray(bounds, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0):
        raise ValueError("segment bounds must be finite and non-decreasing")
    if values[0] < low - 1e-9 or values[-1] > high + 1e-9:
        raise ValueError(f"segment bounds must lie within [{low}, {high}]")


def _orthonormal_frame(direction: np.ndarray) -> tuple:
    """构造以 direction 为 z 轴的正交基 (u, v)。"""
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    return u, v


def sample_screw_points(plan: ScrewPlan, config: TrajectorySamplingConfig = None) -> dict:
    """
    在螺钉圆柱及其界面环带内生成均匀体积采样点

    Returns
    -------
    dict
        points: (N, 3) 物理坐标
        t: (N,) 沿钉轴距进钉点的距离（毫米）
        r: (N,) 到钉轴的径向距离（毫米）
        is_core: (N,) 是否位于螺钉外径以内
        offsets: (N, 3) 采样点相对钉轴的垂直偏移向量
        axis_points: (M, 3) 钉轴中心线采样点
        axis_t: (M,) 中心线采样点的轴向位置
    """
    if config is None:
        config = TrajectorySamplingConfig()
    config.validate()

    direction = np.asarray(plan.direction)
    entry = np.asarray(plan.entry)
    u, v = _orthonormal_frame(direction)

    n_axial = max(1, int(np.ceil(plan.length / config.AXIAL_STEP)))
    axis_t = (np.arange(n_axial) + 0.5) * (plan.length / n_axial)

    core_radius = plan.diameter / 2.0
    outer_radius = core_radius + config.INTERFACE_THICKNESS
    half = int(np.ceil(outer_radius / config.CROSS_STEP))
    grid = (np.arange(-half, half + 1) + 0.0) * config.CROSS_STEP
    gu, gv = np.meshgrid(grid, grid, indexing="ij")
    gu, gv = gu.ravel(), gv.ravel()
    gr = np.hypot(gu, gv)
    keep = gr <= outer_radius + 1e-9
    gu, gv, gr = gu[keep], gv[keep], gr[keep]

    t = np.repeat(axis_t, gu.size)
    r = np.tile(gr, n_axial)
    offsets = np.outer(np.tile(gu, n_axial), u) + np.outer(np.tile(gv, n_axial), v)
    points = entry + np.outer(t, direction) + offsets

    return {
        "points": points,
        "t": t,
        "r": r,
        "is_core": r <= core_radius + 1e-9,
        "offsets": offsets,
        "axis_points": entry + np.outer(axis_t, direction),
        "axis_t": axis_t,
    }


def sample_sphere_points(center: tuple, radius: float, step: float) -> np.ndarray:
    """在球内生成均匀网格采样点。"""
    half = int(np.ceil(radius / step))
    grid = np.arange(-half, half + 1) * step
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    offsets = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    offsets = offsets[np.linalg.norm(offsets, axis=1) <= radius + 1e-9]
    return np.asarray(center, dtype=np.float64) + offsets


def physical_to_continuous_index(image: sitk.Image, points: np.ndarray) -> np.ndarray:
    """
    批量将物理坐标转换为连续体素索引 (i, j, k)，考虑原点、间距与方向矩阵。
    """
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    index_to_physical = direction * spacing
    return np.linalg.solve(index_to_physical, (np.asarray(points, dtype=np.float64) - origin).T).T


def sample_image_trilinear(image: sitk.Image, points: np.ndarray) -> np.ndarray:
    """
    三线性插值采样图像值；超出图像体素中心范围的点返回 NaN。

    使用 GetArrayViewFromImage 避免复制整幅图像，批量评估多个方案时开销只与采样点数相关。
    """
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("image must be a 3D scalar image")
    array = sitk.GetArrayViewFromImage(image)
    size = np.asarray(image.GetSize())

    idx = physical_to_continuous_index(image, points)
    eps = 1e-6
    inside = np.all((idx >= -eps) & (idx <= size - 1 + eps), axis=1)
    values = np.full(idx.shape[0], np.nan)
    if not np.any(inside):
        return values

    ci = np.clip(idx[inside], 0.0, size - 1)
    i0 = np.clip(np.floor(ci).astype(np.int64), 0, np.maximum(size - 2, 0))
    frac = ci - i0
    i1 = np.minimum(i0 + 1, size - 1)

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

    def at(z, y, x):
        return array[z, y, x].astype(np.float64)

    c000 = at(z0, y0, x0)
    c100 = at(z0, y0, x1)
    c010 = at(z0, y1, x0)
    c110 = at(z0, y1, x1)
    c001 = at(z1, y0, x0)
    c101 = at(z1, y0, x1)
    c011 = at(z1, y1, x0)
    c111 = at(z1, y1, x1)

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    values[inside] = c0 * (1 - fz) + c1 * fz
    return values


def sample_mask_nearest(mask: sitk.Image, points: np.ndarray) -> np.ndarray:
    """最近邻采样二值掩膜（非零为骨）；超出掩膜范围的点视为非骨。"""
    array = sitk.GetArrayViewFromImage(mask)
    size = np.asarray(mask.GetSize())
    idx = np.rint(physical_to_continuous_index(mask, points)).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < size), axis=1)
    result = np.zeros(idx.shape[0], dtype=bool)
    sel = idx[inside]
    result[inside] = array[sel[:, 2], sel[:, 1], sel[:, 0]] != 0
    return result


def points_inside_mesh(mesh, points: np.ndarray) -> np.ndarray:
    """判断点是否位于封闭骨网格内部（PyVista/VTK）。"""
    import pyvista as pv

    cloud = pv.PolyData(np.asarray(points, dtype=np.float64))
    if hasattr(cloud, "select_interior_points"):
        # PyVista >= 0.47; select_enclosed_points is deprecated there.
        selected = cloud.select_interior_points(mesh, check_surface=False)
        return np.asarray(selected.point_data["selected_points"], dtype=bool)
    selected = cloud.select_enclosed_points(mesh, check_surface=False)
    return np.asarray(selected.point_data["SelectedPoints"], dtype=bool)


def _bone_membership(points, bone_mask, bone_mesh) -> np.ndarray:
    if bone_mask is not None:
        return sample_mask_nearest(bone_mask, points)
    if bone_mesh is not None:
        return points_inside_mesh(bone_mesh, points)
    return np.ones(len(points), dtype=bool)


def _perpendicular_unit(vector, axis: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    v = v - np.dot(v, axis) * axis
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        raise ValueError(f"{name} direction must not be parallel to the screw axis")
    return v / norm


def sector_masks(offsets: np.ndarray, direction, frame) -> dict:
    """
    按解剖方向把采样点划分为内侧/外侧、头侧/尾侧半环

    Parameters
    ----------
    offsets : (N, 3)
        采样点相对钉轴的垂直偏移（sample_screw_points 的 offsets）
    direction : tuple
        钉轴方向
    frame : object
        具有 medial、cranial 属性的解剖方向（如 AnatomicalFrame）
    """
    axis = np.asarray(direction, dtype=np.float64)
    medial = _perpendicular_unit(frame.medial, axis, "medial")
    cranial = _perpendicular_unit(frame.cranial, axis, "cranial")
    m = offsets @ medial
    c = offsets @ cranial
    return {
        "interface_medial": m > 1e-9,
        "interface_lateral": m < -1e-9,
        "interface_cranial": c > 1e-9,
        "interface_caudal": c < -1e-9,
    }


def _region_stats(values: np.ndarray, in_bone: np.ndarray, hu_values: np.ndarray,
                  config: TrajectorySamplingConfig) -> RegionStats:
    n_samples = int(values.size)
    in_image = np.isfinite(values)
    n_in_image = int(np.count_nonzero(in_image))
    valid = in_image & in_bone
    n_valid = int(np.count_nonzero(valid))
    bone_fraction = float(np.count_nonzero(in_image & in_bone) / n_in_image) if n_in_image else float("nan")

    if n_valid == 0:
        nan = float("nan")
        return RegionStats(n_samples, 0, bone_fraction, nan, nan, nan, nan, nan, nan, nan, nan)

    v = values[valid]
    hu = hu_values[valid]
    mean = float(np.mean(v))
    std = float(np.std(v))
    return RegionStats(
        n_samples=n_samples,
        n_valid=n_valid,
        bone_fraction=bone_fraction,
        mean=mean,
        median=float(np.median(v)),
        std=std,
        p10=float(np.percentile(v, 10)),
        p90=float(np.percentile(v, 90)),
        cv=float(std / abs(mean)) if abs(mean) > 1e-9 else float("nan"),
        low_density_fraction=float(np.mean(hu < config.LOW_DENSITY_HU)),
        cortical_fraction=float(np.mean(hu >= config.CORTICAL_HU)),
    )


def compute_trajectory_features(image: sitk.Image, plan: ScrewPlan,
                                bone_mask: sitk.Image = None, bone_mesh=None,
                                calibration: LinearCalibration = None,
                                config: TrajectorySamplingConfig = None,
                                segment_bounds_mm: tuple = None,
                                frame=None) -> TrajectoryFeatures:
    """
    计算给定螺钉方案的钉道骨密度特征

    Parameters
    ----------
    image : sitk.Image
        CT图像（HU）
    plan : ScrewPlan
        螺钉方案（物理坐标）
    bone_mask : sitk.Image, optional
        骨（椎体）二值掩膜；提供后仅统计骨内采样点
    bone_mesh : pyvista.PolyData, optional
        封闭骨网格；未提供 bone_mask 时用于判断骨内采样点
    calibration : LinearCalibration, optional
        HU到BMD的线性校准；未提供时特征单位为HU
    config : TrajectorySamplingConfig, optional
        采样配置
    segment_bounds_mm : tuple, optional
        6个分界值（毫米，0到螺钉长度），覆盖 SEGMENT_FRACTIONS
    frame : object, optional
        具有 medial、cranial 属性的解剖方向（如 AnatomicalFrame）；提供时
        额外计算界面环带的内侧/外侧、头侧/尾侧半环统计

    Returns
    -------
    TrajectoryFeatures
    """
    if config is None:
        config = TrajectorySamplingConfig()
    config.validate()

    if segment_bounds_mm is None:
        bounds = tuple(float(f) * plan.length for f in config.SEGMENT_FRACTIONS)
    else:
        _validate_bounds(segment_bounds_mm, 0.0, plan.length)
        bounds = tuple(float(b) for b in segment_bounds_mm)

    samples = sample_screw_points(plan, config)
    hu = sample_image_trilinear(image, samples["points"])
    in_bone = _bone_membership(samples["points"], bone_mask, bone_mesh)
    values = calibration.apply(hu) if calibration is not None else hu
    unit = calibration.unit if calibration is not None else "HU"

    t = samples["t"]
    is_core = samples["is_core"]
    region_masks = {
        "core": is_core,
        "interface": ~is_core,
        "all": np.ones_like(is_core),
    }
    if frame is not None:
        for name, sector in sector_masks(samples["offsets"], plan.direction, frame).items():
            region_masks[name] = ~is_core & sector

    segments = {}
    for k, name in enumerate(SEGMENT_NAMES):
        lo, hi = bounds[k], bounds[k + 1]
        last = k == len(SEGMENT_NAMES) - 1
        in_segment = (t >= lo) & ((t <= hi) if last else (t < hi))
        segments[name] = {
            region: _region_stats(values[in_segment & m], in_bone[in_segment & m],
                                  hu[in_segment & m], config)
            for region, m in region_masks.items()
        }

    whole = {
        region: _region_stats(values[m], in_bone[m], hu[m], config)
        for region, m in region_masks.items()
    }

    tip_points = sample_sphere_points(plan.tip, config.TIP_SPHERE_RADIUS, config.CROSS_STEP)
    tip_hu = sample_image_trilinear(image, tip_points)
    tip_values = calibration.apply(tip_hu) if calibration is not None else tip_hu
    tip_stats = _region_stats(tip_values, _bone_membership(tip_points, bone_mask, bone_mesh),
                              tip_hu, config)

    axis_points = samples["axis_points"]
    axis_in_image = np.isfinite(sample_image_trilinear(image, axis_points))
    axis_in_bone = _bone_membership(axis_points, bone_mask, bone_mesh) & axis_in_image
    in_bone_length = float(np.count_nonzero(axis_in_bone) * plan.length / axis_points.shape[0])

    return TrajectoryFeatures(
        plan=plan,
        unit=unit,
        segment_bounds_mm=bounds,
        segments=segments,
        whole=whole,
        tip_sphere=tip_stats,
        in_bone_length=in_bone_length,
        outside_image_fraction=float(np.mean(~np.isfinite(hu))),
    )
