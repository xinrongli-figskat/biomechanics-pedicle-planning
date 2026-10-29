"""
candidates.py
------------------------------------
候选置钉方案生成、安全约束过滤与风险最小化选择

以经验置钉方案为初始解，在解剖坐标系下离散调整进钉点（内外/头尾平移）、
内倾角、头尾倾角、螺钉长度与直径，生成有限候选集；候选方案先经安全硬约束
过滤，再逐一重新计算钉道特征并输入风险评估函数，选取预测风险最低的方案。

角度调整的支点可选进钉点（PIVOT="entry"）或基准钉轴穿过椎弓根峡部的点
（PIVOT="isthmus"）。调整后的进钉点沿新钉轴重新贴回骨表面（SNAP_ENTRY），
螺钉长度从贴合后的进钉点起算。

风险评估函数由调用方提供（如训练好的松动预测模型），签名为
risk_fn(features: dict) -> float，其中 features 为
TrajectoryFeatures.to_flat_dict() 加上可用时的 match_* 轨迹匹配特征。

本模块提供的内置安全约束只覆盖可由骨掩膜与峡部几何判断的条件（螺钉直径与
峡部内径、峡部壁距、钉尖骨内余量）；椎管/神经根距离、终板、关节突等需要
相应分割结果，可通过自定义约束函数接入。
"""

from dataclasses import dataclass, field

import numpy as np

from .trajectory import (
    ScrewPlan,
    TrajectorySamplingConfig,
    _bone_membership,
    compute_trajectory_features,
    points_inside_mesh,
    sample_mask_nearest,
)


PIVOT_OPTIONS = ("entry", "isthmus")
SNAP_STATUSES = ("not_requested", "no_bone_data", "snapped", "failed")


@dataclass(frozen=True)
class AnatomicalFrame:
    """
    椎体解剖方向

    medial 指向内侧（对该侧螺钉而言朝向中线），cranial 指向头侧；
    cranial 会被正交化到 medial 上。
    """

    medial: tuple
    cranial: tuple

    def __post_init__(self):
        m = np.asarray(self.medial, dtype=np.float64)
        c = np.asarray(self.cranial, dtype=np.float64)
        if m.shape != (3,) or c.shape != (3,) or np.linalg.norm(m) < 1e-9:
            raise ValueError("medial and cranial must be non-zero 3-vectors")
        m = m / np.linalg.norm(m)
        c = c - np.dot(c, m) * m
        if np.linalg.norm(c) < 1e-6:
            raise ValueError("cranial must not be parallel to medial")
        c = c / np.linalg.norm(c)
        object.__setattr__(self, "medial", tuple(float(x) for x in m))
        object.__setattr__(self, "cranial", tuple(float(x) for x in c))


@dataclass
class CandidateGridConfig:
    """
    候选方案离散网格（偏移量相对经验方案）

    默认：进钉点内外平移 -2/-1/0/+1/+2 mm，内倾角 -5/0/+5°，螺钉长度 -5/0/+5 mm，
    直径 +0/+0.5 mm，共 5×3×3×2 = 90 个候选。头尾平移与头尾倾角默认不调整。
    正值含义：MEDIAL_* 向内侧，CRANIAL_* 向头侧；角度为钉尖方向的转动。

    PIVOT 为角度调整的支点："entry" 绕（平移后的）进钉点转动；"isthmus" 绕基准钉轴
    穿过峡部截面的点转动（需提供 PedicleGeometry 或显式支点），进钉点随角度移动，
    钉轴始终经过（平移后的）峡部支点。平移量作用于整条钉轴。

    SNAP_ENTRY 为 True 且提供骨掩膜/骨网格时，非基准候选的进钉点沿新钉轴在
    [-SNAP_SEARCH_BACK_MM, +SNAP_SEARCH_FORWARD_MM] 范围内搜索“骨外→骨内”的
    表面交点，取离原进钉点最近者；找不到时该候选标记为 failed，
    由 entry_on_bone_surface 约束剔除。经验方案本身保持不变。
    """

    MEDIAL_OFFSETS_MM: tuple = (-2.0, -1.0, 0.0, 1.0, 2.0)
    CRANIAL_OFFSETS_MM: tuple = (0.0,)
    MEDIAL_ANGLES_DEG: tuple = (-5.0, 0.0, 5.0)
    CRANIAL_ANGLES_DEG: tuple = (0.0,)
    LENGTH_OFFSETS_MM: tuple = (-5.0, 0.0, 5.0)
    DIAMETER_OFFSETS_MM: tuple = (0.0, 0.5)
    PIVOT: str = "entry"
    SNAP_ENTRY: bool = True
    SNAP_SEARCH_BACK_MM: float = 10.0
    SNAP_SEARCH_FORWARD_MM: float = 10.0
    SNAP_STEP_MM: float = 0.25

    def validate(self) -> None:
        for name in ("MEDIAL_OFFSETS_MM", "CRANIAL_OFFSETS_MM", "MEDIAL_ANGLES_DEG",
                     "CRANIAL_ANGLES_DEG", "LENGTH_OFFSETS_MM", "DIAMETER_OFFSETS_MM"):
            values = getattr(self, name)
            if len(values) == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a non-empty tuple of finite values")
        for name in ("MEDIAL_ANGLES_DEG", "CRANIAL_ANGLES_DEG"):
            if np.any(np.abs(getattr(self, name)) >= 90.0):
                raise ValueError(f"{name} must lie strictly within (-90, 90) degrees")
        if self.PIVOT not in PIVOT_OPTIONS:
            raise ValueError(f"PIVOT ({self.PIVOT!r}) must be one of {PIVOT_OPTIONS}")
        for name in ("SNAP_SEARCH_BACK_MM", "SNAP_SEARCH_FORWARD_MM"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} ({getattr(self, name)}) must be non-negative")
        if not self.SNAP_STEP_MM > 0:
            raise ValueError(f"SNAP_STEP_MM ({self.SNAP_STEP_MM}) must be positive")


@dataclass(frozen=True)
class Candidate:
    """
    候选方案及其相对经验方案的调整量

    entry_snap_status 取值见 SNAP_STATUSES；entry_snap_shift_mm 为贴合骨面时
    进钉点沿钉轴的移动量（正值向钉尖方向）。
    """

    plan: ScrewPlan
    medial_offset_mm: float = 0.0
    cranial_offset_mm: float = 0.0
    medial_angle_deg: float = 0.0
    cranial_angle_deg: float = 0.0
    is_baseline: bool = False
    pivot: str = "entry"
    entry_snap_status: str = "not_requested"
    entry_snap_shift_mm: float = 0.0


def _rotate(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues 旋转。"""
    k = axis / np.linalg.norm(axis)
    return (vector * np.cos(angle_rad) + np.cross(k, vector) * np.sin(angle_rad)
            + k * np.dot(k, vector) * (1.0 - np.cos(angle_rad)))


def _rotate_toward(vector: np.ndarray, pivot: np.ndarray, target: np.ndarray,
                   angle_deg: float) -> np.ndarray:
    """绕 pivot 旋转 vector，正角度使 vector 向 target 方向转动。"""
    if angle_deg == 0.0:
        return vector
    pivot = pivot - np.dot(pivot, vector) * vector
    if np.linalg.norm(pivot) < 1e-9:
        raise ValueError("screw direction is parallel to the rotation pivot axis")
    pivot = pivot / np.linalg.norm(pivot)
    if np.dot(np.cross(pivot, vector), target) < 0:
        pivot = -pivot
    return _rotate(vector, pivot, np.radians(angle_deg))


def adjust_plan(baseline: ScrewPlan, frame: AnatomicalFrame, medial_offset_mm: float = 0.0,
                cranial_offset_mm: float = 0.0, medial_angle_deg: float = 0.0,
                cranial_angle_deg: float = 0.0, length: float = None,
                diameter: float = None, pivot_point=None) -> ScrewPlan:
    """
    对经验方案做一次调整（纯几何，不贴合骨面）

    内倾角绕头尾轴转动（横断面内），头尾倾角绕与钉轴、头尾轴都垂直的轴转动
    （矢状面内）；整条钉轴沿 medial/cranial 方向平移。

    pivot_point 为 None 时以（平移后的）进钉点为支点；否则以（平移后的）
    pivot_point 为支点，新钉轴经过该点，进钉点位于新钉轴上、与支点的距离
    等于基准方案中进钉点到支点在钉轴上的投影距离。
    """
    medial = np.asarray(frame.medial)
    cranial = np.asarray(frame.cranial)
    base_direction = np.asarray(baseline.direction)
    shift = medial_offset_mm * medial + cranial_offset_mm * cranial

    direction = _rotate_toward(base_direction, cranial, medial, medial_angle_deg)
    sagittal_pivot = np.cross(cranial, direction)
    direction = _rotate_toward(direction, sagittal_pivot, cranial, cranial_angle_deg)

    base_entry = np.asarray(baseline.entry)
    if pivot_point is None:
        entry = base_entry + shift
    else:
        pivot = np.asarray(pivot_point, dtype=np.float64)
        depth = float(np.dot(pivot - base_entry, base_direction))
        entry = pivot + shift - depth * direction
    return ScrewPlan(
        tuple(entry),
        tuple(direction),
        baseline.length if length is None else length,
        baseline.diameter if diameter is None else diameter,
    )


def snap_entry_to_surface(plan: ScrewPlan, bone_mask=None, bone_mesh=None,
                          search_back_mm: float = 10.0, search_forward_mm: float = 10.0,
                          step_mm: float = 0.25) -> tuple:
    """
    沿钉轴把进钉点移到骨表面

    在进钉点前后 [-search_back_mm, +search_forward_mm] 范围内按 step_mm 采样，
    找出所有“骨外→骨内”的交点，取离原进钉点最近者；方向、长度、直径不变，
    钉尖随进钉点一起移动。

    Returns
    -------
    tuple
        (贴合后的 ScrewPlan, 沿钉轴移动量)；找不到交点时为 (None, nan)
    """
    if bone_mask is None and bone_mesh is None:
        raise ValueError("bone_mask or bone_mesh is required to snap the entry point")
    direction = np.asarray(plan.direction)
    s = np.arange(-search_back_mm, search_forward_mm + 1e-9, step_mm)
    points = np.asarray(plan.entry) + np.outer(s, direction)
    inside = _bone_membership(points, bone_mask, bone_mesh)
    crossings = np.flatnonzero(~inside[:-1] & inside[1:])
    if crossings.size == 0:
        return None, float("nan")
    crossing_s = 0.5 * (s[crossings] + s[crossings + 1])
    shift = float(crossing_s[np.argmin(np.abs(crossing_s))])
    entry = np.asarray(plan.entry) + shift * direction
    return ScrewPlan(tuple(entry), plan.direction, plan.length, plan.diameter), shift


def resolve_pivot_point(baseline: ScrewPlan, pivot: str, pedicle=None, pivot_point=None):
    """返回角度调整的支点坐标；PIVOT="entry" 时返回 None。"""
    if pivot not in PIVOT_OPTIONS:
        raise ValueError(f"pivot ({pivot!r}) must be one of {PIVOT_OPTIONS}")
    if pivot == "entry":
        return None
    if pivot_point is not None:
        return np.asarray(pivot_point, dtype=np.float64)
    if pedicle is None:
        raise ValueError('PIVOT="isthmus" requires a PedicleGeometry or an explicit pivot_point')
    return np.asarray(pedicle.isthmus_crossing(baseline), dtype=np.float64)


def generate_candidates(baseline: ScrewPlan, frame: AnatomicalFrame,
                        config: CandidateGridConfig = None, lengths: tuple = None,
                        diameters: tuple = None, pedicle=None, pivot_point=None,
                        bone_mask=None, bone_mesh=None) -> list:
    """
    生成候选方案列表，第一个元素始终为经验方案本身（is_baseline=True）

    Parameters
    ----------
    lengths, diameters : tuple, optional
        临床可选规格（绝对值，毫米）；提供时替代 LENGTH/DIAMETER_OFFSETS_MM
    pedicle : PedicleGeometry, optional
        PIVOT="isthmus" 时用于确定支点
    pivot_point : tuple, optional
        显式指定 PIVOT="isthmus" 的支点（覆盖 pedicle）
    bone_mask, bone_mesh : optional
        用于把进钉点贴回骨表面（SNAP_ENTRY）
    """
    if config is None:
        config = CandidateGridConfig()
    config.validate()
    pivot = resolve_pivot_point(baseline, config.PIVOT, pedicle, pivot_point)
    has_bone = bone_mask is not None or bone_mesh is not None

    if lengths is None:
        lengths = tuple(baseline.length + d for d in config.LENGTH_OFFSETS_MM)
    if diameters is None:
        diameters = tuple(baseline.diameter + d for d in config.DIAMETER_OFFSETS_MM)
    lengths = tuple(sorted({float(x) for x in lengths if x > 0}))
    diameters = tuple(sorted({float(x) for x in diameters if x > 0}))
    if not lengths or not diameters:
        raise ValueError("at least one positive length and diameter are required")

    candidates = [Candidate(baseline, is_baseline=True, pivot=config.PIVOT)]
    for dm in config.MEDIAL_OFFSETS_MM:
        for dc in config.CRANIAL_OFFSETS_MM:
            for am in config.MEDIAL_ANGLES_DEG:
                for ac in config.CRANIAL_ANGLES_DEG:
                    for length in lengths:
                        for diameter in diameters:
                            if (dm == 0 and dc == 0 and am == 0 and ac == 0
                                    and length == baseline.length and diameter == baseline.diameter):
                                continue
                            plan = adjust_plan(baseline, frame, dm, dc, am, ac, length, diameter, pivot)
                            status, snap_shift = "not_requested", 0.0
                            if config.SNAP_ENTRY and not has_bone:
                                status = "no_bone_data"
                            elif config.SNAP_ENTRY:
                                snapped, snap_shift = snap_entry_to_surface(
                                    plan, bone_mask, bone_mesh, config.SNAP_SEARCH_BACK_MM,
                                    config.SNAP_SEARCH_FORWARD_MM, config.SNAP_STEP_MM)
                                if snapped is None:
                                    status = "failed"
                                else:
                                    status, plan = "snapped", snapped
                            candidates.append(Candidate(plan, float(dm), float(dc), float(am), float(ac),
                                                        pivot=config.PIVOT, entry_snap_status=status,
                                                        entry_snap_shift_mm=snap_shift))
    return candidates


@dataclass
class ConstraintContext:
    """约束函数可用的数据。"""

    image: object = None
    bone_mask: object = None
    bone_mesh: object = None
    pedicle: object = None


def diameter_within_isthmus(margin_mm: float = 1.0):
    """螺钉直径不得超过椎弓根峡部最小内径减 margin_mm。"""
    def check(candidate, features, ctx):
        if ctx.pedicle is None:
            return None
        limit = ctx.pedicle.inner_min_width - margin_mm
        if candidate.plan.diameter > limit + 1e-9:
            return f"diameter {candidate.plan.diameter:.2f} mm exceeds isthmus inner width - {margin_mm} mm ({limit:.2f} mm)"
        return None
    return check


def isthmus_wall_clearance(min_clearance_mm: float = 0.0):
    """峡部处螺钉表面到椎弓根外轮廓的距离不小于 min_clearance_mm（防止穿破椎弓根壁）。"""
    def check(candidate, features, ctx):
        if ctx.pedicle is None:
            return None
        clearance = ctx.pedicle.trajectory_match(candidate.plan)["outer_clearance_mm"]
        if clearance < min_clearance_mm - 1e-9:
            return f"pedicle wall clearance {clearance:.2f} mm < {min_clearance_mm} mm"
        return None
    return check


def tip_inside_bone(margin_mm: float = 2.0, n_directions: int = 26):
    """钉尖周围 margin_mm 半径内全部位于骨内（近似钉尖到皮质的安全距离）。"""
    def check(candidate, features, ctx):
        if ctx.bone_mask is None and ctx.bone_mesh is None:
            return None
        tip = np.asarray(candidate.plan.tip)
        offsets = [np.zeros(3)]
        for d in _sphere_directions(n_directions):
            offsets.extend([d * margin_mm, d * margin_mm * 0.5])
        points = tip + np.asarray(offsets)
        if ctx.bone_mask is not None:
            inside = sample_mask_nearest(ctx.bone_mask, points)
        else:
            inside = points_inside_mesh(ctx.bone_mesh, points)
        if not np.all(inside):
            return f"screw tip within {margin_mm} mm of the bone surface"
        return None
    return check


def entry_on_bone_surface():
    """进钉点必须能沿钉轴贴合到骨表面（SNAP_ENTRY 搜索失败的候选剔除）。"""
    def check(candidate, features, ctx):
        if candidate.entry_snap_status == "failed":
            return "entry point could not be placed on the bone surface along the screw axis"
        return None
    return check


def _sphere_directions(n: int) -> np.ndarray:
    """近似均匀的单位方向（Fibonacci 球面）。"""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def default_constraints() -> list:
    """内置约束；缺少所需数据（峡部几何/骨掩膜）的约束自动跳过。"""
    return [entry_on_bone_surface(), diameter_within_isthmus(1.0), isthmus_wall_clearance(0.0),
            tip_inside_bone(2.0)]


@dataclass
class CandidateEvaluation:
    candidate: Candidate
    feasible: bool
    violations: list
    features: dict = None
    risk: float = float("nan")


@dataclass
class OptimizationResult:
    """优化结果；best 为可行方案中预测风险最低者，无可行方案时为 None。"""

    baseline: CandidateEvaluation
    best: CandidateEvaluation
    evaluations: list = field(default_factory=list)
    frame: AnatomicalFrame = None

    @property
    def n_candidates(self) -> int:
        return len(self.evaluations)

    @property
    def n_feasible(self) -> int:
        return sum(1 for e in self.evaluations if e.feasible)

    def summary(self) -> dict:
        """经验方案与优化方案的风险、调整量与风险下降幅度。"""
        best = self.best
        baseline_risk = self.baseline.risk
        result = {
            "baseline_risk": baseline_risk,
            "baseline_feasible": self.baseline.feasible,
            "n_candidates": self.n_candidates,
            "n_feasible": self.n_feasible,
        }
        if best is None:
            result["best_risk"] = float("nan")
            return result
        c = best.candidate
        result.update({
            "best_risk": best.risk,
            "risk_reduction": baseline_risk - best.risk,
            "risk_reduction_relative": ((baseline_risk - best.risk) / baseline_risk
                                        if np.isfinite(baseline_risk) and baseline_risk != 0.0
                                        else float("nan")),
            "pivot": c.pivot,
            "medial_offset_mm": c.medial_offset_mm,
            "cranial_offset_mm": c.cranial_offset_mm,
            "medial_angle_deg": c.medial_angle_deg,
            "cranial_angle_deg": c.cranial_angle_deg,
            "length_mm": c.plan.length,
            "diameter_mm": c.plan.diameter,
            "entry": c.plan.entry,
            "direction": c.plan.direction,
            "entry_snap_status": c.entry_snap_status,
            "entry_snap_shift_mm": c.entry_snap_shift_mm,
        })
        result.update(entry_displacement(self.baseline.candidate.plan, c.plan, self.frame))
        return result


def entry_displacement(baseline: ScrewPlan, plan: ScrewPlan, frame: AnatomicalFrame = None) -> dict:
    """
    最终进钉点相对经验进钉点的实际位移（含支点转动与贴合骨面带来的移动）

    分解到内侧、头侧和基准钉轴方向；未提供 frame 时只给出总位移。
    """
    delta = np.asarray(plan.entry) - np.asarray(baseline.entry)
    result = {"entry_displacement_mm": float(np.linalg.norm(delta))}
    if frame is not None:
        result["entry_displacement_medial_mm"] = float(np.dot(delta, frame.medial))
        result["entry_displacement_cranial_mm"] = float(np.dot(delta, frame.cranial))
        result["entry_displacement_axial_mm"] = float(np.dot(delta, baseline.direction))
    return result


def candidate_features(image, candidate: Candidate, ctx: ConstraintContext,
                       calibration=None, sampling_config: TrajectorySamplingConfig = None) -> dict:
    """计算候选方案的风险模型输入特征（钉道骨密度特征 + 轨迹匹配特征）。"""
    bounds = ctx.pedicle.segment_bounds_for(candidate.plan) if ctx.pedicle is not None else None
    features = compute_trajectory_features(
        image, candidate.plan, bone_mask=ctx.bone_mask, bone_mesh=ctx.bone_mesh,
        calibration=calibration, config=sampling_config, segment_bounds_mm=bounds,
    ).to_flat_dict()
    if ctx.pedicle is not None:
        for key, value in ctx.pedicle.trajectory_match(candidate.plan).items():
            features[f"match_{key}"] = value
    return features


def optimize_screw_plan(image, baseline: ScrewPlan, frame: AnatomicalFrame, risk_fn,
                        bone_mask=None, bone_mesh=None, pedicle=None, constraints: list = None,
                        grid: CandidateGridConfig = None, lengths: tuple = None,
                        diameters: tuple = None, calibration=None,
                        sampling_config: TrajectorySamplingConfig = None,
                        pivot_point=None) -> OptimizationResult:
    """
    在安全约束下搜索预测松动风险最低的置钉方案

    Parameters
    ----------
    image : sitk.Image
        CT图像（HU）
    baseline : ScrewPlan
        医生经验置钉方案（初始解与对照基准）
    frame : AnatomicalFrame
        该侧螺钉的内侧/头侧方向
    risk_fn : callable
        risk_fn(features: dict) -> float，值越小越好
    pedicle : PedicleGeometry, optional
        detect_pedicle_isthmus 的结果；提供时用于分段边界、匹配特征和峡部约束
    constraints : list, optional
        约束函数 f(candidate, features, ctx) -> None 或违规说明字符串；
        默认 default_constraints()
    grid : CandidateGridConfig, optional
        候选网格，含支点（PIVOT）与进钉点贴合骨面（SNAP_ENTRY）设置
    pivot_point : tuple, optional
        PIVOT="isthmus" 时显式指定支点（默认取基准钉轴与峡部截面的交点）
    """
    if constraints is None:
        constraints = default_constraints()
    ctx = ConstraintContext(image=image, bone_mask=bone_mask, bone_mesh=bone_mesh, pedicle=pedicle)

    evaluations = []
    candidates = generate_candidates(baseline, frame, grid, lengths, diameters, pedicle=pedicle,
                                     pivot_point=pivot_point, bone_mask=bone_mask, bone_mesh=bone_mesh)
    for candidate in candidates:
        features = candidate_features(image, candidate, ctx, calibration, sampling_config)
        violations = [msg for msg in (check(candidate, features, ctx) for check in constraints) if msg]
        feasible = not violations
        risk = float(risk_fn(features)) if (feasible or candidate.is_baseline) else float("nan")
        evaluations.append(CandidateEvaluation(candidate, feasible, violations, features, risk))

    feasible = [e for e in evaluations if e.feasible and np.isfinite(e.risk)]
    best = min(feasible, key=lambda e: e.risk) if feasible else None
    return OptimizationResult(baseline=evaluations[0], best=best, evaluations=evaluations, frame=frame)
