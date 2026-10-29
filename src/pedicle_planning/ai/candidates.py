"""
candidates.py
------------------------------------
候选置钉方案生成、安全约束过滤与风险最小化选择

以经验置钉方案为初始解，在解剖坐标系下离散调整进钉点（内外/头尾平移）、
内倾角、头尾倾角、螺钉长度与直径，生成有限候选集；候选方案先经安全硬约束
过滤，再逐一重新计算钉道特征并输入风险评估函数，选取预测风险最低的方案。

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
    compute_trajectory_features,
    points_inside_mesh,
    sample_mask_nearest,
)


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
    """

    MEDIAL_OFFSETS_MM: tuple = (-2.0, -1.0, 0.0, 1.0, 2.0)
    CRANIAL_OFFSETS_MM: tuple = (0.0,)
    MEDIAL_ANGLES_DEG: tuple = (-5.0, 0.0, 5.0)
    CRANIAL_ANGLES_DEG: tuple = (0.0,)
    LENGTH_OFFSETS_MM: tuple = (-5.0, 0.0, 5.0)
    DIAMETER_OFFSETS_MM: tuple = (0.0, 0.5)

    def validate(self) -> None:
        for name in ("MEDIAL_OFFSETS_MM", "CRANIAL_OFFSETS_MM", "MEDIAL_ANGLES_DEG",
                     "CRANIAL_ANGLES_DEG", "LENGTH_OFFSETS_MM", "DIAMETER_OFFSETS_MM"):
            values = getattr(self, name)
            if len(values) == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a non-empty tuple of finite values")
        for name in ("MEDIAL_ANGLES_DEG", "CRANIAL_ANGLES_DEG"):
            if np.any(np.abs(getattr(self, name)) >= 90.0):
                raise ValueError(f"{name} must lie strictly within (-90, 90) degrees")


@dataclass(frozen=True)
class Candidate:
    """候选方案及其相对经验方案的调整量。"""

    plan: ScrewPlan
    medial_offset_mm: float = 0.0
    cranial_offset_mm: float = 0.0
    medial_angle_deg: float = 0.0
    cranial_angle_deg: float = 0.0
    is_baseline: bool = False


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
                diameter: float = None) -> ScrewPlan:
    """
    对经验方案做一次调整

    内倾角绕头尾轴转动（横断面内），头尾倾角绕与钉轴、头尾轴都垂直的轴转动
    （矢状面内）；进钉点沿 medial/cranial 方向平移，角度以平移后的进钉点为支点。
    """
    medial = np.asarray(frame.medial)
    cranial = np.asarray(frame.cranial)
    direction = np.asarray(baseline.direction)

    direction = _rotate_toward(direction, cranial, medial, medial_angle_deg)
    sagittal_pivot = np.cross(cranial, direction)
    direction = _rotate_toward(direction, sagittal_pivot, cranial, cranial_angle_deg)

    entry = np.asarray(baseline.entry) + medial_offset_mm * medial + cranial_offset_mm * cranial
    return ScrewPlan(
        tuple(entry),
        tuple(direction),
        baseline.length if length is None else length,
        baseline.diameter if diameter is None else diameter,
    )


def generate_candidates(baseline: ScrewPlan, frame: AnatomicalFrame,
                        config: CandidateGridConfig = None, lengths: tuple = None,
                        diameters: tuple = None) -> list:
    """
    生成候选方案列表，第一个元素始终为经验方案本身（is_baseline=True）

    Parameters
    ----------
    lengths, diameters : tuple, optional
        临床可选规格（绝对值，毫米）；提供时替代 LENGTH/DIAMETER_OFFSETS_MM
    """
    if config is None:
        config = CandidateGridConfig()
    config.validate()

    if lengths is None:
        lengths = tuple(baseline.length + d for d in config.LENGTH_OFFSETS_MM)
    if diameters is None:
        diameters = tuple(baseline.diameter + d for d in config.DIAMETER_OFFSETS_MM)
    lengths = tuple(sorted({float(x) for x in lengths if x > 0}))
    diameters = tuple(sorted({float(x) for x in diameters if x > 0}))
    if not lengths or not diameters:
        raise ValueError("at least one positive length and diameter are required")

    candidates = [Candidate(baseline, is_baseline=True)]
    for dm in config.MEDIAL_OFFSETS_MM:
        for dc in config.CRANIAL_OFFSETS_MM:
            for am in config.MEDIAL_ANGLES_DEG:
                for ac in config.CRANIAL_ANGLES_DEG:
                    for length in lengths:
                        for diameter in diameters:
                            if (dm == 0 and dc == 0 and am == 0 and ac == 0
                                    and length == baseline.length and diameter == baseline.diameter):
                                continue
                            plan = adjust_plan(baseline, frame, dm, dc, am, ac, length, diameter)
                            candidates.append(Candidate(plan, float(dm), float(dc), float(am), float(ac)))
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


def _sphere_directions(n: int) -> np.ndarray:
    """近似均匀的单位方向（Fibonacci 球面）。"""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def default_constraints() -> list:
    """内置约束；缺少所需数据（峡部几何/骨掩膜）的约束自动跳过。"""
    return [diameter_within_isthmus(1.0), isthmus_wall_clearance(0.0), tip_inside_bone(2.0)]


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
            "medial_offset_mm": c.medial_offset_mm,
            "cranial_offset_mm": c.cranial_offset_mm,
            "medial_angle_deg": c.medial_angle_deg,
            "cranial_angle_deg": c.cranial_angle_deg,
            "length_mm": c.plan.length,
            "diameter_mm": c.plan.diameter,
            "entry": c.plan.entry,
            "direction": c.plan.direction,
        })
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
                        sampling_config: TrajectorySamplingConfig = None) -> OptimizationResult:
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
    """
    if constraints is None:
        constraints = default_constraints()
    ctx = ConstraintContext(image=image, bone_mask=bone_mask, bone_mesh=bone_mesh, pedicle=pedicle)

    evaluations = []
    for candidate in generate_candidates(baseline, frame, grid, lengths, diameters):
        features = candidate_features(image, candidate, ctx, calibration, sampling_config)
        violations = [msg for msg in (check(candidate, features, ctx) for check in constraints) if msg]
        feasible = not violations
        risk = float(risk_fn(features)) if (feasible or candidate.is_baseline) else float("nan")
        evaluations.append(CandidateEvaluation(candidate, feasible, violations, features, risk))

    feasible = [e for e in evaluations if e.feasible and np.isfinite(e.risk)]
    best = min(feasible, key=lambda e: e.risk) if feasible else None
    return OptimizationResult(baseline=evaluations[0], best=best, evaluations=evaluations)
