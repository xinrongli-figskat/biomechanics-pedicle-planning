"""
vertebra_frame.py
------------------------------------
由单个椎体分割自动建立椎体解剖坐标系

1. 左右轴：椎体近似左右镜像对称。以图像病人方向为初值，在横断面旋转与冠状面
   倾斜两个角度上搜索使掩膜镜像重合度最高的正中矢状面，其法向即左右轴。
   该步骤可处理椎体轴向旋转与侧弯。
2. 头尾轴：取正中矢状面内、椎体前部（椎体本身，排除椎弓与棘突）的采样点，
   在头尾初值附近搜索投影宽度最小的方向（即椎体高度方向），避免棘突倾斜等
   后部结构的干扰。该步骤可处理矢状面倾斜。
3. 前后轴：由头尾轴与左右轴叉乘得到。

图像病人方向（默认 SimpleITK/DICOM 的 LPS 物理坐标：左 +x、后 +y、上 +z）
只用于确定各轴的正负号与搜索初值，不直接作为结果。

VertebraFrame.anatomical_frame_for(entry) 根据进钉点位于正中矢状面的哪一侧，
给出该侧螺钉的 AnatomicalFrame（medial 指向中线）。
"""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from .candidates import AnatomicalFrame, _rotate
from .trajectory import sample_mask_nearest


LPS_CRANIAL = (0.0, 0.0, 1.0)
LPS_ANTERIOR = (0.0, -1.0, 0.0)


@dataclass
class VertebraFrameConfig:
    """椎体坐标系估计配置参数。"""

    MAX_SAMPLE_POINTS: int = 20000
    """参与计算的掩膜体素上限（超过时以固定随机种子下采样，结果可复现）。"""

    SYMMETRY_SEARCH_DEG: float = 40.0
    """对称面搜索范围（相对初值，横断面旋转与冠状面倾斜各 ±该值）。"""

    SYMMETRY_STEPS_DEG: tuple = (4.0, 1.0, 0.25)
    """对称面由粗到细的搜索步长。"""

    CRANIAL_SEARCH_DEG: float = 30.0
    """头尾轴在正中矢状面内的搜索范围（相对初值 ±该值）。"""

    CRANIAL_STEP_DEG: float = 0.25

    BODY_FRACTION: float = 0.4
    """前后方向上最前部的该比例视为椎体，用于头尾轴估计与原点。"""

    WIDTH_PERCENTILES: tuple = (1.0, 99.0)
    """投影宽度采用的百分位区间，降低分割毛刺与骨赘的影响。"""

    ITERATIONS: int = 2
    """头尾轴/椎体区域交替估计次数。"""

    MIN_SYMMETRY_SCORE: float = 0.85
    """
    对称重合度低于该值时 reliable=False，建议检查分割或退回图像坐标轴。
    启发式阈值：完整合成椎体约 0.97–1.0，缺失一侧后部结构时约 0.84，
    需在真实分割上校准。
    """

    def validate(self) -> None:
        if self.MAX_SAMPLE_POINTS < 100:
            raise ValueError(f"MAX_SAMPLE_POINTS ({self.MAX_SAMPLE_POINTS}) must be at least 100")
        for name in ("SYMMETRY_SEARCH_DEG", "CRANIAL_SEARCH_DEG", "CRANIAL_STEP_DEG"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} ({getattr(self, name)}) must be positive")
        if not self.SYMMETRY_STEPS_DEG or any(s <= 0 for s in self.SYMMETRY_STEPS_DEG):
            raise ValueError("SYMMETRY_STEPS_DEG must be non-empty and positive")
        if not 0.0 < self.BODY_FRACTION <= 1.0:
            raise ValueError(f"BODY_FRACTION ({self.BODY_FRACTION}) must be in (0, 1]")
        lo, hi = self.WIDTH_PERCENTILES
        if not 0.0 <= lo < hi <= 100.0:
            raise ValueError(f"WIDTH_PERCENTILES ({self.WIDTH_PERCENTILES}) must satisfy 0 <= lo < hi <= 100")
        if self.ITERATIONS < 1:
            raise ValueError(f"ITERATIONS ({self.ITERATIONS}) must be at least 1")


@dataclass
class VertebraFrame:
    """
    椎体解剖坐标系（右手系：left = cranial × anterior）

    origin 为椎体部分质心；symmetry_score 为最优对称面的镜像重合度（0–1）；
    method 为 "symmetry"（由分割估计）或 "image_axes"（图像病人方向）。
    """

    origin: np.ndarray
    left: np.ndarray
    anterior: np.ndarray
    cranial: np.ndarray
    symmetry_score: float
    method: str
    reliable: bool

    def side_of(self, point) -> str:
        """点位于正中矢状面的哪一侧："left" 或 "right"。"""
        offset = float(np.dot(np.asarray(point, dtype=np.float64) - self.origin, self.left))
        return "left" if offset >= 0 else "right"

    def anatomical_frame_for(self, entry) -> AnatomicalFrame:
        """按进钉点所在侧返回该螺钉的 AnatomicalFrame（medial 指向中线）。"""
        medial = -self.left if self.side_of(entry) == "left" else self.left
        return AnatomicalFrame(medial=tuple(medial), cranial=tuple(self.cranial))

    def to_local(self, point) -> np.ndarray:
        """物理坐标 → 椎体坐标 (left, anterior, cranial)。"""
        d = np.asarray(point, dtype=np.float64) - self.origin
        return np.array([np.dot(d, self.left), np.dot(d, self.anterior), np.dot(d, self.cranial)])

    def angles_to(self, cranial=LPS_CRANIAL, anterior=LPS_ANTERIOR) -> dict:
        """
        相对给定（如图像）方向的偏差角（度）

        axial_rotation_deg: 前向轴绕头尾轴转向病人右侧为正；
        coronal_tilt_deg: 头尾轴向病人右侧倾斜为正；
        sagittal_tilt_deg: 头尾轴向前倾斜为正。
        """
        ref = _orthonormal_axes(cranial, anterior)
        ref_left, ref_anterior, ref_cranial = ref
        return {
            "axial_rotation_deg": float(np.degrees(np.arctan2(np.dot(self.anterior, -ref_left),
                                                              np.dot(self.anterior, ref_anterior)))),
            "coronal_tilt_deg": float(np.degrees(np.arctan2(np.dot(self.cranial, -ref_left),
                                                            np.dot(self.cranial, ref_cranial)))),
            "sagittal_tilt_deg": float(np.degrees(np.arctan2(np.dot(self.cranial, ref_anterior),
                                                             np.dot(self.cranial, ref_cranial)))),
        }


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("direction must be non-zero")
    return v / n


def _orthonormal_axes(cranial, anterior) -> tuple:
    """返回 (left, anterior, cranial)，anterior 正交化到 cranial 上。"""
    c = _unit(cranial)
    a = np.asarray(anterior, dtype=np.float64)
    a = a - np.dot(a, c) * c
    a = _unit(a)
    return np.cross(c, a), a, c


def mask_points(bone_mask: sitk.Image, max_points: int = None, seed: int = 0) -> np.ndarray:
    """掩膜非零体素中心的物理坐标；超过 max_points 时以固定种子下采样。"""
    array = sitk.GetArrayViewFromImage(bone_mask)
    kji = np.argwhere(array != 0)
    if kji.shape[0] == 0:
        raise ValueError("bone_mask is empty")
    if max_points is not None and kji.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        kji = kji[rng.choice(kji.shape[0], max_points, replace=False)]
    ijk = kji[:, ::-1].astype(np.float64)
    origin = np.asarray(bone_mask.GetOrigin(), dtype=np.float64)
    matrix = np.asarray(bone_mask.GetDirection(), dtype=np.float64).reshape(3, 3) * np.asarray(bone_mask.GetSpacing())
    return origin + ijk @ matrix.T


def symmetry_score(bone_mask: sitk.Image, points: np.ndarray, plane_point, normal) -> float:
    """以 (plane_point, normal) 为镜面反射采样点后仍落在掩膜内的比例。"""
    n = _unit(normal)
    d = (points - np.asarray(plane_point)) @ n
    reflected = points - 2.0 * np.outer(d, n)
    return float(np.mean(sample_mask_nearest(bone_mask, reflected)))


def _search_symmetry_plane(bone_mask, points, centroid, left0, anterior0, cranial0, config) -> tuple:
    def normal_for(theta, psi):
        n = _rotate(left0, cranial0, np.radians(theta))
        return _rotate(n, anterior0, np.radians(psi))

    best_theta, best_psi = 0.0, 0.0
    span = config.SYMMETRY_SEARCH_DEG
    best_score = -1.0
    for step in config.SYMMETRY_STEPS_DEG:
        thetas = np.arange(best_theta - span, best_theta + span + 1e-9, step)
        psis = np.arange(best_psi - span, best_psi + span + 1e-9, step)
        scores = np.array([[symmetry_score(bone_mask, points, centroid, normal_for(t, p)) for p in psis]
                           for t in thetas])
        ti, pi = np.unravel_index(int(np.argmax(scores)), scores.shape)
        best_theta, best_psi, best_score = float(thetas[ti]), float(psis[pi]), float(scores[ti, pi])
        span = step
    return normal_for(best_theta, best_psi), best_score


def _min_width_direction(points2d: np.ndarray, center_angle: float, span_deg: float,
                         step_deg: float, percentiles: tuple) -> float:
    angles = np.radians(np.arange(-span_deg, span_deg + 1e-9, step_deg)) + center_angle
    lo, hi = percentiles
    widths = []
    for beta in angles:
        proj = points2d[:, 0] * np.cos(beta) + points2d[:, 1] * np.sin(beta)
        p_lo, p_hi = np.percentile(proj, [lo, hi])
        widths.append(p_hi - p_lo)
    return float(angles[int(np.argmin(widths))])


def estimate_vertebra_frame(bone_mask: sitk.Image, cranial_hint=LPS_CRANIAL,
                            anterior_hint=LPS_ANTERIOR,
                            config: VertebraFrameConfig = None) -> VertebraFrame:
    """
    由单个椎体（含椎体与后部结构）的二值掩膜估计解剖坐标系

    Parameters
    ----------
    bone_mask : sitk.Image
        单个椎体掩膜（非零为骨）
    cranial_hint, anterior_hint : tuple
        图像中病人头侧、前侧的大致方向（物理坐标），只用于符号与搜索初值；
        默认按 LPS 约定
    """
    if config is None:
        config = VertebraFrameConfig()
    config.validate()

    points = mask_points(bone_mask, config.MAX_SAMPLE_POINTS)
    centroid = points.mean(axis=0)
    left0, anterior0, cranial0 = _orthonormal_axes(cranial_hint, anterior_hint)

    left, score = _search_symmetry_plane(bone_mask, points, centroid, left0, anterior0, cranial0, config)
    if np.dot(left, left0) < 0:
        left = -left
    # midsagittal plane passes through the centroid of the symmetric shape
    plane_point = centroid

    cranial = cranial0 - np.dot(cranial0, left) * left
    cranial = _unit(cranial)
    body_points = points
    for _ in range(config.ITERATIONS):
        anterior = np.cross(left, cranial)
        ap = (points - plane_point) @ anterior
        cut = ap.max() - config.BODY_FRACTION * (ap.max() - ap.min())
        body_points = points[ap >= cut]
        rel = body_points - body_points.mean(axis=0)
        coords = np.stack([rel @ cranial, rel @ anterior], axis=1)
        beta = _min_width_direction(coords, 0.0, config.CRANIAL_SEARCH_DEG,
                                    config.CRANIAL_STEP_DEG, config.WIDTH_PERCENTILES)
        cranial = _unit(np.cos(beta) * cranial + np.sin(beta) * anterior)

    if np.dot(cranial, cranial0) < 0:
        cranial = -cranial
    anterior = np.cross(left, cranial)
    if np.dot(anterior, anterior0) < 0:
        anterior = -anterior
    # right-handed: left = cranial x anterior (signs follow the hints)
    left = np.cross(cranial, anterior)
    body_centroid = body_points.mean(axis=0)
    # place the origin on the midsagittal plane
    origin = body_centroid - np.dot(body_centroid - plane_point, left) * left

    return VertebraFrame(
        origin=origin,
        left=left,
        anterior=anterior,
        cranial=cranial,
        symmetry_score=score,
        method="symmetry",
        reliable=bool(score >= config.MIN_SYMMETRY_SCORE),
    )


def image_axes_frame(bone_mask: sitk.Image, cranial_hint=LPS_CRANIAL,
                     anterior_hint=LPS_ANTERIOR) -> VertebraFrame:
    """退回方案：直接用图像病人方向，原点取掩膜质心。"""
    points = mask_points(bone_mask, VertebraFrameConfig().MAX_SAMPLE_POINTS)
    left, anterior, cranial = _orthonormal_axes(cranial_hint, anterior_hint)
    centroid = points.mean(axis=0)
    return VertebraFrame(
        origin=centroid,
        left=left,
        anterior=anterior,
        cranial=cranial,
        symmetry_score=symmetry_score(bone_mask, points, centroid, left),
        method="image_axes",
        reliable=True,
    )
