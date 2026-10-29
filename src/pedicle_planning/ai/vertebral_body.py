"""
vertebral_body.py
------------------------------------
椎体整体松质骨骨密度（患者固有特征）

在椎体解剖坐标系下：
1. 取椎体掩膜中前后方向最前部 BODY_FRACTION 的部分作为椎体（排除椎弓根、
   椎板与棘突），与 vertebra_frame 中头尾轴估计所用的椎体区域一致；
2. 向内腐蚀 CORTICAL_MARGIN_MM，去除皮质壳与终板；
3. 统计整个松质骨区域（trabecular），以及椎体中部高度、厚度为
   MIDBODY_SLAB_MM 的层（midbody，近似临床上在椎体中部层面放置的松质骨 ROI）。

结果不随置钉方案变化，可通过 optimize_screw_plan(patient_features=...) 与
钉道特征一起输入风险模型。
"""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from .trajectory import LinearCalibration, RegionStats, _region_stats, sample_image_trilinear
from .vertebra_frame import VertebraFrame, estimate_vertebra_frame


@dataclass
class VertebralBodyConfig:
    """椎体整体骨密度配置参数（长度单位为毫米）。"""

    BODY_FRACTION: float = 0.4
    """前后方向上最前部的该比例视为椎体。"""

    CORTICAL_MARGIN_MM: float = 3.0
    """从椎体表面向内腐蚀的距离，用于排除皮质壳与终板。"""

    MIDBODY_SLAB_MM: float = 10.0
    """椎体中部层的厚度（沿头尾轴）。"""

    HEIGHT_PERCENTILES: tuple = (1.0, 99.0)
    """估计椎体上下界（高度与中点）时使用的百分位。"""

    LOW_DENSITY_HU: float = 110.0
    CORTICAL_HU: float = 800.0

    def validate(self) -> None:
        if not 0.0 < self.BODY_FRACTION <= 1.0:
            raise ValueError(f"BODY_FRACTION ({self.BODY_FRACTION}) must be in (0, 1]")
        if self.CORTICAL_MARGIN_MM < 0:
            raise ValueError(f"CORTICAL_MARGIN_MM ({self.CORTICAL_MARGIN_MM}) must be non-negative")
        if not self.MIDBODY_SLAB_MM > 0:
            raise ValueError(f"MIDBODY_SLAB_MM ({self.MIDBODY_SLAB_MM}) must be positive")
        lo, hi = self.HEIGHT_PERCENTILES
        if not 0.0 <= lo < hi <= 100.0:
            raise ValueError(f"HEIGHT_PERCENTILES ({self.HEIGHT_PERCENTILES}) must satisfy 0 <= lo < hi <= 100")


@dataclass
class VertebralBodyBMD:
    """
    椎体整体骨密度

    trabecular: 腐蚀后的整个椎体松质骨；midbody: 其中椎体中部层；
    body_height_mm: 椎体沿头尾轴的高度（按 HEIGHT_PERCENTILES）。
    """

    unit: str
    frame: VertebraFrame
    trabecular: RegionStats
    midbody: RegionStats
    body_volume_mm3: float
    trabecular_volume_mm3: float
    body_height_mm: float

    def to_flat_dict(self, prefix: str = "vertebra_") -> dict:
        flat = {}
        for region in ("trabecular", "midbody"):
            for key, value in getattr(self, region).as_dict().items():
                flat[f"{prefix}{region}_{key}"] = value
        flat[f"{prefix}body_volume_mm3"] = self.body_volume_mm3
        flat[f"{prefix}trabecular_volume_mm3"] = self.trabecular_volume_mm3
        flat[f"{prefix}body_height_mm"] = self.body_height_mm
        flat[f"{prefix}frame_symmetry_score"] = self.frame.symmetry_score
        return flat


def _voxel_points(mask_image: sitk.Image, kji: np.ndarray) -> np.ndarray:
    origin = np.asarray(mask_image.GetOrigin(), dtype=np.float64)
    matrix = (np.asarray(mask_image.GetDirection(), dtype=np.float64).reshape(3, 3)
              * np.asarray(mask_image.GetSpacing()))
    return origin + kji[:, ::-1].astype(np.float64) @ matrix.T


def compute_vertebral_body_bmd(image: sitk.Image, vertebra_mask: sitk.Image,
                               vertebra_frame: VertebraFrame = None,
                               calibration: LinearCalibration = None,
                               config: VertebralBodyConfig = None) -> VertebralBodyBMD:
    """
    计算单个椎体的整体松质骨骨密度

    Parameters
    ----------
    image : sitk.Image
        CT图像（HU）；可与掩膜几何不同，按物理坐标插值采样
    vertebra_mask : sitk.Image
        单个椎体（含后部结构）二值掩膜
    vertebra_frame : VertebraFrame, optional
        椎体坐标系；未提供时由 estimate_vertebra_frame 估计
    calibration : LinearCalibration, optional
        HU到BMD的线性校准
    """
    if config is None:
        config = VertebralBodyConfig()
    config.validate()
    if vertebra_frame is None:
        vertebra_frame = estimate_vertebra_frame(vertebra_mask)

    mask_array = sitk.GetArrayViewFromImage(vertebra_mask)
    kji = np.argwhere(mask_array != 0)
    if kji.shape[0] == 0:
        raise ValueError("vertebra_mask is empty")
    points = _voxel_points(vertebra_mask, kji)
    ap = (points - vertebra_frame.origin) @ vertebra_frame.anterior
    cut = ap.max() - config.BODY_FRACTION * (ap.max() - ap.min())
    body_kji = kji[ap >= cut]

    body_array = np.zeros(mask_array.shape, dtype=np.uint8)
    body_array[tuple(body_kji.T)] = 1
    spacing = np.asarray(vertebra_mask.GetSpacing(), dtype=np.float64)
    voxel_volume = float(np.prod(spacing))

    if config.CORTICAL_MARGIN_MM > 0:
        body_image = sitk.GetImageFromArray(body_array)
        body_image.CopyInformation(vertebra_mask)
        radius = [max(1, int(round(config.CORTICAL_MARGIN_MM / sp))) for sp in spacing]
        trabecular_array = sitk.GetArrayFromImage(sitk.BinaryErode(body_image, radius))
    else:
        trabecular_array = body_array
    trab_kji = np.argwhere(trabecular_array != 0)

    body_points = _voxel_points(vertebra_mask, body_kji)
    height = (body_points - vertebra_frame.origin) @ vertebra_frame.cranial
    h_lo, h_hi = np.percentile(height, config.HEIGHT_PERCENTILES)
    mid_height = 0.5 * (h_lo + h_hi)

    trab_points = _voxel_points(vertebra_mask, trab_kji) if trab_kji.shape[0] else np.zeros((0, 3))
    trab_hu = sample_image_trilinear(image, trab_points) if trab_points.shape[0] else np.zeros(0)
    trab_height = (trab_points - vertebra_frame.origin) @ vertebra_frame.cranial
    in_slab = np.abs(trab_height - mid_height) <= 0.5 * config.MIDBODY_SLAB_MM + 1e-9

    values = calibration.apply(trab_hu) if calibration is not None else trab_hu
    all_bone = np.ones(trab_hu.shape[0], dtype=bool)

    return VertebralBodyBMD(
        unit=calibration.unit if calibration is not None else "HU",
        frame=vertebra_frame,
        trabecular=_region_stats(values, all_bone, trab_hu, config),
        midbody=_region_stats(values[in_slab], all_bone[in_slab], trab_hu[in_slab], config),
        body_volume_mm3=float(body_kji.shape[0] * voxel_volume),
        trabecular_volume_mm3=float(trab_kji.shape[0] * voxel_volume),
        body_height_mm=float(h_hi - h_lo),
    )
