"""
utils.py
------------------------------------
AI辅助置钉点预测模块 - 工具函数

提供CT预处理、坐标转换、HU归一化、邻域提取等功能
"""

import numpy as np
import SimpleITK as sitk

from .config import config


def get_ct_array(sitk_image: sitk.Image) -> np.ndarray:
    """
    将SimpleITK图像转换为numpy数组

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像

    Returns
    -------
    np.ndarray
        CT图像的numpy数组表示，形状为(z, y, x)
    """
    return sitk.GetArrayFromImage(sitk_image)


def get_spacing(sitk_image: sitk.Image) -> tuple:
    """
    获取CT图像的像素间距

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像

    Returns
    -------
    tuple
        三个方向的像素间距 (spacing_x, spacing_y, spacing_z)
    """
    return tuple(sitk_image.GetSpacing())


def get_origin(sitk_image: sitk.Image) -> tuple:
    """
    获取CT图像的原点坐标

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像

    Returns
    -------
    tuple
        图像原点坐标 (origin_x, origin_y, origin_z)
    """
    return tuple(sitk_image.GetOrigin())


def get_size(sitk_image: sitk.Image) -> tuple:
    """
    获取CT图像的尺寸

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像

    Returns
    -------
    tuple
        三个方向的图像尺寸 (size_x, size_y, size_z)
    """
    return tuple(sitk_image.GetSize())


def get_direction(sitk_image: sitk.Image) -> tuple:
    """
    获取CT图像的方向矩阵

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像

    Returns
    -------
    tuple
        3x3方向矩阵，以9元素元组形式返回
    """
    return tuple(sitk_image.GetDirection())


def world_to_voxel(world_coords: tuple, origin: tuple, spacing: tuple) -> tuple:
    """
    将世界坐标转换为体素坐标

    转换公式：voxel = (world - origin) / spacing

    Parameters
    ----------
    world_coords : tuple
        世界坐标 (x, y, z)，单位为毫米
    origin : tuple
        图像原点坐标 (origin_x, origin_y, origin_z)
    spacing : tuple
        像素间距 (spacing_x, spacing_y, spacing_z)，单位为毫米/像素

    Returns
    -------
    tuple
        体素坐标 (i, j, k)，整数类型，对应图像数组的(x, y, z)索引
    """
    voxel = [
        int((world_coords[0] - origin[0]) / spacing[0]),
        int((world_coords[1] - origin[1]) / spacing[1]),
        int((world_coords[2] - origin[2]) / spacing[2])
    ]
    return tuple(voxel)


def voxel_to_world(voxel_coords: tuple, origin: tuple, spacing: tuple) -> tuple:
    """
    将体素坐标转换为世界坐标

    转换公式：world = voxel * spacing + origin

    Parameters
    ----------
    voxel_coords : tuple
        体素坐标 (i, j, k)，对应图像数组的(x, y, z)索引
    origin : tuple
        图像原点坐标
    spacing : tuple
        像素间距

    Returns
    -------
    tuple
        世界坐标 (x, y, z)，单位为毫米
    """
    world = [
        voxel_coords[0] * spacing[0] + origin[0],
        voxel_coords[1] * spacing[1] + origin[1],
        voxel_coords[2] * spacing[2] + origin[2]
    ]
    return tuple(world)


def clip_hu(image_array: np.ndarray, hu_min: int = None, hu_max: int = None) -> np.ndarray:
    """
    HU值截断

    将图像数组中超出[hu_min, hu_max]范围的HU值裁剪到边界值。

    Parameters
    ----------
    image_array : np.ndarray
        CT图像数组，形状为(z, y, x)
    hu_min : int, optional
        HU值下限，低于此值的像素将被设置为hu_min。
        默认为config.ct_preprocessing.HU_MIN
    hu_max : int, optional
        HU值上限，高于此值的像素将被设置为hu_max。
        默认为config.ct_preprocessing.HU_MAX

    Returns
    -------
    np.ndarray
        截断后的图像数组，数据类型与输入一致
    """
    if hu_min is None:
        hu_min = config.ct_preprocessing.HU_MIN
    if hu_max is None:
        hu_max = config.ct_preprocessing.HU_MAX

    return np.clip(image_array, hu_min, hu_max)


def normalize_hu(image_array: np.ndarray, hu_min: int = None, hu_max: int = None,
                 norm_min: float = None, norm_max: float = None) -> np.ndarray:
    """
    HU值归一化

    先进行HU值截断，然后将截断后的值线性映射到[norm_min, norm_max]范围。

    Parameters
    ----------
    image_array : np.ndarray
        CT图像数组
    hu_min : int, optional
        HU值截断下限
    hu_max : int, optional
        HU值截断上限
    norm_min : float, optional
        归一化下限，默认为config.ct_preprocessing.NORMALIZE_MIN
    norm_max : float, optional
        归一化上限，默认为config.ct_preprocessing.NORMALIZE_MAX

    Returns
    -------
    np.ndarray
        归一化后的图像数组，值范围在[norm_min, norm_max]之间，数据类型为float
    """
    if hu_min is None:
        hu_min = config.ct_preprocessing.HU_MIN
    if hu_max is None:
        hu_max = config.ct_preprocessing.HU_MAX
    if norm_min is None:
        norm_min = config.ct_preprocessing.NORMALIZE_MIN
    if norm_max is None:
        norm_max = config.ct_preprocessing.NORMALIZE_MAX

    clipped = clip_hu(image_array, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min) * (norm_max - norm_min) + norm_min
    return normalized


def extract_neighborhood(image_array: np.ndarray, voxel_coords: tuple,
                         radius: int) -> np.ndarray:
    """
    提取指定体素坐标周围的邻域

    从CT图像数组中提取以指定体素为中心的立方体邻域。
    如果邻域超出图像边界，使用零填充。

    Parameters
    ----------
    image_array : np.ndarray
        CT图像数组，形状为(z, y, x)
    voxel_coords : tuple
        中心体素坐标 (i, j, k)，对应图像数组的(x, y, z)索引
    radius : int
        邻域半径（体素数），邻域大小为(2*radius+1)^3

    Returns
    -------
    np.ndarray
        邻域子数组，如果坐标超出边界则用零填充，形状为(2*radius+1, 2*radius+1, 2*radius+1)
    """
    x, y, z = voxel_coords
    k, j, i = z, y, x

    z_size, y_size, x_size = image_array.shape

    z_min = max(0, k - radius)
    z_max = min(z_size, k + radius + 1)
    y_min = max(0, j - radius)
    y_max = min(y_size, j + radius + 1)
    x_min = max(0, i - radius)
    x_max = min(x_size, i + radius + 1)

    neighborhood = image_array[z_min:z_max, y_min:y_max, x_min:x_max]

    full_size = 2 * radius + 1
    if neighborhood.shape != (full_size, full_size, full_size):
        padded = np.zeros((full_size, full_size, full_size), dtype=image_array.dtype)
        dz = z_max - z_min
        dy = y_max - y_min
        dx = x_max - x_min
        pad_z = (full_size - dz) // 2
        pad_y = (full_size - dy) // 2
        pad_x = (full_size - dx) // 2
        padded[pad_z:pad_z + dz, pad_y:pad_y + dy, pad_x:pad_x + dx] = neighborhood
        return padded

    return neighborhood


def extract_neighborhood_world(sitk_image: sitk.Image, world_coords: tuple,
                               radius_mm: float) -> np.ndarray:
    """
    在世界坐标系下提取邻域

    根据世界坐标和毫米级半径，自动计算对应的体素邻域并提取。

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像
    world_coords : tuple
        世界坐标 (x, y, z)，单位为毫米
    radius_mm : float
        邻域半径（毫米）

    Returns
    -------
    np.ndarray
        邻域子数组，形状为(2*radius_voxel+1, 2*radius_voxel+1, 2*radius_voxel+1)
    """
    origin = get_origin(sitk_image)
    spacing = get_spacing(sitk_image)

    voxel_coords = world_to_voxel(world_coords, origin, spacing)

    radius_voxel = max(1, int(radius_mm / min(spacing)))

    image_array = get_ct_array(sitk_image)

    return extract_neighborhood(image_array, voxel_coords, radius_voxel)


def resample_image(sitk_image: sitk.Image, target_spacing: tuple = None) -> sitk.Image:
    """
    图像重采样

    将图像重采样到目标分辨率，保持物理空间范围不变。

    Parameters
    ----------
    sitk_image : sitk.Image
        原始SimpleITK图像
    target_spacing : tuple, optional
        目标像素间距 (spacing_x, spacing_y, spacing_z)，单位为毫米/像素。
        默认为config.ct_preprocessing.TARGET_SPACING

    Returns
    -------
    sitk.Image
        重采样后的图像
    """
    if target_spacing is None:
        target_spacing = config.ct_preprocessing.TARGET_SPACING

    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    new_size = [
        int(original_size[0] * original_spacing[0] / target_spacing[0]),
        int(original_size[1] * original_spacing[1] / target_spacing[1]),
        int(original_size[2] * original_spacing[2] / target_spacing[2])
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(sitk_image)


def get_hu_at_point(sitk_image: sitk.Image, world_coords: tuple) -> float:
    """
    获取指定世界坐标处的HU值

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像
    world_coords : tuple
        世界坐标 (x, y, z)，单位为毫米

    Returns
    -------
    float
        该点的HU值，如果超出边界则返回0
    """
    origin = get_origin(sitk_image)
    spacing = get_spacing(sitk_image)
    size = sitk_image.GetSize()

    voxel = world_to_voxel(world_coords, origin, spacing)

    if (0 <= voxel[0] < size[0] and
        0 <= voxel[1] < size[1] and
            0 <= voxel[2] < size[2]):
        return float(sitk_image.GetPixel(voxel))

    return 0.0


def compute_local_hu_statistics(image_array: np.ndarray, voxel_coords: tuple,
                                radius_voxel: int) -> dict:
    """
    计算局部HU值统计特征

    在指定体素周围的邻域内计算多种统计特征。

    Parameters
    ----------
    image_array : np.ndarray
        CT图像数组，形状为(z, y, x)
    voxel_coords : tuple
        体素坐标 (i, j, k)，对应图像数组的(x, y, z)索引
    radius_voxel : int
        邻域半径（体素数）

    Returns
    -------
    dict
        包含以下统计特征的字典：
        - mean: 平均值
        - std: 标准差
        - median: 中位数
        - max: 最大值
        - min: 最小值
        - percentile25: 第25百分位数
        - percentile75: 第75百分位数
    """
    neighborhood = extract_neighborhood(image_array, voxel_coords, radius_voxel)

    if neighborhood.size == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'max': 0.0,
            'min': 0.0,
            'percentile25': 0.0,
            'percentile75': 0.0
        }

    flat = neighborhood.flatten()

    return {
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'median': float(np.median(flat)),
        'max': float(np.max(flat)),
        'min': float(np.min(flat)),
        'percentile25': float(np.percentile(flat, 25)),
        'percentile75': float(np.percentile(flat, 75))
    }


def compute_local_hu_statistics_world(sitk_image: sitk.Image, world_coords: tuple,
                                       radius_mm: float) -> dict:
    """
    在世界坐标系下计算局部HU值统计特征

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像
    world_coords : tuple
        世界坐标 (x, y, z)
    radius_mm : float
        邻域半径（毫米）

    Returns
    -------
    dict
        包含mean, std, median, max, min, percentile25, percentile75的字典
    """
    origin = get_origin(sitk_image)
    spacing = get_spacing(sitk_image)

    voxel_coords = world_to_voxel(world_coords, origin, spacing)
    radius_voxel = max(1, int(radius_mm / min(spacing)))

    image_array = get_ct_array(sitk_image)

    return compute_local_hu_statistics(image_array, voxel_coords, radius_voxel)


def preprocess_ct(sitk_image: sitk.Image) -> sitk.Image:
    """
    CT图像预处理流水线

    根据配置参数对CT图像进行预处理，包括重采样等操作。

    Parameters
    ----------
    sitk_image : sitk.Image
        原始CT图像

    Returns
    -------
    sitk.Image
        预处理后的CT图像
    """
    if config.ct_preprocessing.RESAMPLE:
        sitk_image = resample_image(sitk_image)

    return sitk_image


def is_point_inside_image(sitk_image: sitk.Image, world_coords: tuple) -> bool:
    """
    判断世界坐标点是否在图像范围内

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像
    world_coords : tuple
        世界坐标 (x, y, z)，单位为毫米

    Returns
    -------
    bool
        如果点在图像范围内返回True，否则返回False
    """
    origin = get_origin(sitk_image)
    spacing = get_spacing(sitk_image)
    size = sitk_image.GetSize()

    voxel = world_to_voxel(world_coords, origin, spacing)

    return (0 <= voxel[0] < size[0] and
            0 <= voxel[1] < size[1] and
            0 <= voxel[2] < size[2])


def get_image_bounds(sitk_image: sitk.Image) -> tuple:
    """
    获取图像的世界坐标边界

    Parameters
    ----------
    sitk_image : sitk.Image
        SimpleITK格式的CT图像

    Returns
    -------
    tuple
        图像边界 (x_min, x_max, y_min, y_max, z_min, z_max)，单位为毫米
    """
    origin = get_origin(sitk_image)
    spacing = get_spacing(sitk_image)
    size = get_size(sitk_image)

    x_min = origin[0]
    x_max = origin[0] + size[0] * spacing[0]
    y_min = origin[1]
    y_max = origin[1] + size[1] * spacing[1]
    z_min = origin[2]
    z_max = origin[2] + size[2] * spacing[2]

    return (x_min, x_max, y_min, y_max, z_min, z_max)