"""
geometry.py
------------------------------------
AI辅助置钉点预测模块 - 几何工具函数

提供距离计算、向量工具、搜索球生成、局部坐标系等功能
"""

import numpy as np
import pyvista as pv

from .config import config


_GEOMETRY_WARNED = set()


def _warn_once(key: str, message: str):
    if key not in _GEOMETRY_WARNED:
        print(message)
        _GEOMETRY_WARNED.add(key)


def _mesh_points(mesh: pv.PolyData) -> np.ndarray:
    points = getattr(mesh, "points", None)
    if points is None:
        raise RuntimeError("mesh does not expose points")
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        raise RuntimeError("mesh must have at least one point")
    return points.reshape((-1, 3))


def _safe_sample_mesh_points(mesh: pv.PolyData, num_points: int) -> list:
    method = getattr(mesh, "sample_points", None)
    if callable(method):
        try:
            sampled = method(number_of_points=num_points)
            return [tuple(point) for point in sampled.points]
        except TypeError:
            sampled = method(num_points)
            return [tuple(point) for point in sampled.points]

    points = _mesh_points(mesh)
    replace = num_points > len(points)
    indices = np.random.choice(len(points), size=int(num_points), replace=replace)
    return [tuple(point) for point in points[indices]]


def _safe_distance_point_to_mesh(point: tuple, mesh: pv.PolyData) -> float:
    method = getattr(mesh, "distance", None)
    if callable(method):
        try:
            return float(method(point))
        except TypeError:
            pass
        except AttributeError:
            pass

    p = np.asarray(point, dtype=np.float64).reshape(3)
    points = _mesh_points(mesh)

    find_closest_point = getattr(mesh, "find_closest_point", None)
    if callable(find_closest_point):
        point_id = int(find_closest_point(p))
        if 0 <= point_id < len(points):
            return float(np.linalg.norm(p - points[point_id]))

    return float(np.min(np.linalg.norm(points - p, axis=1)))


def _safe_find_closest_cell(mesh: pv.PolyData, point: tuple, return_closest_point: bool = False):
    method = getattr(mesh, "find_closest_cell", None)
    if not callable(method):
        raise RuntimeError("mesh does not support find_closest_cell")

    if return_closest_point:
        try:
            return method(point, return_closest_point=True)
        except TypeError:
            cell_id = method(point)
            if hasattr(mesh, "cell_centers"):
                centers = mesh.cell_centers().points
                if 0 <= int(cell_id) < len(centers):
                    return cell_id, centers[int(cell_id)]
            points = _mesh_points(mesh)
            point_array = np.asarray(point, dtype=np.float64).reshape(3)
            distances = np.linalg.norm(points - point_array, axis=1)
            return cell_id, points[int(np.argmin(distances))]

    return method(point)


def _safe_find_closest_point_on_mesh(point: tuple, mesh: pv.PolyData) -> tuple:
    point_array = np.asarray(point, dtype=np.float64).reshape(3)

    try:
        _, closest_point = _safe_find_closest_cell(mesh, point_array, return_closest_point=True)
        return tuple(np.asarray(closest_point, dtype=np.float64).reshape(3))
    except Exception:
        pass

    points = _mesh_points(mesh)
    find_closest_point = getattr(mesh, "find_closest_point", None)
    if callable(find_closest_point):
        point_id = int(find_closest_point(point_array))
        if 0 <= point_id < len(points):
            return tuple(points[point_id])

    distances = np.linalg.norm(points - point_array, axis=1)
    return tuple(points[int(np.argmin(distances))])


def distance_point_to_point(p1: tuple, p2: tuple) -> float:
    """
    计算两点之间的欧氏距离

    Parameters
    ----------
    p1 : tuple
        第一个点的坐标 (x, y, z)
    p2 : tuple
        第二个点的坐标 (x, y, z)

    Returns
    -------
    float
        两点之间的距离
    """
    if p1 is None or p2 is None:
        raise ValueError("p1 and p2 must not be None")

    if len(p1) != 3 or len(p2) != 3:
        raise ValueError("Points must have 3 coordinates")

    return np.linalg.norm(np.array(p1) - np.array(p2))


def distance_point_to_mesh(point: tuple, mesh: pv.PolyData) -> float:
    """
    计算点到网格表面的最近距离

    Parameters
    ----------
    point : tuple
        查询点的坐标 (x, y, z)
    mesh : pv.PolyData
        目标网格

    Returns
    -------
    float
        点到网格表面的最近距离
    """
    if mesh is None:
        raise ValueError("mesh must not be None")

    if mesh.n_points == 0:
        raise ValueError("mesh must have at least one point")

    if point is None:
        raise ValueError("point must not be None")

    if len(point) != 3:
        raise ValueError("point must have 3 coordinates")

    try:
        return _safe_distance_point_to_mesh(point, mesh)
    except Exception as e:
        raise RuntimeError("Failed to compute point-to-mesh distance") from e


def distance_point_to_plane(point: tuple, plane_point: tuple, plane_normal: tuple) -> float:
    """
    计算点到平面的距离

    Parameters
    ----------
    point : tuple
        查询点的坐标 (x, y, z)
    plane_point : tuple
        平面上一点的坐标 (x, y, z)
    plane_normal : tuple
        平面法向量 (nx, ny, nz)

    Returns
    -------
    float
        点到平面的有符号距离
    """
    if point is None or plane_point is None or plane_normal is None:
        raise ValueError("point, plane_point, and plane_normal must not be None")

    if len(point) != 3 or len(plane_point) != 3 or len(plane_normal) != 3:
        raise ValueError("All points must have 3 coordinates")

    p = np.array(point, dtype=np.float64)
    pp = np.array(plane_point, dtype=np.float64)
    n = np.array(plane_normal, dtype=np.float64)

    norm = np.linalg.norm(n)
    if norm > 1e-10:
        n = n / norm

    return float(np.dot(p - pp, n))


def normalize_vector(v: tuple) -> tuple:
    """
    归一化向量

    Parameters
    ----------
    v : tuple
        输入向量 (x, y, z)

    Returns
    -------
    tuple
        归一化后的单位向量
    """
    if v is None:
        raise ValueError("v must not be None")

    if len(v) != 3:
        raise ValueError("vector must have 3 components")

    arr = np.array(v, dtype=np.float64)
    norm = np.linalg.norm(arr)

    if norm < 1e-10:
        return (0.0, 0.0, 0.0)

    return tuple(arr / norm)


def cross_product(v1: tuple, v2: tuple) -> tuple:
    """
    计算两个向量的叉积

    Parameters
    ----------
    v1 : tuple
        第一个向量 (x, y, z)
    v2 : tuple
        第二个向量 (x, y, z)

    Returns
    -------
    tuple
        叉积结果向量
    """
    if v1 is None or v2 is None:
        raise ValueError("v1 and v2 must not be None")

    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("vectors must have 3 components")

    return tuple(np.cross(np.array(v1, dtype=np.float64), np.array(v2, dtype=np.float64)))


def dot_product(v1: tuple, v2: tuple) -> float:
    """
    计算两个向量的点积

    Parameters
    ----------
    v1 : tuple
        第一个向量 (x, y, z)
    v2 : tuple
        第二个向量 (x, y, z)

    Returns
    -------
    float
        点积结果
    """
    if v1 is None or v2 is None:
        raise ValueError("v1 and v2 must not be None")

    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("vectors must have 3 components")

    return float(np.dot(np.array(v1, dtype=np.float64), np.array(v2, dtype=np.float64)))


def angle_between_vectors(v1: tuple, v2: tuple) -> float:
    """
    计算两个向量之间的夹角（弧度）

    Parameters
    ----------
    v1 : tuple
        第一个向量
    v2 : tuple
        第二个向量

    Returns
    -------
    float
        夹角（弧度），范围在[0, pi]之间
    """
    if v1 is None or v2 is None:
        raise ValueError("v1 and v2 must not be None")

    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    dp = dot_product(v1_norm, v2_norm)
    dp = max(min(dp, 1.0), -1.0)

    return float(np.arccos(dp))


def generate_search_sphere(center: tuple, radius: float, step: float) -> list:
    """
    在球面上生成均匀分布的搜索点

    Parameters
    ----------
    center : tuple
        球心坐标 (x, y, z)
    radius : float
        球半径（毫米）
    step : float
        步长（毫米）

    Returns
    -------
    list
        球面上的点坐标列表
    """
    if center is None:
        raise ValueError("center must not be None")

    if len(center) != 3:
        raise ValueError("center must have 3 coordinates")

    if radius <= 0:
        raise ValueError("radius must be positive")

    if step <= 0:
        raise ValueError("step must be positive")

    points = []

    theta_steps = max(3, int(np.pi * radius / step))
    phi_steps = max(3, int(2 * np.pi * radius / step))

    for i in range(theta_steps):
        theta = np.pi * i / (theta_steps - 1)
        for j in range(phi_steps):
            phi = 2 * np.pi * j / phi_steps
            x = center[0] + radius * np.sin(theta) * np.cos(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(theta)
            points.append((x, y, z))

    return points


def generate_search_ball(center: tuple, radius: float, step: float) -> list:
    """
    在球体内生成均匀分布的搜索点

    Parameters
    ----------
    center : tuple
        球心坐标 (x, y, z)
    radius : float
        球半径（毫米）
    step : float
        步长（毫米）

    Returns
    -------
    list
        球体内的点坐标列表
    """
    if center is None:
        raise ValueError("center must not be None")

    if len(center) != 3:
        raise ValueError("center must have 3 coordinates")

    if radius <= 0:
        raise ValueError("radius must be positive")

    if step <= 0:
        raise ValueError("step must be positive")

    points = []

    r_steps = max(1, int(radius / step))
    theta_steps = max(3, int(np.pi * radius / step))
    phi_steps = max(3, int(2 * np.pi * radius / step))

    for r_idx in range(r_steps + 1):
        r = r_idx * step
        for i in range(theta_steps):
            theta = np.pi * i / (theta_steps - 1)
            for j in range(phi_steps):
                phi = 2 * np.pi * j / phi_steps
                x = center[0] + r * np.sin(theta) * np.cos(phi)
                y = center[1] + r * np.sin(theta) * np.sin(phi)
                z = center[2] + r * np.cos(theta)
                points.append((x, y, z))

    return points


def generate_grid_points(center: tuple, size: tuple, step: float) -> list:
    """
    在矩形区域内生成均匀分布的网格点

    Parameters
    ----------
    center : tuple
        中心坐标 (x, y, z)
    size : tuple
        三个方向的尺寸 (dx, dy, dz)
    step : float
        步长（毫米）

    Returns
    -------
    list
        网格点坐标列表
    """
    if center is None:
        raise ValueError("center must not be None")

    if size is None:
        raise ValueError("size must not be None")

    if len(center) != 3 or len(size) != 3:
        raise ValueError("center and size must have 3 coordinates")

    if step <= 0:
        raise ValueError("step must be positive")

    points = []

    half_dx = size[0] / 2.0
    half_dy = size[1] / 2.0
    half_dz = size[2] / 2.0

    x_steps = max(1, int(size[0] / step))
    y_steps = max(1, int(size[1] / step))
    z_steps = max(1, int(size[2] / step))

    for i in range(x_steps + 1):
        x = center[0] - half_dx + i * step
        for j in range(y_steps + 1):
            y = center[1] - half_dy + j * step
            for k in range(z_steps + 1):
                z = center[2] - half_dz + k * step
                points.append((x, y, z))

    return points


def get_mesh_normal_at_point(mesh: pv.PolyData, point: tuple) -> tuple:
    """
    获取网格表面在指定点处的法向量

    Parameters
    ----------
    mesh : pv.PolyData
        目标网格
    point : tuple
        查询点坐标 (x, y, z)

    Returns
    -------
    tuple
        法向量，如果无法计算则返回(0, 0, 1)
    """
    if mesh is None:
        raise ValueError("mesh must not be None")

    if mesh.n_points == 0:
        raise ValueError("mesh must have at least one point")

    if point is None:
        raise ValueError("point must not be None")

    if len(point) != 3:
        raise ValueError("point must have 3 coordinates")

    try:
        cell_id = _safe_find_closest_cell(mesh, point)
        if cell_id >= 0:
            normals = mesh.cell_normals
            if normals is not None and cell_id < len(normals):
                normal = np.asarray(normals[cell_id], dtype=np.float64)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    return tuple(normal / norm)

    except Exception as e:
        _warn_once("get_mesh_normal_at_point", "Failed to compute mesh normal; using fallback normal.")

    return (0.0, 0.0, 1.0)


def get_mesh_bounds_center(mesh: pv.PolyData) -> tuple:
    """
    获取网格的边界中心

    Parameters
    ----------
    mesh : pv.PolyData
        目标网格

    Returns
    -------
    tuple
        边界中心坐标 (x, y, z)
    """
    if mesh is None:
        raise ValueError("mesh must not be None")

    if mesh.n_points == 0:
        raise ValueError("mesh must have at least one point")

    bounds = mesh.bounds
    x = (bounds[0] + bounds[1]) / 2.0
    y = (bounds[2] + bounds[3]) / 2.0
    z = (bounds[4] + bounds[5]) / 2.0

    return (x, y, z)


def get_mesh_surface_points(mesh: pv.PolyData, num_points: int = 1000) -> list:
    """
    从网格表面采样点

    Parameters
    ----------
    mesh : pv.PolyData
        目标网格
    num_points : int, optional
        采样点数，默认为1000

    Returns
    -------
    list
        表面采样点坐标列表
    """
    if mesh is None:
        raise ValueError("mesh must not be None")

    if mesh.n_points == 0:
        raise ValueError("mesh must have at least one point")

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    try:
        return _safe_sample_mesh_points(mesh, num_points)
    except Exception as e:
        raise RuntimeError("Failed to sample mesh surface points") from e


def project_point_to_plane(point: tuple, plane_point: tuple, plane_normal: tuple) -> tuple:
    """
    将点投影到平面上

    Parameters
    ----------
    point : tuple
        要投影的点
    plane_point : tuple
        平面上一点
    plane_normal : tuple
        平面法向量

    Returns
    -------
    tuple
        投影点坐标
    """
    if point is None or plane_point is None or plane_normal is None:
        raise ValueError("point, plane_point, and plane_normal must not be None")

    if len(point) != 3 or len(plane_point) != 3 or len(plane_normal) != 3:
        raise ValueError("All points must have 3 coordinates")

    p = np.array(point, dtype=np.float64)
    pp = np.array(plane_point, dtype=np.float64)
    n = np.array(plane_normal, dtype=np.float64)

    norm = np.linalg.norm(n)
    if norm > 1e-10:
        n = n / norm

    dist = np.dot(p - pp, n)
    projected = p - dist * n

    return tuple(projected)


def create_local_coordinate_system(center: tuple, normal: tuple) -> dict:
    """
    创建局部坐标系

    Parameters
    ----------
    center : tuple
        局部坐标系原点
    normal : tuple
        局部坐标系z轴方向（法向量）

    Returns
    -------
    dict
        包含origin, x_axis, y_axis, z_axis的字典
    """
    if center is None:
        raise ValueError("center must not be None")

    if normal is None:
        raise ValueError("normal must not be None")

    if len(center) != 3 or len(normal) != 3:
        raise ValueError("center and normal must have 3 coordinates")

    z_axis = normalize_vector(normal)

    if abs(z_axis[0]) < 0.9:
        x_axis = normalize_vector(cross_product(z_axis, (1.0, 0.0, 0.0)))
    else:
        x_axis = normalize_vector(cross_product(z_axis, (0.0, 1.0, 0.0)))

    y_axis = normalize_vector(cross_product(z_axis, x_axis))

    return {
        'origin': center,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_axis': z_axis
    }


def transform_to_local_coordinates(world_point: tuple, local_cs: dict) -> tuple:
    """
    将世界坐标转换为局部坐标

    Parameters
    ----------
    world_point : tuple
        世界坐标点
    local_cs : dict
        局部坐标系，包含origin, x_axis, y_axis, z_axis

    Returns
    -------
    tuple
        局部坐标点
    """
    if world_point is None:
        raise ValueError("world_point must not be None")

    if local_cs is None:
        raise ValueError("local_cs must not be None")

    if len(world_point) != 3:
        raise ValueError("world_point must have 3 coordinates")

    p = np.array(world_point, dtype=np.float64)
    o = np.array(local_cs['origin'], dtype=np.float64)
    dx = p - o

    x = float(np.dot(dx, local_cs['x_axis']))
    y = float(np.dot(dx, local_cs['y_axis']))
    z = float(np.dot(dx, local_cs['z_axis']))

    return (x, y, z)


def transform_to_world_coordinates(local_point: tuple, local_cs: dict) -> tuple:
    """
    将局部坐标转换为世界坐标

    Parameters
    ----------
    local_point : tuple
        局部坐标点
    local_cs : dict
        局部坐标系

    Returns
    -------
    tuple
        世界坐标点
    """
    if local_point is None:
        raise ValueError("local_point must not be None")

    if local_cs is None:
        raise ValueError("local_cs must not be None")

    if len(local_point) != 3:
        raise ValueError("local_point must have 3 coordinates")

    lp = np.array(local_point, dtype=np.float64)
    o = np.array(local_cs['origin'], dtype=np.float64)
    xa = np.array(local_cs['x_axis'], dtype=np.float64)
    ya = np.array(local_cs['y_axis'], dtype=np.float64)
    za = np.array(local_cs['z_axis'], dtype=np.float64)

    world = o + lp[0] * xa + lp[1] * ya + lp[2] * za

    return tuple(world)


def is_point_inside_mesh(point: tuple, mesh: pv.PolyData) -> bool:
    """
    判断点是否在网格内部

    Parameters
    ----------
    point : tuple
        查询点坐标
    mesh : pv.PolyData
        目标网格

    Returns
    -------
    bool
        如果点在网格内部返回True，否则返回False
    """
    if mesh is None:
        raise ValueError("mesh must not be None")

    if mesh.n_points == 0:
        raise ValueError("mesh must have at least one point")

    if point is None:
        raise ValueError("point must not be None")

    if len(point) != 3:
        raise ValueError("point must have 3 coordinates")

    try:
        point_mesh = pv.PolyData(np.asarray([point], dtype=np.float64))
        result = point_mesh.select_enclosed_points(mesh, check_surface=False)
        if result is not None and 'SelectedPoints' in result.array_names:
            return bool(result['SelectedPoints'][0] > 0.5)
    except Exception as e:
        _warn_once("is_point_inside_mesh", "Failed to test whether point is inside mesh; using False fallback.")

    return False


def find_closest_point_on_mesh(point: tuple, mesh: pv.PolyData) -> tuple:
    """
    查找网格上距离给定点最近的点

    Parameters
    ----------
    point : tuple
        查询点坐标
    mesh : pv.PolyData
        目标网格

    Returns
    -------
    tuple
        最近点坐标
    """
    if mesh is None:
        raise ValueError("mesh must not be None")

    if mesh.n_points == 0:
        raise ValueError("mesh must have at least one point")

    if point is None:
        raise ValueError("point must not be None")

    if len(point) != 3:
        raise ValueError("point must have 3 coordinates")

    try:
        return _safe_find_closest_point_on_mesh(point, mesh)
    except Exception as e:
        _warn_once("find_closest_point_on_mesh", "Failed to find closest mesh point; returning input point.")
        return point


def compute_mesh_thickness_at_point(mesh: pv.PolyData, point: tuple,
                                     direction: tuple, max_distance: float = 50.0) -> float:
    """
    计算网格在指定点沿指定方向的厚度

    Parameters
    ----------
    mesh : pv.PolyData
        目标网格
    point : tuple
        查询点坐标
    direction : tuple
        测量方向向量
    max_distance : float, optional
        最大搜索距离，默认为50毫米

    Returns
    -------
    float
        厚度值，如果未找到交点则返回0
    """
    if mesh is None:
        raise ValueError("mesh must not be None")

    if mesh.n_points == 0:
        raise ValueError("mesh must have at least one point")

    if point is None or direction is None:
        raise ValueError("point and direction must not be None")

    if len(point) != 3 or len(direction) != 3:
        raise ValueError("point and direction must have 3 coordinates")

    if max_distance <= 0:
        raise ValueError("max_distance must be positive")

    try:
        direction_norm = normalize_vector(direction)
        end_point = tuple(np.array(point, dtype=np.float64) + max_distance * np.array(direction_norm, dtype=np.float64))

        intersection = mesh.ray_trace(point, end_point)
        if len(intersection[0]) >= 2:
            return distance_point_to_point(intersection[0][0], intersection[0][1])
    except Exception as e:
        _warn_once("compute_mesh_thickness_at_point", "Failed to compute mesh thickness; using 0.0 fallback.")

    return 0.0
