"""
scorer.py
------------------------------------
AI辅助置钉点预测模块 - 综合评分器

基于局部HU密度、骨皮质距离、邻域稳定性、平滑度和解剖中心趋势
计算候选置钉点的综合评分
"""

import numpy as np
from time import perf_counter

DEBUG_SCORE_DETAIL = False
SLOW_SCORE_THRESHOLD = 1.0

try:
    import vtkmodules.all as vtk
except Exception:
    vtk = None

from .config import config
from .utils import world_to_voxel, extract_neighborhood, get_ct_array, get_spacing, get_origin
from .geometry import (
    distance_point_to_mesh,
    get_mesh_normal_at_point,
    get_mesh_bounds_center,
    normalize_vector,
    dot_product,
    generate_search_sphere,
    _warn_once
)


class EntryPointScorer:
    """
    置钉点评分器

    根据多个解剖学和影像学特征计算候选点的综合评分。
    评分越高表示该点越适合作为置钉点。
    """

    def __init__(self, sitk_image=None, bone_mesh=None):
        """
        初始化评分器

        Parameters
        ----------
        sitk_image : SimpleITK.Image, optional
            CT图像
        bone_mesh : pyvista.PolyData, optional
            骨骼网格
        """
        self.sitk_image = sitk_image
        self.bone_mesh = bone_mesh
        self.ct_array = None
        self.spacing = None
        self.origin = None
        self.mesh_center = None
        self.mesh_points = None
        self.mesh_cell_normals = None
        self._point_locator = None
        self._cell_locator = None
        self._generic_cell = None
        self._score_count = 0

        if sitk_image is not None:
            self.ct_array = get_ct_array(sitk_image)
            self.spacing = get_spacing(sitk_image)
            self.origin = get_origin(sitk_image)

        if bone_mesh is not None:
            self.mesh_center = get_mesh_bounds_center(bone_mesh)
            self._build_mesh_cache(bone_mesh)

    def _build_mesh_cache(self, bone_mesh):
        t_total = perf_counter()

        try:
            self.mesh_points = np.asarray(bone_mesh.points, dtype=np.float64)
        except Exception:
            self.mesh_points = None

        t_normals = perf_counter()
        try:
            self.mesh_cell_normals = np.asarray(bone_mesh.cell_normals, dtype=np.float64)
        except Exception:
            self.mesh_cell_normals = None
        normals_cost = perf_counter() - t_normals

        point_locator_cost = 0.0
        cell_locator_cost = 0.0

        if vtk is not None:
            try:
                t_locator = perf_counter()
                self._point_locator = vtk.vtkPointLocator()
                self._point_locator.SetDataSet(bone_mesh)
                self._point_locator.BuildLocator()
                point_locator_cost = perf_counter() - t_locator
            except Exception:
                self._point_locator = None

            try:
                t_locator = perf_counter()
                self._cell_locator = vtk.vtkCellLocator()
                self._cell_locator.SetDataSet(bone_mesh)
                self._cell_locator.BuildLocator()
                self._generic_cell = vtk.vtkGenericCell()
                cell_locator_cost = perf_counter() - t_locator
            except Exception:
                self._cell_locator = None
                self._generic_cell = None

        print(
            "[SCORE-INIT] mesh cache built, "
            f"cost={perf_counter() - t_total:.3f}s, "
            f"n_points={getattr(bone_mesh, 'n_points', 'unknown')}, "
            f"n_cells={getattr(bone_mesh, 'n_cells', 'unknown')}, "
            f"normals_cost={normals_cost:.3f}s, "
            f"point_locator_cost={point_locator_cost:.3f}s, "
            f"cell_locator_cost={cell_locator_cost:.3f}s"
        )

    def _distance_point_to_mesh_cached(self, point: tuple) -> float:
        if self._point_locator is not None and self.mesh_points is not None:
            point_array = np.asarray(point, dtype=np.float64).reshape(3)
            point_id = int(self._point_locator.FindClosestPoint(point_array))
            if 0 <= point_id < len(self.mesh_points):
                return float(np.linalg.norm(point_array - self.mesh_points[point_id]))

        return distance_point_to_mesh(point, self.bone_mesh)

    def _mesh_normal_at_point_cached(self, point: tuple) -> tuple:
        if (
            self._cell_locator is not None and
            self._generic_cell is not None and
            self.mesh_cell_normals is not None
        ):
            point_array = np.asarray(point, dtype=np.float64).reshape(3)
            closest_point = [0.0, 0.0, 0.0]
            cell_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.mutable(0.0)
            self._cell_locator.FindClosestPoint(
                point_array,
                closest_point,
                self._generic_cell,
                cell_id,
                sub_id,
                dist2
            )
            cell_id = int(cell_id)
            if 0 <= cell_id < len(self.mesh_cell_normals):
                normal = self.mesh_cell_normals[cell_id]
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    return tuple(normal / norm)

        return get_mesh_normal_at_point(self.bone_mesh, point)

    def score(self, point: tuple) -> float:
        """
        计算候选点的综合评分

        Parameters
        ----------
        point : tuple
            候选点坐标 (x, y, z)

        Returns
        -------
        float
            归一化综合评分，范围[0, 1]，越高越好
        """
        self._score_count += 1
        total_t = perf_counter()
        scores = []
        weights = []

        _t = perf_counter()
        hu_score = self._calculate_hu_density_score(point)
        hu_cost = perf_counter() - _t
        scores.append(hu_score)
        weights.append(config.scoring.HU_WEIGHT)

        _t = perf_counter()
        cortex_score = self._calculate_cortex_distance_score(point)
        cortex_cost = perf_counter() - _t
        scores.append(cortex_score)
        weights.append(config.scoring.CORTEX_DISTANCE_WEIGHT)

        _t = perf_counter()
        stability_score = self._calculate_neighborhood_stability_score(point)
        stability_cost = perf_counter() - _t
        scores.append(stability_score)
        weights.append(config.scoring.STABILITY_WEIGHT)

        _t = perf_counter()
        smoothness_score = self._calculate_smoothness_score(point)
        smoothness_cost = perf_counter() - _t
        scores.append(smoothness_score)
        weights.append(config.scoring.SMOOTHNESS_WEIGHT)

        _t = perf_counter()
        center_score = self._calculate_anatomical_center_score(point)
        anatomical_cost = perf_counter() - _t
        scores.append(center_score)
        weights.append(config.scoring.ANATOMICAL_CENTER_WEIGHT)

        total_weight = sum(weights)
        total_cost = perf_counter() - total_t
        if DEBUG_SCORE_DETAIL or total_cost > SLOW_SCORE_THRESHOLD:
            print(f"[SLOW SCORE] total={total_cost:.3f}s, point={point}")
            print(f"[SCORE] hu_score cost={hu_cost:.3f}s")
            print(f"[SCORE] cortex_score cost={cortex_cost:.3f}s")
            print(f"[SCORE] stability_score cost={stability_cost:.3f}s")
            print(f"[SCORE] smoothness_score cost={smoothness_cost:.3f}s")
            print(f"[SCORE] anatomical_score cost={anatomical_cost:.3f}s")

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return float(weighted_sum / total_weight)

    def _calculate_hu_density_score(self, point: tuple) -> float:
        """
        计算局部HU密度评分

        椎弓根置钉点应位于致密骨区域（较高HU值）。
        HU值范围：皮质骨约1000-2000 HU，松质骨约100-800 HU。

        Parameters
        ----------
        point : tuple
            候选点坐标

        Returns
        -------
        float
            HU密度评分，范围[0, 1]
        """
        if self.ct_array is None or self.spacing is None or self.origin is None:
            return 0.5

        try:
            total_t = perf_counter()
            voxel_t = perf_counter()
            voxel = world_to_voxel(point, self.origin, self.spacing)
            voxel_cost = perf_counter() - voxel_t
            radius = config.neighborhood.RADIUS
            neighborhood_t = perf_counter()
            neighborhood = extract_neighborhood(self.ct_array, voxel, radius)
            neighborhood_cost = perf_counter() - neighborhood_t

            if neighborhood.size == 0:
                return 0.0

            stats_t = perf_counter()
            valid_voxels = neighborhood[neighborhood > config.ct_preprocessing.HU_MIN]
            if valid_voxels.size == 0:
                return 0.0

            mean_hu = float(np.mean(valid_voxels))
            std_hu = float(np.std(valid_voxels)) if valid_voxels.size > 1 else 0.0

            hu_score = (mean_hu - config.ct_preprocessing.HU_MIN) / (config.ct_preprocessing.HU_MAX - config.ct_preprocessing.HU_MIN)
            hu_score = max(0.0, min(1.0, hu_score))

            density_score = hu_score

            if std_hu > 0:
                density_score = density_score * (1 - std_hu / config.ct_preprocessing.HU_MAX)

            total_cost = perf_counter() - total_t
            if total_cost > 1.0:
                print(
                    "[SCORE-HU] "
                    f"total={total_cost:.3f}s, voxel_cost={voxel_cost:.3f}s, "
                    f"neighborhood_cost={neighborhood_cost:.3f}s, "
                    f"stats_cost={perf_counter() - stats_t:.3f}s, "
                    f"radius={radius}, neighborhood_shape={neighborhood.shape}, "
                    f"voxel_count={neighborhood.size}, valid_voxel_count={valid_voxels.size}"
                )

            return float(max(0.0, min(1.0, density_score)))

        except Exception:
            _warn_once("score_hu_density", "Failed to calculate HU density score; using fallback score.")
            return 0.3

    def _calculate_cortex_distance_score(self, point: tuple) -> float:
        """
        计算骨皮质距离评分

        置钉点应位于骨皮质表面，但不能太靠近边缘，以确保足够的骨把持力。

        Parameters
        ----------
        point : tuple
            候选点坐标

        Returns
        -------
        float
            骨皮质距离评分，范围[0, 1]
        """
        if self.bone_mesh is None:
            return 0.5

        try:
            total_t = perf_counter()
            distance_t = perf_counter()
            distance = self._distance_point_to_mesh_cached(point)
            distance_cost = perf_counter() - distance_t

            ideal_distance = config.cortex_detection.TARGET_DISTANCE
            tolerance = config.cortex_detection.TOLERANCE

            if distance <= ideal_distance + tolerance:
                score = 1.0 - abs(distance - ideal_distance) / tolerance
            else:
                score = max(0.0, 1.0 - (distance - ideal_distance - tolerance) / (5 * tolerance))

            total_cost = perf_counter() - total_t
            if total_cost > 1.0:
                print(
                    "[SCORE-CORTEX] "
                    f"total={total_cost:.3f}s, distance_cost={distance_cost:.3f}s, "
                    f"distance={distance:.3f}, point_locator_cached={self._point_locator is not None}"
                )

            return float(max(0.0, min(1.0, score)))

        except Exception:
            _warn_once("score_cortex_distance", "Failed to calculate cortex distance score; using fallback score.")
            return 0.3

    def _calculate_neighborhood_stability_score(self, point: tuple) -> float:
        """
        计算邻域稳定性评分

        评估候选点周围区域的稳定性，避免选择在结构不稳定或变异性大的区域。

        Parameters
        ----------
        point : tuple
            候选点坐标

        Returns
        -------
        float
            邻域稳定性评分，范围[0, 1]
        """
        if self.ct_array is None or self.spacing is None or self.origin is None:
            return 0.5

        try:
            total_t = perf_counter()
            voxel_t = perf_counter()
            voxel = world_to_voxel(point, self.origin, self.spacing)
            voxel_cost = perf_counter() - voxel_t
            radius = config.neighborhood.RADIUS
            neighborhood_t = perf_counter()
            neighborhood = extract_neighborhood(self.ct_array, voxel, radius)
            neighborhood_cost = perf_counter() - neighborhood_t

            if neighborhood.size < 10:
                return 0.2

            stats_t = perf_counter()
            mean_hu = float(np.mean(neighborhood))
            std_hu = float(np.std(neighborhood))

            if mean_hu == 0:
                return 0.0

            cv = std_hu / mean_hu

            max_cv = 1.0
            stability_score = max(0.0, 1.0 - cv / max_cv)

            total_cost = perf_counter() - total_t
            if total_cost > 1.0:
                print(
                    "[SCORE-STABILITY] "
                    f"total={total_cost:.3f}s, voxel_cost={voxel_cost:.3f}s, "
                    f"neighborhood_cost={neighborhood_cost:.3f}s, "
                    f"stats_cost={perf_counter() - stats_t:.3f}s, "
                    f"radius={radius}, neighborhood_shape={neighborhood.shape}, "
                    f"voxel_count={neighborhood.size}"
                )

            return float(max(0.0, min(1.0, stability_score)))

        except Exception:
            _warn_once("score_neighborhood_stability", "Failed to calculate neighborhood stability score; using fallback score.")
            return 0.3

    def _calculate_smoothness_score(self, point: tuple) -> float:
        """
        计算表面平滑度评分

        置钉点应位于骨骼表面相对平滑的区域，避免选择尖锐边缘或突起处。

        Parameters
        ----------
        point : tuple
            候选点坐标

        Returns
        -------
        float
            表面平滑度评分，范围[0, 1]
        """
        if self.bone_mesh is None:
            return 0.5

        try:
            total_t = perf_counter()
            center_normal_t = perf_counter()
            normal = self._mesh_normal_at_point_cached(point)
            center_normal_cost = perf_counter() - center_normal_t

            search_radius = config.smoothness.SEARCH_RADIUS
            step = config.smoothness.STEP
            sphere_t = perf_counter()
            sphere_points = generate_search_sphere(point, search_radius, step)
            sphere_cost = perf_counter() - sphere_t

            if len(sphere_points) < 5:
                return 0.3

            normal_deviations = []
            normals_t = perf_counter()
            for sphere_point in sphere_points:
                sphere_normal = self._mesh_normal_at_point_cached(sphere_point)
                dot = dot_product(normal, sphere_normal)
                deviation = 1.0 - abs(dot)
                normal_deviations.append(deviation)
            normals_cost = perf_counter() - normals_t

            stats_t = perf_counter()
            avg_deviation = float(np.mean(normal_deviations))

            max_deviation = 0.5
            smoothness_score = max(0.0, 1.0 - avg_deviation / max_deviation)

            total_cost = perf_counter() - total_t
            if total_cost > 1.0:
                print(
                    "[SCORE-SMOOTHNESS] "
                    f"total={total_cost:.3f}s, center_normal_cost={center_normal_cost:.3f}s, "
                    f"sphere_cost={sphere_cost:.3f}s, normals_cost={normals_cost:.3f}s, "
                    f"stats_cost={perf_counter() - stats_t:.3f}s, "
                    f"sphere_points={len(sphere_points)}, normal_queries={len(sphere_points) + 1}, "
                    f"cell_locator_cached={self._cell_locator is not None}, "
                    f"cell_normals_cached={self.mesh_cell_normals is not None}"
                )

            return float(max(0.0, min(1.0, smoothness_score)))

        except Exception:
            _warn_once("score_smoothness", "Failed to calculate smoothness score; using fallback score.")
            return 0.4

    def _calculate_anatomical_center_score(self, point: tuple) -> float:
        """
        计算解剖中心趋势评分

        椎弓根置钉点通常位于椎弓根的中心区域，该评分衡量候选点与解剖中心的接近程度。

        Parameters
        ----------
        point : tuple
            候选点坐标

        Returns
        -------
        float
            解剖中心趋势评分，范围[0, 1]
        """
        if self.bone_mesh is None or self.mesh_center is None:
            return 0.5

        try:
            total_t = perf_counter()
            bounds_t = perf_counter()
            bounds = self.bone_mesh.bounds
            bounds_cost = perf_counter() - bounds_t
            size_x = bounds[1] - bounds[0]
            size_y = bounds[3] - bounds[2]
            size_z = bounds[5] - bounds[4]

            max_dim = max(size_x, size_y, size_z)
            distance_t = perf_counter()
            distance = np.linalg.norm(np.array(point) - np.array(self.mesh_center))
            distance_cost = perf_counter() - distance_t

            max_distance = max_dim * 0.5
            if max_distance == 0:
                return 1.0

            score = max(0.0, 1.0 - distance / max_distance)

            total_cost = perf_counter() - total_t
            if total_cost > 1.0:
                print(
                    "[SCORE-ANATOMICAL] "
                    f"total={total_cost:.3f}s, bounds_cost={bounds_cost:.3f}s, "
                    f"distance_cost={distance_cost:.3f}s"
                )

            return float(max(0.0, min(1.0, score)))

        except Exception:
            _warn_once("score_anatomical_center", "Failed to calculate anatomical center score; using fallback score.")
            return 0.4

    def get_detailed_scores(self, point: tuple) -> dict:
        """
        获取候选点的详细评分信息

        Parameters
        ----------
        point : tuple
            候选点坐标

        Returns
        -------
        dict
            包含各项评分的字典
        """
        return {
            'hu_density': self._calculate_hu_density_score(point),
            'cortex_distance': self._calculate_cortex_distance_score(point),
            'neighborhood_stability': self._calculate_neighborhood_stability_score(point),
            'smoothness': self._calculate_smoothness_score(point),
            'anatomical_center': self._calculate_anatomical_center_score(point),
            'overall': self.score(point)
        }


def score_entry_point(point: tuple, sitk_image=None, bone_mesh=None) -> float:
    """
    计算候选置钉点的综合评分（便捷函数）

    Parameters
    ----------
    point : tuple
        候选点坐标 (x, y, z)
    sitk_image : SimpleITK.Image, optional
        CT图像
    bone_mesh : pyvista.PolyData, optional
        骨骼网格

    Returns
    -------
    float
        归一化综合评分，范围[0, 1]
    """
    scorer = EntryPointScorer(sitk_image, bone_mesh)
    return scorer.score(point)
