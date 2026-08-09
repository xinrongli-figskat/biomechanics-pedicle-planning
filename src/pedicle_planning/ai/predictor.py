"""
predictor.py
------------------------------------
AI辅助置钉点预测模块 - 预测器

生成候选置钉点，调用评分器和优化器，返回最终预测点
"""

import numpy as np
import pyvista as pv
from time import perf_counter

from .config import config
from .scorer import EntryPointScorer
from .optimizer import EntryPointOptimizer
from .geometry import (
    get_mesh_bounds_center,
    get_mesh_surface_points,
    generate_search_sphere,
    find_closest_point_on_mesh,
    get_mesh_normal_at_point
)


class EntryPointPredictor:
    """
    AI辅助置钉点预测器

    基于医学影像分析的轻量级智能决策模块，用于预测椎弓根螺钉的最佳置钉点。

    预测流程：
    1. 生成候选置钉点
    2. 使用评分器评估每个候选点
    3. 使用优化器对高分候选点进行局部优化
    4. 返回评分最高的置钉点
    """

    def __init__(self):
        """
        初始化预测器
        """
        self.last_point = None
        self.last_score = None

    def predict(self, bone_model: pv.PolyData, sitk_image=None) -> list:
        """
        预测最佳置钉点

        Parameters
        ----------
        bone_model : pv.PolyData
            骨骼网格模型（STL）
        sitk_image : SimpleITK.Image, optional
            CT图像

        Returns
        -------
        list
            预测的置钉点坐标 [x, y, z]
        """
        print("STEP3")
        if bone_model is None:
            raise RuntimeError("Bone model is None.")

        point = self._predict_with_ai(bone_model, sitk_image)

        self.last_point = point

        if config.debug.ENABLED:
            print("========== AI Predictor ==========")
            print("Predicted Entry Point:", point)
            print("Score:", self.last_score)
            print("==================================")

        return point

    def _predict_with_ai(self, bone_model: pv.PolyData, sitk_image=None) -> list:
        """
        使用AI辅助方法进行置钉点预测

        Parameters
        ----------
        bone_model : pv.PolyData
            骨骼网格模型
        sitk_image : SimpleITK.Image, optional
            CT图像

        Returns
        -------
        list
            预测的置钉点坐标 [x, y, z]
        """
        print("STEP4")
        print("bone_model.n_points:", bone_model.n_points)
        print("bone_model.n_cells:", bone_model.n_cells)
        print("TOP_CANDIDATES:", config.candidate_generation.TOP_CANDIDATES)
        print("NUM_CANDIDATES:", config.candidate_generation.NUM_CANDIDATES)
        print("SURFACE_SAMPLES:", config.candidate_generation.SURFACE_SAMPLES)
        print("OPTIMIZATION_ITERATIONS:", config.optimization.ITERATIONS)

        print("[STEP4.1] before create EntryPointScorer")
        tic = perf_counter()
        scorer = EntryPointScorer(sitk_image, bone_model)
        print(f"[STEP4.2] after create EntryPointScorer, cost={perf_counter() - tic:.3f}s")

        print("[STEP4.3] before create EntryPointOptimizer")
        tic = perf_counter()
        optimizer = EntryPointOptimizer(scorer, bone_model)
        print(f"[STEP4.4] after create EntryPointOptimizer, cost={perf_counter() - tic:.3f}s")

        print("[STEP4.5] before generate candidate points")
        tic = perf_counter()
        candidate_points = self._generate_candidate_points(bone_model)
        print(f"[STEP4.6] after generate candidate points, cost={perf_counter() - tic:.3f}s, num_candidates={len(candidate_points)}")

        if len(candidate_points) == 0:
            return self._geometry_predict(bone_model)

        print("[STEP4.7] before scoring candidates")
        tic = perf_counter()
        scored_points = []
        total_candidates = len(candidate_points)
        for i, point in enumerate(candidate_points):
            if i % 100 == 0:
                print(f"[PROGRESS] scoring candidates: {i}/{total_candidates}")
            score_tic = perf_counter()
            score = scorer.score(point)
            score_cost = perf_counter() - score_tic
            if score_cost > 1.0:
                print(f"[SLOW SCORE] scorer.score cost={score_cost:.3f}s, index={i}, point={point}")
            scored_points.append((point, score))
        scoring_total_cost = perf_counter() - tic
        avg_score_cost = scoring_total_cost / total_candidates if total_candidates > 0 else 0.0
        print(f"[SUMMARY] scoring candidates done, count={len(candidate_points)}, total_cost={scoring_total_cost:.3f}s, avg_cost={avg_score_cost:.4f}s")
        print(f"[STEP4.8] after scoring candidates, cost={scoring_total_cost:.3f}s")

        print("[STEP4.9] before sorting scored points")
        scored_points.sort(key=lambda x: x[1], reverse=True)
        print("[STEP4.10] after sorting scored points")

        top_candidates = scored_points[:config.candidate_generation.TOP_CANDIDATES]

        best_point = None
        best_score = -np.inf

        print("[STEP4.11] before optimizing top candidates")
        tic = perf_counter()
        total_top_candidates = len(top_candidates)
        for i, (candidate_point, _) in enumerate(top_candidates):
            print(f"[PROGRESS] optimizing top candidate: {i + 1}/{total_top_candidates}")
            optimize_tic = perf_counter()
            optimized_point = optimizer.optimize(candidate_point)
            optimize_cost = perf_counter() - optimize_tic
            if optimize_cost > 2.0:
                print(f"[SLOW OPTIMIZE] optimizer.optimize cost={optimize_cost:.3f}s, index={i}, point={candidate_point}")
            score = scorer.score(optimized_point)

            if score > best_score:
                best_score = score
                best_point = optimized_point
        print(f"[STEP4.12] after optimizing top candidates, cost={perf_counter() - tic:.3f}s")

        if best_point is None:
            best_point = scored_points[0][0]
            best_score = scored_points[0][1]

        self.last_score = best_score

        return [float(best_point[0]), float(best_point[1]), float(best_point[2])]

    def _generate_candidate_points(self, bone_model: pv.PolyData) -> list:
        """
        生成候选置钉点

        根据骨骼网格的几何特征生成多个候选点，包括：
        - 网格边界中心附近的点
        - 网格表面采样点
        - 沿法向量方向偏移的点

        Parameters
        ----------
        bone_model : pv.PolyData
            骨骼网格模型

        Returns
        -------
        list
            候选点坐标列表
        """
        candidates = []

        center = get_mesh_bounds_center(bone_model)
        candidates.append(center)

        bounds = bone_model.bounds
        size_x = bounds[1] - bounds[0]
        size_y = bounds[3] - bounds[2]
        size_z = bounds[5] - bounds[4]

        offset_x = size_x * 0.1
        offset_y = size_y * 0.1
        offset_z = size_z * 0.1

        offset_points = [
            (center[0] + offset_x, center[1], center[2]),
            (center[0] - offset_x, center[1], center[2]),
            (center[0], center[1] + offset_y, center[2]),
            (center[0], center[1] - offset_y, center[2]),
            (center[0], center[1], center[2] + offset_z),
            (center[0], center[1], center[2] - offset_z),
        ]
        candidates.extend(offset_points)

        surface_points = get_mesh_surface_points(bone_model, num_points=config.candidate_generation.SURFACE_SAMPLES)
        candidates.extend(surface_points)

        for point in surface_points[:config.candidate_generation.SURFACE_SAMPLES // 2]:
            normal = get_mesh_normal_at_point(bone_model, point)
            offset_distance = config.candidate_generation.OFFSET_DISTANCE

            offset_point_in = tuple(np.array(point) + np.array(normal) * offset_distance)
            offset_point_out = tuple(np.array(point) - np.array(normal) * offset_distance)

            candidates.append(offset_point_in)
            candidates.append(offset_point_out)

        search_radius = config.candidate_generation.SEARCH_RADIUS
        step = config.candidate_generation.STEP

        sphere_points = generate_search_sphere(center, search_radius, step)
        candidates.extend(sphere_points)

        unique_candidates = []
        seen_points = set()

        for point in candidates:
            rounded = tuple(round(p, 2) for p in point)
            if rounded not in seen_points:
                seen_points.add(rounded)
                unique_candidates.append(point)

        return unique_candidates

    def _geometry_predict(self, bone_model: pv.PolyData) -> list:
        """
        使用几何方法进行初始预测（备用方案）

        Parameters
        ----------
        bone_model : pv.PolyData
            骨骼网格模型

        Returns
        -------
        list
            预测的置钉点坐标 [x, y, z]
        """
        bounds = bone_model.bounds
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        x = (xmin + xmax) * 0.5
        y = ymax
        z = (zmin + zmax) * 0.5

        return [float(x), float(y), float(z)]

    def get_last_prediction(self):
        """
        获取最近一次预测结果

        Returns
        -------
        list or None
            最近一次预测的置钉点坐标，如果没有预测过则返回None
        """
        return self.last_point

    def get_last_score(self):
        """
        获取最近一次预测的评分

        Returns
        -------
        float or None
            最近一次预测的评分，如果没有预测过则返回None
        """
        return self.last_score


predictor = EntryPointPredictor()
