"""
optimizer.py
------------------------------------
AI辅助置钉点预测模块 - 局部优化器

在初始点周围的搜索空间内执行局部优化，
寻找评分最高的置钉点
"""

import numpy as np
from time import perf_counter

from .config import config
from .geometry import (
    generate_search_sphere,
    generate_search_ball,
    find_closest_point_on_mesh,
    is_point_inside_mesh
)


class EntryPointOptimizer:
    """
    置钉点局部优化器

    在初始点周围的搜索空间内寻找评分最高的置钉点。
    支持多种搜索策略：球面搜索、球体搜索、网格搜索。
    """

    def __init__(self, scorer, bone_mesh=None):
        """
        初始化优化器

        Parameters
        ----------
        scorer : EntryPointScorer
            评分器实例
        bone_mesh : pyvista.PolyData, optional
            骨骼网格，用于约束搜索范围
        """
        self.scorer = scorer
        self.bone_mesh = bone_mesh

    def optimize(self, initial_point: tuple, radius: float = None,
                  step: float = None, iterations: int = None) -> tuple:
        """
        执行局部优化

        Parameters
        ----------
        initial_point : tuple
            初始点坐标 (x, y, z)
        radius : float, optional
            搜索半径（毫米），默认为配置中的值
        step : float, optional
            搜索步长（毫米），默认为配置中的值
        iterations : int, optional
            迭代次数，默认为配置中的值

        Returns
        -------
        tuple
            优化后的最佳置钉点坐标
        """
        if radius is None:
            radius = config.optimization.RADIUS
        if step is None:
            step = config.optimization.STEP
        if iterations is None:
            iterations = config.optimization.ITERATIONS

        print("[OPTIMIZER] enter optimize")
        print("initial_point:", initial_point)
        print("config.radius:", radius)
        print("config.step:", step)
        print("config.iterations:", iterations)

        total_t = perf_counter()

        if iterations <= 0:
            raise RuntimeError("Optimizer loop may not converge: iterations must be positive")
        if step <= 0:
            raise RuntimeError("Optimizer loop may not converge: step must be positive")
        if radius <= 0:
            raise RuntimeError("Optimizer loop may not converge: radius must be positive")

        _t = perf_counter()
        best_point = initial_point
        best_score = self.scorer.score(initial_point)
        print(f"[OPTIMIZER] initial score cost={perf_counter() - _t:.3f}s, best_score={best_score}")

        for iteration in range(iterations):
            iteration_t = perf_counter()
            current_radius = radius * (1.0 - iteration / iterations)
            print(f"[OPTIMIZER] iteration {iteration + 1}/{iterations} start, radius={current_radius}, step={step}")

            _t = perf_counter()
            search_points = generate_search_sphere(best_point, current_radius, step)
            generate_cost = perf_counter() - _t
            print(f"[OPTIMIZER] generate local points cost={generate_cost:.3f}s, radius={current_radius}, count={len(search_points)}")
            print("num_local_points:", len(search_points))

            scoring_t = perf_counter()
            inside_cost_total = 0.0
            closest_cost_total = 0.0
            score_cost_total = 0.0
            update_cost_total = 0.0
            adjusted_count = 0
            for idx, point in enumerate(search_points):
                if idx % 50 == 0:
                    print(f"[OPTIMIZER-PROGRESS] scoring local point {idx}/{len(search_points)}")

                if self.bone_mesh is not None:
                    _inside_t = perf_counter()
                    if is_point_inside_mesh(point, self.bone_mesh):
                        inside_cost_total += perf_counter() - _inside_t
                        _closest_t = perf_counter()
                        point = find_closest_point_on_mesh(point, self.bone_mesh)
                        closest_cost_total += perf_counter() - _closest_t
                        adjusted_count += 1
                    else:
                        inside_cost_total += perf_counter() - _inside_t

                _score_t = perf_counter()
                score = self.scorer.score(point)
                score_cost_total += perf_counter() - _score_t

                _update_t = perf_counter()
                if score > best_score:
                    best_score = score
                    best_point = point
                update_cost_total += perf_counter() - _update_t

            print(
                "[OPTIMIZER] score local points done, "
                f"cost={perf_counter() - scoring_t:.3f}s, count={len(search_points)}, "
                f"inside_cost={inside_cost_total:.3f}s, closest_cost={closest_cost_total:.3f}s, "
                f"score_cost={score_cost_total:.3f}s, update_best_cost={update_cost_total:.3f}s, "
                f"adjusted_inside_points={adjusted_count}"
            )

            shrink_t = perf_counter()
            if current_radius < step:
                print(
                    f"[OPTIMIZER] iteration {iteration + 1}/{iterations} done, "
                    f"cost={perf_counter() - iteration_t:.3f}s, best_score={best_score}"
                )
                print(f"[OPTIMIZER] shrink/break check cost={perf_counter() - shrink_t:.3f}s, current_radius={current_radius}")
                break
            print(f"[OPTIMIZER] shrink/break check cost={perf_counter() - shrink_t:.3f}s, current_radius={current_radius}")

            print(
                f"[OPTIMIZER] iteration {iteration + 1}/{iterations} done, "
                f"cost={perf_counter() - iteration_t:.3f}s, best_score={best_score}"
            )

        final_t = perf_counter()
        print("[OPTIMIZER] before final local ball search")
        _t = perf_counter()
        final_points = generate_search_ball(best_point, step * 2, step * 0.5)
        print(f"[OPTIMIZER] generate final points cost={perf_counter() - _t:.3f}s, count={len(final_points)}")
        print("num_local_points:", len(final_points))

        scoring_t = perf_counter()
        inside_cost_total = 0.0
        closest_cost_total = 0.0
        score_cost_total = 0.0
        update_cost_total = 0.0
        adjusted_count = 0
        for idx, point in enumerate(final_points):
            if idx % 50 == 0:
                print(f"[OPTIMIZER-PROGRESS] scoring final local point {idx}/{len(final_points)}")

            if self.bone_mesh is not None:
                _inside_t = perf_counter()
                if is_point_inside_mesh(point, self.bone_mesh):
                    inside_cost_total += perf_counter() - _inside_t
                    _closest_t = perf_counter()
                    point = find_closest_point_on_mesh(point, self.bone_mesh)
                    closest_cost_total += perf_counter() - _closest_t
                    adjusted_count += 1
                else:
                    inside_cost_total += perf_counter() - _inside_t

            _score_t = perf_counter()
            score = self.scorer.score(point)
            score_cost_total += perf_counter() - _score_t

            _update_t = perf_counter()
            if score > best_score:
                best_score = score
                best_point = point
            update_cost_total += perf_counter() - _update_t

        print(
            "[OPTIMIZER] score final points done, "
            f"cost={perf_counter() - scoring_t:.3f}s, count={len(final_points)}, "
            f"inside_cost={inside_cost_total:.3f}s, closest_cost={closest_cost_total:.3f}s, "
            f"score_cost={score_cost_total:.3f}s, update_best_cost={update_cost_total:.3f}s, "
            f"adjusted_inside_points={adjusted_count}"
        )
        print(f"[OPTIMIZER] final local ball search done, cost={perf_counter() - final_t:.3f}s, best_score={best_score}")
        print(f"[OPTIMIZER] optimize done, total_cost={perf_counter() - total_t:.3f}s")

        return best_point

    def optimize_with_gradients(self, initial_point: tuple, radius: float = None,
                                step: float = None, iterations: int = None) -> tuple:
        """
        使用梯度上升法执行局部优化

        Parameters
        ----------
        initial_point : tuple
            初始点坐标 (x, y, z)
        radius : float, optional
            搜索半径
        step : float, optional
            搜索步长
        iterations : int, optional
            迭代次数

        Returns
        -------
        tuple
            优化后的最佳置钉点坐标
        """
        if radius is None:
            radius = config.optimization.RADIUS
        if step is None:
            step = config.optimization.STEP
        if iterations is None:
            iterations = config.optimization.ITERATIONS

        current_point = initial_point
        best_point = initial_point
        best_score = self.scorer.score(initial_point)

        for iteration in range(iterations):
            grad = self._compute_gradient(current_point, step * 0.5)
            grad_norm = np.linalg.norm(grad)

            if grad_norm > 0:
                grad = grad / grad_norm

            new_point = tuple(np.array(current_point) + grad * step)

            if self.bone_mesh is not None:
                if is_point_inside_mesh(new_point, self.bone_mesh):
                    new_point = find_closest_point_on_mesh(new_point, self.bone_mesh)

            new_score = self.scorer.score(new_point)

            if new_score > best_score:
                best_score = new_score
                best_point = new_point
                current_point = new_point
            else:
                step *= 0.5
                if step < 0.1:
                    break

        return best_point

    def _compute_gradient(self, point: tuple, delta: float) -> np.ndarray:
        """
        计算评分函数在指定点的梯度

        Parameters
        ----------
        point : tuple
            当前点坐标
        delta : float
            差分步长

        Returns
        -------
        np.ndarray
            梯度向量
        """
        base_score = self.scorer.score(point)

        grad = np.zeros(3)

        for i in range(3):
            offset = np.zeros(3)
            offset[i] = delta
            plus_point = tuple(np.array(point) + offset)
            plus_score = self.scorer.score(plus_point)

            offset[i] = -delta
            minus_point = tuple(np.array(point) + offset)
            minus_score = self.scorer.score(minus_point)

            grad[i] = (plus_score - minus_score) / (2 * delta)

        return grad

    def optimize_multistart(self, initial_points: list, radius: float = None,
                            step: float = None, iterations: int = None) -> tuple:
        """
        多起点优化

        Parameters
        ----------
        initial_points : list
            初始点列表
        radius : float, optional
            搜索半径
        step : float, optional
            搜索步长
        iterations : int, optional
            迭代次数

        Returns
        -------
        tuple
            所有起点中最优的置钉点坐标
        """
        best_point = None
        best_score = -np.inf

        for initial_point in initial_points:
            optimized_point = self.optimize(initial_point, radius, step, iterations)
            score = self.scorer.score(optimized_point)

            if score > best_score:
                best_score = score
                best_point = optimized_point

        if best_point is None and len(initial_points) > 0:
            best_point = initial_points[0]

        return best_point


def optimize_entry_point(initial_point: tuple, scorer, bone_mesh=None,
                         radius: float = None, step: float = None,
                         iterations: int = None) -> tuple:
    """
    优化置钉点（便捷函数）

    Parameters
    ----------
    initial_point : tuple
        初始点坐标 (x, y, z)
    scorer : EntryPointScorer
        评分器实例
    bone_mesh : pyvista.PolyData, optional
        骨骼网格
    radius : float, optional
        搜索半径
    step : float, optional
        搜索步长
    iterations : int, optional
        迭代次数

    Returns
    -------
    tuple
        优化后的最佳置钉点坐标
    """
    optimizer = EntryPointOptimizer(scorer, bone_mesh)
    return optimizer.optimize(initial_point, radius, step, iterations)
