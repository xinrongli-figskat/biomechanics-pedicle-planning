"""
inference.py
------------------------------------
AI辅助置钉点预测模块 - 推理接口

提供唯一的外部接口：predict_entry_point()

该接口接收CT图像和骨骼网格，返回预测的置钉点坐标。
"""

import pyvista as pv

from .predictor import EntryPointPredictor


_predictor = EntryPointPredictor()


def predict_entry_point(sitk_image, bone_mesh) -> tuple:
    """
    预测椎弓根螺钉的最佳置钉点

    这是AI辅助置钉点预测模块的唯一外部接口。

    Parameters
    ----------
    sitk_image : SimpleITK.Image
        CT图像，用于HU值分析和密度评估
    bone_mesh : pyvista.PolyData
        骨骼网格模型，用于几何分析和空间约束

    Returns
    -------
    tuple
        预测的置钉点坐标 (x, y, z)，单位为毫米

    Raises
    ------
    ValueError
        如果输入参数无效
    RuntimeError
        如果预测过程失败
    """
    if bone_mesh is None:
        raise ValueError("bone_mesh cannot be None")

    if not isinstance(bone_mesh, pv.PolyData):
        raise ValueError("bone_mesh must be a pyvista.PolyData object")

    try:
        result = _predictor.predict(bone_mesh, sitk_image)

        if result is None or len(result) != 3:
            raise RuntimeError("Prediction returned invalid result")

        return (float(result[0]), float(result[1]), float(result[2]))

    except RuntimeError as e:
        import traceback
        traceback.print_exc()
        raise e
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Prediction failed: {str(e)}") from e


def get_last_prediction() -> tuple:
    """
    获取最近一次预测结果

    Returns
    -------
    tuple or None
        最近一次预测的置钉点坐标，如果没有预测过则返回None
    """
    result = _predictor.get_last_prediction()
    if result is None:
        return None
    return (float(result[0]), float(result[1]), float(result[2]))


def get_last_score() -> float:
    """
    获取最近一次预测的评分

    Returns
    -------
    float or None
        最近一次预测的综合评分，范围[0, 1]，如果没有预测过则返回None
    """
    return _predictor.get_last_score()
