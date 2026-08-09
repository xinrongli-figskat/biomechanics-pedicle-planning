"""
ai/__init__.py
------------------------------------
AI辅助椎弓根螺钉置钉点预测模块

基于医学影像分析的轻量级智能决策模块，用于预测椎弓根螺钉的最佳置钉点。

主要功能：
- CT图像预处理和HU值分析
- 骨骼网格几何分析
- 候选置钉点生成
- 综合评分系统（HU密度、骨皮质距离、稳定性、平滑度、解剖中心）
- 局部优化算法
- 统一的推理接口

使用方法：
    from script.funcs.ai import predict_entry_point

    point = predict_entry_point(sitk_image, bone_mesh)
"""
print("=" * 60)
print("AI MODULE LOADED")
print("=" * 60)

from .inference import (
    predict_entry_point,
    get_last_prediction,
    get_last_score
)

from .predictor import EntryPointPredictor, predictor

from .config import config

__all__ = [
    'predict_entry_point',
    'get_last_prediction',
    'get_last_score',
    'EntryPointPredictor',
    'predictor',
    'config'
]