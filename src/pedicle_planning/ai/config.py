from dataclasses import dataclass, field


@dataclass
class CTPreprocessingConfig:
    """
    CT图像预处理配置参数

    CT图像在输入AI模块前需要进行预处理，包括：
    - HU值截断：将超出范围的HU值裁剪到有效区间
    - 重采样：统一图像分辨率
    - 归一化：将HU值映射到[0, 1]范围

    HU值说明：
    - 空气: -1000 HU
    - 水: 0 HU
    - 松质骨: 100-800 HU
    - 皮质骨: 800-2000 HU
    """

    HU_MIN: int = -200
    """
    HU值截断下限。
    低于此值的像素将被设置为HU_MIN。
    默认-200，排除空气和软组织，保留骨骼相关区域。
    """

    HU_MAX: int = 2000
    """
    HU值截断上限。
    高于此值的像素将被设置为HU_MAX。
    默认2000，覆盖皮质骨的最高HU值。
    """

    RESAMPLE: bool = True
    """
    是否进行图像重采样。
    设置为True时，将图像分辨率统一为TARGET_SPACING。
    """

    TARGET_SPACING: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """
    重采样目标分辨率，单位为毫米/像素。
    格式为(x, y, z)三元组。
    默认(1.0, 1.0, 1.0)，即各向同性1mm分辨率。
    """

    NORMALIZE_MIN: float = 0.0
    """
    归一化最小值。
    截断后的HU值将被线性映射到此值。
    """

    NORMALIZE_MAX: float = 1.0
    """
    归一化最大值。
    截断后的HU值将被线性映射到此值。
    """

    def validate(self) -> None:
        """
        验证配置参数的合法性

        Raises
        ------
        ValueError
            如果配置参数不合法
        """
        if self.HU_MIN >= self.HU_MAX:
            raise ValueError(f"HU_MIN ({self.HU_MIN}) must be less than HU_MAX ({self.HU_MAX})")

        if self.NORMALIZE_MIN >= self.NORMALIZE_MAX:
            raise ValueError(f"NORMALIZE_MIN ({self.NORMALIZE_MIN}) must be less than NORMALIZE_MAX ({self.NORMALIZE_MAX})")

        if len(self.TARGET_SPACING) != 3:
            raise ValueError(f"TARGET_SPACING must be a tuple of 3 elements, got {len(self.TARGET_SPACING)}")

        for i, spacing in enumerate(self.TARGET_SPACING):
            if spacing <= 0:
                raise ValueError(f"TARGET_SPACING[{i}] ({spacing}) must be positive")


@dataclass
class NeighborhoodConfig:
    """
    邻域分析配置参数

    邻域分析用于评估候选置钉点周围的局部特征，包括：
    - HU值分布
    - 密度稳定性
    - 解剖结构一致性

    邻域大小直接影响分析的局部性：
    - 小邻域：更敏感，捕捉精细结构
    - 大邻域：更稳定，反映宏观特征
    """

    RADIUS: int = 3
    """
    邻域半径，单位为体素。
    邻域大小为(2*RADIUS+1)^3个体素。
    默认3，表示7x7x7的立方体邻域。
    """

    SPACING: float = 0.5
    """
    邻域采样间距，单位为毫米。
    用于在世界坐标系中定义邻域范围。
    """

    def validate(self) -> None:
        """
        验证配置参数的合法性

        Raises
        ------
        ValueError
            如果配置参数不合法
        """
        if self.RADIUS <= 0:
            raise ValueError(f"RADIUS ({self.RADIUS}) must be greater than 0")

        if self.SPACING <= 0:
            raise ValueError(f"SPACING ({self.SPACING}) must be greater than 0")


@dataclass
class CortexDetectionConfig:
    """
    骨皮质检测配置参数

    骨皮质是骨骼外层的致密骨结构，置钉点应位于骨皮质表面。
    该配置控制骨皮质检测的各项参数：
    - HU阈值：区分皮质骨和松质骨
    - 搜索半径：寻找骨皮质边界的范围
    - 厚度要求：最小骨皮质厚度
    """

    HU_THRESHOLD: int = 800
    """
    骨皮质HU阈值。
    高于此阈值的像素被认为是皮质骨。
    默认800 HU，这是皮质骨和松质骨的典型分界值。
    """

    SEARCH_RADIUS: float = 10.0
    """
    骨皮质搜索半径，单位为毫米。
    在该范围内搜索骨皮质边界。
    """

    MIN_THICKNESS: float = 1.0
    """
    最小骨皮质厚度，单位为毫米。
    低于此厚度的区域不适合作为置钉点。
    """

    TARGET_DISTANCE: float = 0.0
    """
    置钉点与骨皮质表面的理想距离，单位为毫米。
    0表示置钉点应位于骨皮质表面。
    """

    TOLERANCE: float = 2.0
    """
    骨皮质距离容差，单位为毫米。
    置钉点距离骨皮质表面在[TARGET_DISTANCE-TOLERANCE, TARGET_DISTANCE+TOLERANCE]
    范围内被认为是可接受的。
    """

    def validate(self) -> None:
        """
        验证配置参数的合法性

        Raises
        ------
        ValueError
            如果配置参数不合法
        """
        if self.HU_THRESHOLD <= 0:
            raise ValueError(f"HU_THRESHOLD ({self.HU_THRESHOLD}) must be greater than 0")

        if self.SEARCH_RADIUS <= 0:
            raise ValueError(f"SEARCH_RADIUS ({self.SEARCH_RADIUS}) must be greater than 0")

        if self.MIN_THICKNESS <= 0:
            raise ValueError(f"MIN_THICKNESS ({self.MIN_THICKNESS}) must be greater than 0")

        if self.TOLERANCE < 0:
            raise ValueError(f"TOLERANCE ({self.TOLERANCE}) must be non-negative")


@dataclass
class ScoringConfig:
    """
    综合评分权重配置

    候选置钉点的综合评分由多个子评分加权组成：
    - HU密度评分：评估局部骨密度
    - 骨皮质距离评分：评估置钉点与骨皮质的距离
    - 邻域稳定性评分：评估周围区域的一致性
    - 表面平滑度评分：评估骨骼表面的平滑程度
    - 解剖中心趋势评分：评估置钉点与解剖中心的接近程度

    所有权重之和应为1.0。
    """

    HU_WEIGHT: float = 0.35
    """
    HU密度评分权重。
    较高的权重表示更重视骨密度特征。
    默认0.35，是最重要的评分因素。
    """

    CORTEX_DISTANCE_WEIGHT: float = 0.25
    """
    骨皮质距离评分权重。
    默认0.25，确保置钉点位于骨皮质表面附近。
    """

    STABILITY_WEIGHT: float = 0.20
    """
    邻域稳定性评分权重。
    默认0.20，评估周围区域的一致性。
    """

    SMOOTHNESS_WEIGHT: float = 0.10
    """
    表面平滑度评分权重。
    默认0.10，避免选择尖锐边缘或突起处。
    """

    ANATOMICAL_CENTER_WEIGHT: float = 0.10
    """
    解剖中心趋势评分权重。
    默认0.10，倾向于选择椎弓根中心区域。
    """

    def validate(self) -> None:
        """
        验证配置参数的合法性

        检查五个权重之和是否等于1（允许1e-6的误差）。

        Raises
        ------
        ValueError
            如果权重之和不等于1
        """
        total_weight = (
            self.HU_WEIGHT
            + self.CORTEX_DISTANCE_WEIGHT
            + self.STABILITY_WEIGHT
            + self.SMOOTHNESS_WEIGHT
            + self.ANATOMICAL_CENTER_WEIGHT
        )

        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Sum of weights must be 1.0, got {total_weight:.6f}. "
                f"Individual weights: HU={self.HU_WEIGHT}, CORTEX={self.CORTEX_DISTANCE_WEIGHT}, "
                f"STABILITY={self.STABILITY_WEIGHT}, SMOOTHNESS={self.SMOOTHNESS_WEIGHT}, "
                f"ANATOMICAL_CENTER={self.ANATOMICAL_CENTER_WEIGHT}"
            )


@dataclass
class SmoothnessConfig:
    """
    表面平滑度检测配置参数

    平滑度检测用于评估骨骼表面在候选点周围的弯曲程度。
    置钉点应位于相对平滑的区域，避免选择尖锐边缘。
    """

    SEARCH_RADIUS: float = 5.0
    """
    平滑度搜索半径，单位为毫米。
    在该范围内采样表面法向量，评估方向一致性。
    """

    STEP: float = 1.0
    """
    采样步长，单位为毫米。
    较小的步长提供更密集的采样，但计算成本更高。
    """

    def validate(self) -> None:
        """
        验证配置参数的合法性

        Raises
        ------
        ValueError
            如果配置参数不合法
        """
        if self.SEARCH_RADIUS <= 0:
            raise ValueError(f"SEARCH_RADIUS ({self.SEARCH_RADIUS}) must be greater than 0")

        if self.STEP <= 0:
            raise ValueError(f"STEP ({self.STEP}) must be greater than 0")


@dataclass
class CandidateGenerationConfig:
    """
    候选置钉点生成配置参数

    候选点生成策略包括：
    - 网格中心附近的点
    - 表面采样点
    - 沿法向量偏移的点
    - 球面搜索点

    这些候选点将通过评分器筛选，选择最优的若干个进行优化。
    """

    SPHERE_RADIUS: float = 5.0
    """
    候选点搜索球半径，单位为毫米。
    """

    SPACING: float = 1.0
    """
    候选点采样间距，单位为毫米。
    """

    NUM_CANDIDATES: int = 20
    """
    初始候选点数量。
    """

    TOP_CANDIDATES: int = 5
    """
    进入局部优化阶段的候选点数量。
    从所有候选点中选择评分最高的TOP_CANDIDATES个进行优化。
    """

    SURFACE_SAMPLES: int = 500
    """
    骨骼表面采样点数。
    从骨骼网格表面均匀采样的点数量。
    """

    OFFSET_DISTANCE: float = 2.0
    """
    沿法向量偏移的距离，单位为毫米。
    从表面点向内部和外部偏移生成额外候选点。
    """

    SEARCH_RADIUS: float = 10.0
    """
    候选点全局搜索半径，单位为毫米。
    围绕网格中心在该半径内生成候选点。
    """

    STEP: float = 2.0
    """
    全局搜索步长，单位为毫米。
    """

    def validate(self) -> None:
        """
        验证配置参数的合法性

        Raises
        ------
        ValueError
            如果配置参数不合法
        """
        if self.NUM_CANDIDATES <= 0:
            raise ValueError(f"NUM_CANDIDATES ({self.NUM_CANDIDATES}) must be greater than 0")

        if self.TOP_CANDIDATES > self.NUM_CANDIDATES:
            raise ValueError(f"TOP_CANDIDATES ({self.TOP_CANDIDATES}) must be less than or equal to NUM_CANDIDATES ({self.NUM_CANDIDATES})")

        if self.SURFACE_SAMPLES <= 0:
            raise ValueError(f"SURFACE_SAMPLES ({self.SURFACE_SAMPLES}) must be greater than 0")

        if self.SPHERE_RADIUS <= 0:
            raise ValueError(f"SPHERE_RADIUS ({self.SPHERE_RADIUS}) must be greater than 0")

        if self.STEP <= 0:
            raise ValueError(f"STEP ({self.STEP}) must be greater than 0")


@dataclass
class OptimizationConfig:
    """
    局部优化配置参数

    局部优化在候选点周围的搜索空间内寻找评分更高的点。
    采用迭代式球面搜索策略，逐步缩小搜索范围。
    """

    RADIUS: float = 3.0
    """
    初始优化搜索半径，单位为毫米。
    """

    STEP: float = 0.5
    """
    优化搜索步长，单位为毫米。
    较小的步长提供更精细的搜索，但计算成本更高。
    """

    ITERATIONS: int = 5
    """
    优化迭代次数。
    每次迭代后搜索半径按比例缩小。
    """

    def validate(self) -> None:
        """
        验证配置参数的合法性

        Raises
        ------
        ValueError
            如果配置参数不合法
        """
        if self.RADIUS <= 0:
            raise ValueError(f"RADIUS ({self.RADIUS}) must be greater than 0")

        if self.STEP <= 0:
            raise ValueError(f"STEP ({self.STEP}) must be greater than 0")

        if self.ITERATIONS <= 0:
            raise ValueError(f"ITERATIONS ({self.ITERATIONS}) must be greater than 0")


@dataclass
class DebugConfig:
    """
    调试配置参数

    控制调试信息的输出和保存。
    """

    ENABLED: bool = True
    """
    是否启用调试模式。
    启用后会打印详细的预测过程信息。
    """

    SAVE_DEBUG_RESULT: bool = False
    """
    是否保存调试结果。
    """

    SAVE_POINT: bool = False
    """
    是否保存预测点坐标。
    """

    def validate(self) -> None:
        """
        验证配置参数的合法性

        DebugConfig无需特殊验证。
        """
        pass


@dataclass
class AIConfig:
    """
    AI辅助置钉点预测模块全局配置

    包含所有子配置类的实例，提供统一的配置访问入口。
    """

    VERSION: str = "2.0"
    """
    模块版本号
    """

    MODULE_NAME: str = "AI-assisted Pedicle Screw Entry Point Predictor"
    """
    模块名称
    """

    RANDOM_SEED: int = 2026
    """
    随机种子，用于保证结果的可重复性。
    """

    ct_preprocessing: CTPreprocessingConfig = field(
        default_factory=CTPreprocessingConfig
    )
    """CT图像预处理配置"""

    neighborhood: NeighborhoodConfig = field(
        default_factory=NeighborhoodConfig
    )
    """邻域分析配置"""

    cortex_detection: CortexDetectionConfig = field(
        default_factory=CortexDetectionConfig
    )
    """骨皮质检测配置"""

    scoring: ScoringConfig = field(
        default_factory=ScoringConfig
    )
    """综合评分权重配置"""

    smoothness: SmoothnessConfig = field(
        default_factory=SmoothnessConfig
    )
    """表面平滑度检测配置"""

    candidate_generation: CandidateGenerationConfig = field(
        default_factory=CandidateGenerationConfig
    )
    """候选置钉点生成配置"""

    optimization: OptimizationConfig = field(
        default_factory=OptimizationConfig
    )
    """局部优化配置"""

    debug: DebugConfig = field(
        default_factory=DebugConfig
    )
    """调试配置"""

    def __post_init__(self) -> None:
        """
        初始化完成后自动验证所有子配置
        """
        self.validate()

    def validate(self) -> None:
        """
        验证所有子配置的合法性

        Raises
        ------
        ValueError
            如果任何子配置不合法
        """
        self.ct_preprocessing.validate()
        self.neighborhood.validate()
        self.cortex_detection.validate()
        self.scoring.validate()
        self.smoothness.validate()
        self.candidate_generation.validate()
        self.optimization.validate()
        self.debug.validate()

    def summary(self) -> None:
        """
        打印整个配置摘要，方便调试
        """
        print("=" * 45)
        print("AI Config")
        print("=" * 45)
        print(f"Version       : {self.VERSION}")
        print(f"Module        : {self.MODULE_NAME}")
        print(f"Random Seed   : {self.RANDOM_SEED}")
        print()
        print("CT Preprocessing")
        print(f"  HU Range    : {self.ct_preprocessing.HU_MIN} ~ {self.ct_preprocessing.HU_MAX}")
        print(f"  Resample    : {self.ct_preprocessing.RESAMPLE}")
        print(f"  Target Spacing : {self.ct_preprocessing.TARGET_SPACING}")
        print(f"  Normalize   : [{self.ct_preprocessing.NORMALIZE_MIN}, {self.ct_preprocessing.NORMALIZE_MAX}]")
        print()
        print("Neighborhood")
        print(f"  Radius      : {self.neighborhood.RADIUS} voxels")
        print(f"  Spacing     : {self.neighborhood.SPACING} mm")
        print()
        print("Cortex Detection")
        print(f"  HU Threshold : {self.cortex_detection.HU_THRESHOLD}")
        print(f"  Search Radius : {self.cortex_detection.SEARCH_RADIUS} mm")
        print(f"  Min Thickness : {self.cortex_detection.MIN_THICKNESS} mm")
        print(f"  Target Distance : {self.cortex_detection.TARGET_DISTANCE} mm")
        print(f"  Tolerance   : {self.cortex_detection.TOLERANCE} mm")
        print()
        print("Scoring Weights")
        print(f"  HU Density  : {self.scoring.HU_WEIGHT}")
        print(f"  Cortex Dist : {self.scoring.CORTEX_DISTANCE_WEIGHT}")
        print(f"  Stability   : {self.scoring.STABILITY_WEIGHT}")
        print(f"  Smoothness  : {self.scoring.SMOOTHNESS_WEIGHT}")
        print(f"  Anatomical  : {self.scoring.ANATOMICAL_CENTER_WEIGHT}")
        print(f"  Total       : {sum([self.scoring.HU_WEIGHT, self.scoring.CORTEX_DISTANCE_WEIGHT, self.scoring.STABILITY_WEIGHT, self.scoring.SMOOTHNESS_WEIGHT, self.scoring.ANATOMICAL_CENTER_WEIGHT]):.2f}")
        print()
        print("Candidate Generation")
        print(f"  Candidates  : {self.candidate_generation.NUM_CANDIDATES}")
        print(f"  Top Candidates : {self.candidate_generation.TOP_CANDIDATES}")
        print(f"  Surface Samples : {self.candidate_generation.SURFACE_SAMPLES}")
        print(f"  Sphere Radius  : {self.candidate_generation.SPHERE_RADIUS} mm")
        print(f"  Search Radius  : {self.candidate_generation.SEARCH_RADIUS} mm")
        print(f"  Step         : {self.candidate_generation.STEP} mm")
        print(f"  Offset Distance : {self.candidate_generation.OFFSET_DISTANCE} mm")
        print()
        print("Optimization")
        print(f"  Radius      : {self.optimization.RADIUS} mm")
        print(f"  Step        : {self.optimization.STEP} mm")
        print(f"  Iterations  : {self.optimization.ITERATIONS}")
        print()
        print("Debug")
        print(f"  Enabled     : {self.debug.ENABLED}")
        print(f"  Save Result : {self.debug.SAVE_DEBUG_RESULT}")
        print(f"  Save Point  : {self.debug.SAVE_POINT}")
        print("=" * 45)


config = AIConfig()
"""
全局配置实例。
在模块内部通过`from .config import config`访问配置参数。
"""


__all__ = [
    "AIConfig",
    "CTPreprocessingConfig",
    "NeighborhoodConfig",
    "CortexDetectionConfig",
    "ScoringConfig",
    "SmoothnessConfig",
    "CandidateGenerationConfig",
    "OptimizationConfig",
    "DebugConfig",
    "config",
]