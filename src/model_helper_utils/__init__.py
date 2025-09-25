"""model_helper_utils包初始化文件

此文件负责在包加载时完成解析器和可视化器的注册工作，实现控制反转设计，
避免注册中心与具体实现之间的循环引用问题。
"""

# 导入注册中心
from .metric_parsers import MetricParserRegistry

# 导入具体解析器实现
from .parsers_impl import EpochMetricsParser, FullMetricsParser, ConfusionMatrixParser


# ==================================================
# 解析器注册 - 实现控制反转设计
# ==================================================
# 注册中心只负责管理，不关心具体实现
# 具体实现在外部定义，并通过register方法手动注册

# 注册epoch指标解析器
MetricParserRegistry.register(EpochMetricsParser())

# 注册完整指标解析器
MetricParserRegistry.register(FullMetricsParser())

# 注册混淆矩阵解析器
MetricParserRegistry.register(ConfusionMatrixParser())

# 导入可视化器注册中心
from .visualizers import VisualizerRegistry
from .visualizers_impl import CurveVisualizer, ConfusionMatrixVisualizer

VisualizerRegistry.register(CurveVisualizer())
VisualizerRegistry.register(ConfusionMatrixVisualizer())

# ==================================================
# 包导出 - 提供统一的公共API
# ==================================================
__all__ = [
    # 核心数据结构
    'MetricData',
    
    # 注册中心
    'MetricParserRegistry',
    
    # 抽象接口
    'BaseMetricParser',
    'BaseVisualizer'
]

# 按需导入导出的组件
from .metric_parsers import MetricData, BaseMetricParser
from .visualizers import BaseVisualizer