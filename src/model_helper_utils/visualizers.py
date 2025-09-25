import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Type


# 导入自定义日志模块
from src.utils.log_utils import get_logger

# 导入工具注册中心
from src.helper_utils.helper_tools_registry import ToolRegistry

# 导入MetricData类
from src.model_helper_utils.metric_parsers import MetricData

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_analysis.log", global_level="INFO")

# ==================================================
# 可视化核心层：通用可视化接口与实现
# ==================================================
class BaseVisualizer(ABC):
    """可视化器抽象接口：定义所有可视化器的统一规范"""
    @abstractmethod
    def support(self, metric_data: MetricData) -> bool:
        """判断当前可视化器是否支持该指标数据"""
        pass

    @abstractmethod
    def visualize(self, metric_data: MetricData, show: bool = True) -> Any:
        """绘制可视化结果（返回绘图对象）"""
        pass


class VisualizerRegistry:
    """可视化器注册中心：管理所有可视化器，实现自动匹配"""
    _visualizers: List[BaseVisualizer] = []

    @classmethod
    def register(cls, visualizer: BaseVisualizer):
        """注册可视化器实例"""
        cls._visualizers.append(visualizer)


    @classmethod
    def get_matched_visualizer(cls, metric_data: MetricData) -> Optional[BaseVisualizer]:
        """根据指标数据匹配最合适的可视化器"""
        for vis in cls._visualizers:
            if vis.support(metric_data):
                return vis
        return None

    @classmethod
    def draw(cls, metric_data: MetricData, show: bool = True) -> Any:
        """自动绘制可视化结果的入口方法"""
        vis = cls.get_matched_visualizer(metric_data)
        if not vis:
            logger.warning(f"未找到匹配的可视化器: {metric_data.metric_type}")
            return None
            
        try:
            return vis.visualize(metric_data, show)
        except Exception as e:
            logger.error(f"可视化失败 {metric_data.source_path}: {str(e)}")
            return None
