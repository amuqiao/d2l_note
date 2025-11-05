from typing import List, Optional, Any, Dict, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import time
from prettytable import PrettyTable
from src.model_visualization.data_models import ModelInfoData
from src.helper_utils.helper_tools_registry import ToolRegistry
from src.utils.log_utils.log_utils import get_logger


# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_visualizer.log", global_level="INFO")


# ==================================================
# ModelInfoData可视化层：负责将标准化数据模型转换为可视化结果
# ==================================================
class BaseModelInfoVisualizer(ABC):
    """模型信息可视化器抽象接口：定义所有可视化器的统一规范"""
    
    @abstractmethod
    def support(self, model_info: ModelInfoData) -> bool:
        """判断当前可视化器是否支持该模型信息数据"""
        pass
    
    @abstractmethod
    def visualize(self, model_info: ModelInfoData, show: bool = True) -> Any:
        """绘制可视化结果（返回绘图对象）"""
        pass


class ModelInfoVisualizerRegistry:
    """模型信息可视化器注册中心：管理所有可视化器，实现自动匹配"""
    
    _visualizers: Dict[str, List[BaseModelInfoVisualizer]] = {"default": []}  # 按命名空间组织可视化器
    
    @classmethod
    def register(cls, visualizer: BaseModelInfoVisualizer, namespace: str = "default"):
        """注册可视化器实例"""
        if namespace not in cls._visualizers:
            cls._visualizers[namespace] = []
        cls._visualizers[namespace].append(visualizer)
    
    @classmethod
    def get_matched_visualizer(
        cls, model_info: ModelInfoData
    ) -> Optional[BaseModelInfoVisualizer]:
        """根据模型信息数据匹配最合适的可视化器"""
        # 首先在model_info的命名空间中查找
        namespace = model_info.namespace
        if namespace in cls._visualizers:
            for vis in cls._visualizers[namespace]:
                if vis.support(model_info):
                    return vis
        
        # 如果在model_info的命名空间中未找到，尝试在默认命名空间中查找
        if namespace != "default" and "default" in cls._visualizers:
            for vis in cls._visualizers["default"]:
                if vis.support(model_info):
                    return vis
        
        return None
    
    @classmethod
    def draw(cls, model_info: ModelInfoData, show: bool = True) -> Any:
        """自动绘制可视化结果的入口方法"""
        vis = cls.get_matched_visualizer(model_info)
        if not vis:
            logger.warning(f"未找到匹配的模型信息可视化器: {model_info.type}")
            return None
        
        try:
            return vis.visualize(model_info, show)
        except Exception as e:
            logger.error(f"模型信息可视化失败 {model_info.path}: {str(e)}")
            return None



# ------------------------------
# 注册可视化器函数（延迟导入以避免循环依赖）
# ------------------------------
def register_visualizers():
    """注册所有可视化器，应在应用程序启动时调用"""
    # 延迟导入以避免循环依赖
    from .model_info_visualizers_impl import ModelSummaryVisualizer, TrainingMetricsVisualizer, ModelComparisonVisualizer
    
    # 注册所有可视化器
    ModelInfoVisualizerRegistry.register(ModelSummaryVisualizer(), namespace="default")
    ModelInfoVisualizerRegistry.register(TrainingMetricsVisualizer(), namespace="default")
    ModelInfoVisualizerRegistry.register(ModelComparisonVisualizer(), namespace="default")


# 在模块导入时不自动注册，而是提供注册函数供调用方使用
# 如果需要自动注册（不推荐），可以取消下面这行的注释
# register_visualizers()
