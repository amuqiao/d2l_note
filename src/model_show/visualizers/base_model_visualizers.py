from abc import ABC, abstractmethod
import os
from src.model_show.data_models import ModelInfoData
from src.utils.log_utils import get_logger
from typing import Dict, List, Optional, Type, Any


# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_visualizer.log", global_level="INFO")


class BaseModelVisualizer(ABC):
    """模型可视化器抽象接口：定义所有可视化器的统一规范"""
    
    # 可视化器优先级，数值越高优先级越高
    priority: int = 50
    
    @abstractmethod
    def support(self, model_info: ModelInfoData, namespace: str = "default") -> bool:
        """判断当前可视化器是否支持该模型信息
        
        Args:
            model_info: 模型信息数据
            namespace: 命名空间，默认为"default"
        
        Returns:
            bool: 是否支持该模型信息
        """
        pass
    
    @abstractmethod
    def visualize(self, model_info: ModelInfoData, namespace: str = "default") -> Dict[str, Any]:
        """将模型信息可视化为指定格式
        
        Args:
            model_info: 模型信息数据
            namespace: 命名空间，默认为"default"
        
        Returns:
            Dict[str, Any]: 可视化结果
        """
        pass


class ModelVisualizerRegistry:
    """模型可视化器注册中心：管理所有可视化器，实现自动匹配"""
    
    _visualizers: Dict[str, List[BaseModelVisualizer]] = {"default": []}  # 按命名空间组织可视化器
    
    @classmethod
    def register(cls, visualizer_cls=None, namespace: str = "default"):
        """注册可视化器类（支持直接调用和装饰器用法）"""
        # 如果visualizer_cls为None，说明是作为带参数的装饰器使用
        if visualizer_cls is None:
            # 返回一个可以接受visualizer_cls的装饰器函数
            def decorator(cls_to_register):
                return cls.register(cls_to_register, namespace)
            return decorator
        
        # 正常注册逻辑
        if namespace not in cls._visualizers:
            cls._visualizers[namespace] = []
        
        # 实例化可视化器并添加到注册表
        visualizer_instance = visualizer_cls()
        cls._visualizers[namespace].append(visualizer_instance)
        
        # 按优先级排序
        cls._visualizers[namespace].sort(key=lambda v: v.priority, reverse=True)
        
        return visualizer_cls
    
    @classmethod
    def unregister(cls, visualizer_cls: Type[BaseModelVisualizer], namespace: str = "default"):
        """注销可视化器"""
        if namespace in cls._visualizers:
            cls._visualizers[namespace] = [
                v for v in cls._visualizers[namespace] 
                if not isinstance(v, visualizer_cls)
            ]
    
    @classmethod
    def get_matched_visualizer(cls, model_info: ModelInfoData, namespace: str = "default") -> Optional[BaseModelVisualizer]:
        """根据模型信息匹配最合适的可视化器（考虑优先级）"""
        # 首先在指定命名空间中查找
        if namespace in cls._visualizers:
            for visualizer in cls._visualizers[namespace]:
                if visualizer.support(model_info, namespace):
                    return visualizer
        
        # 如果指定命名空间中未找到，尝试在默认命名空间中查找
        if namespace != "default" and "default" in cls._visualizers:
            for visualizer in cls._visualizers["default"]:
                if visualizer.support(model_info, "default"):
                    return visualizer
        
        # 尝试在所有命名空间中查找，只要可视化器支持该模型信息
        for ns, visualizers in cls._visualizers.items():
            if ns == namespace:
                continue  # 已经在指定命名空间中查找过了
            for visualizer in visualizers:
                if visualizer.support(model_info, ns):
                    return visualizer
        
        return None
    
    @classmethod
    def visualize_model(cls, model_info: ModelInfoData, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """自动可视化模型的入口方法"""
        if not model_info:
            logger.warning(f"模型信息为空")
            return None
            
        visualizer = cls.get_matched_visualizer(model_info, namespace)
        if not visualizer:
            logger.warning(f"未找到匹配的模型可视化器: {model_info.name}")
            return None
        
        try:
            return visualizer.visualize(model_info, namespace)
        except Exception as e:
            logger.error(f"可视化模型失败 {model_info.name}: {str(e)}")
            return None