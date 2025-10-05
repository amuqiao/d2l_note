import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any, Callable

from src.utils.log_utils.log_utils import get_logger


# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_center.log", global_level="INFO")


class BaseRegistry(ABC):
    """注册中心基类：定义通用的注册和管理接口"""
    
    @classmethod
    @abstractmethod
    def register(cls, component_cls=None, namespace: str = "default"):
        """注册组件"""
        pass
        
    @classmethod
    @abstractmethod
    def unregister(cls, component_cls: Type, namespace: str = "default"):
        """注销组件"""
        pass
        
    @classmethod
    @abstractmethod
    def get(cls, name: str, namespace: str = "default") -> Optional[Any]:
        """获取组件"""
        pass
        
    @classmethod
    @abstractmethod
    def list_components(cls, namespace: str = None) -> List[str]:
        """列出所有组件"""
        pass


class ComponentRegistry(BaseRegistry):
    """组件注册中心：通用的组件注册和管理实现"""
    
    # 存储组件的字典，格式：{namespace: {component_name: component}}
    _components: Dict[str, Dict[str, Any]] = {"default": {}}
    
    @classmethod
    def register(cls, component_cls=None, namespace: str = "default", name: str = None):
        """注册组件类或实例（支持直接调用和装饰器用法）
        
        Args:
            component_cls: 组件类或实例
            namespace: 命名空间，默认为"default"
            name: 组件名称，如果不提供则使用类名或实例类名
            
        Returns:
            注册的组件类或装饰器函数
        """
        # 如果component_cls为None，说明是作为带参数的装饰器使用
        if component_cls is None:
            # 返回一个可以接受component_cls的装饰器函数
            def decorator(cls_to_register):
                return cls.register(cls_to_register, namespace, name)
            return decorator
        
        # 确保命名空间存在
        if namespace not in cls._components:
            cls._components[namespace] = {}
        
        # 确定组件名称
        component_name = name
        if component_name is None:
            # 如果是类，使用类名
            if isinstance(component_cls, type):
                component_name = component_cls.__name__
            # 如果是实例，使用实例类名
            else:
                component_name = component_cls.__class__.__name__
        
        # 注册组件
        cls._components[namespace][component_name] = component_cls
        logger.debug(f"组件 '{component_name}' 已在命名空间 '{namespace}' 中注册")
        
        return component_cls
    
    @classmethod
    def unregister(cls, component_name: str, namespace: str = "default"):
        """注销指定名称的组件
        
        Args:
            component_name: 组件名称
            namespace: 命名空间，默认为"default"
        """
        if namespace in cls._components and component_name in cls._components[namespace]:
            del cls._components[namespace][component_name]
            logger.debug(f"组件 '{component_name}' 已从命名空间 '{namespace}' 中注销")
    
    @classmethod
    def get(cls, component_name: str, namespace: str = "default") -> Optional[Any]:
        """获取指定名称的组件
        
        Args:
            component_name: 组件名称
            namespace: 命名空间，默认为"default"
            
        Returns:
            组件类或实例，如果未找到则返回None
        """
        # 首先在指定命名空间中查找
        if namespace in cls._components and component_name in cls._components[namespace]:
            return cls._components[namespace][component_name]
        
        # 如果指定命名空间中未找到，尝试在默认命名空间中查找
        if namespace != "default" and "default" in cls._components and component_name in cls._components["default"]:
            logger.debug(f"在命名空间 '{namespace}' 中未找到组件 '{component_name}'，使用默认命名空间中的组件")
            return cls._components["default"][component_name]
        
        logger.warning(f"未找到组件 '{component_name}'（命名空间: '{namespace}'）")
        return None
    
    @classmethod
    def list_components(cls, namespace: str = None) -> List[str]:
        """列出所有组件名称
        
        Args:
            namespace: 可选，指定命名空间，None表示所有命名空间
            
        Returns:
            组件名称列表
        """
        if namespace:
            # 列出指定命名空间中的组件
            if namespace in cls._components:
                return list(cls._components[namespace].keys())
            return []
        
        # 列出所有命名空间中的组件
        all_components = []
        for ns_components in cls._components.values():
            all_components.extend(ns_components.keys())
        return all_components
    
    @classmethod
    def get_namespaces(cls) -> List[str]:
        """获取所有可用的命名空间
        
        Returns:
            命名空间列表
        """
        return list(cls._components.keys())
    
    @classmethod
    def clear(cls, namespace: str = None):
        """清空指定命名空间或所有命名空间中的组件
        
        Args:
            namespace: 可选，指定命名空间，None表示所有命名空间
        """
        if namespace:
            if namespace in cls._components:
                cls._components[namespace] = {}
                logger.debug(f"已清空命名空间 '{namespace}' 中的所有组件")
        else:
            cls._components = {"default": {}}
            logger.debug("已清空所有命名空间中的组件")