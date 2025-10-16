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
        
    @classmethod
    @abstractmethod
    def get_namespaces(cls) -> List[str]:
        """获取所有可用的命名空间"""
        pass
        
    @classmethod
    @abstractmethod
    def clear(cls, namespace: str = None, type: str = None):
        """清空指定命名空间、类型或所有组件"""
        pass


class ComponentRegistry(BaseRegistry):
    """组件注册中心：通用的组件注册和管理实现"""
    
    # 存储组件的字典，格式：{namespace: {type: {component_name: component}}}
    _components: Dict[str, Dict[str, Dict[str, Any]]] = {"default": {"default": {}}}
    
    @classmethod
    def register(cls, component_cls=None, namespace: str = "default", name: str = None, type: str = "default"):
        """注册组件类或实例（支持直接调用和装饰器用法）
        
        Args:
            component_cls: 组件类或实例
            namespace: 命名空间，默认为"default"
            name: 组件名称，如果不提供则使用类名或实例类名
            type: 组件类型，默认为"default"
            
        Returns:
            注册的组件类或装饰器函数
        """
        # 如果component_cls为None，说明是作为带参数的装饰器使用
        if component_cls is None:
            # 返回一个可以接受component_cls的装饰器函数
            def decorator(cls_to_register):
                return cls.register(cls_to_register, namespace, name, type)
            return decorator
        
        # 确保命名空间存在
        if namespace not in cls._components:
            cls._components[namespace] = {}
            
        # 确保类型存在
        if type not in cls._components[namespace]:
            cls._components[namespace][type] = {}
        
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
        cls._components[namespace][type][component_name] = component_cls
        logger.debug(f"组件 '{component_name}' 已在命名空间 '{namespace}' 的类型 '{type}' 中注册")
        
        return component_cls
    
    @classmethod
    def unregister(cls, component_name: str, namespace: str = "default", type: str = None):
        """注销指定名称的组件
        
        Args:
            component_name: 组件名称
            namespace: 命名空间，默认为"default"
            type: 组件类型，如果为None则在所有类型中查找并注销
        """
        if type is not None:
            if namespace in cls._components and type in cls._components[namespace] and component_name in cls._components[namespace][type]:
                del cls._components[namespace][type][component_name]
                logger.debug(f"组件 '{component_name}' 已从命名空间 '{namespace}' 的类型 '{type}' 中注销")
        else:
            # 在所有类型中查找并注销
            if namespace in cls._components:
                for t in list(cls._components[namespace].keys()):
                    if component_name in cls._components[namespace][t]:
                        del cls._components[namespace][t][component_name]
                        logger.debug(f"组件 '{component_name}' 已从命名空间 '{namespace}' 的类型 '{t}' 中注销")
    
    @classmethod
    def get(cls, component_name: str, namespace: str = "default", type: str = None) -> Optional[Any]:
        """获取指定名称的组件
        
        Args:
            component_name: 组件名称
            namespace: 命名空间，默认为"default"
            type: 组件类型，如果为None则在所有类型中查找
            
        Returns:
            组件类或实例，如果未找到则返回None
        """
        # 如果指定了类型，在特定类型中查找
        if type is not None:
            if namespace in cls._components and type in cls._components[namespace] and component_name in cls._components[namespace][type]:
                return cls._components[namespace][type][component_name]
            
            # 如果在特定类型中未找到，且不是默认命名空间，尝试在默认命名空间的相同类型中查找
            if namespace != "default" and "default" in cls._components and type in cls._components["default"] and component_name in cls._components["default"][type]:
                logger.debug(f"在命名空间 '{namespace}' 的类型 '{type}' 中未找到组件 '{component_name}'，使用默认命名空间中的组件")
                return cls._components["default"][type][component_name]
        else:
            # 在所有类型中查找
            if namespace in cls._components:
                for t in cls._components[namespace]:
                    if component_name in cls._components[namespace][t]:
                        return cls._components[namespace][t][component_name]
            
            # 如果在指定命名空间的所有类型中未找到，且不是默认命名空间，尝试在默认命名空间中查找
            if namespace != "default" and "default" in cls._components:
                for t in cls._components["default"]:
                    if component_name in cls._components["default"][t]:
                        logger.debug(f"在命名空间 '{namespace}' 中未找到组件 '{component_name}'，使用默认命名空间中的组件")
                        return cls._components["default"][t][component_name]
        
        logger.warning(f"未找到组件 '{component_name}'（命名空间: '{namespace}'，类型: '{type}'）")
        return None
    
    @classmethod
    def list_components(cls, namespace: str = None, type: str = None) -> List[str]:
        """列出所有组件名称
        
        Args:
            namespace: 可选，指定命名空间，None表示所有命名空间
            type: 可选，指定类型，None表示所有类型
            
        Returns:
            组件名称列表
        """
        if namespace:
            # 列出指定命名空间中的组件
            if namespace in cls._components:
                if type:
                    # 列出指定命名空间和类型中的组件
                    if type in cls._components[namespace]:
                        return list(cls._components[namespace][type].keys())
                    return []
                else:
                    # 列出指定命名空间中所有类型的组件
                    all_components = []
                    for t_components in cls._components[namespace].values():
                        all_components.extend(t_components.keys())
                    return all_components
            return []
        
        # 列出所有命名空间中的组件
        all_components = []
        for ns_components in cls._components.values():
            if type:
                # 列出所有命名空间中指定类型的组件
                if type in ns_components:
                    all_components.extend(ns_components[type].keys())
            else:
                # 列出所有命名空间中所有类型的组件
                for t_components in ns_components.values():
                    all_components.extend(t_components.keys())
        return all_components
    
    @classmethod
    def get_namespaces(cls) -> List[str]:
        """获取所有可用的命名空间
        
        Returns:
            命名空间列表
        """
        return list(cls._components.keys())
    
    @classmethod
    def clear(cls, namespace: str = None, type: str = None): 
        """清空指定命名空间、类型或所有组件
        
        Args:
            namespace: 可选，指定命名空间，None表示所有命名空间
            type: 可选，指定类型，None表示所有类型
        """
        if namespace:
            if namespace in cls._components:
                if type:
                    # 清空指定命名空间和类型中的组件
                    if type in cls._components[namespace]:
                        cls._components[namespace][type] = {}
                        logger.debug(f"已清空命名空间 '{namespace}' 的类型 '{type}' 中的所有组件")
                else:
                    # 清空指定命名空间中的所有组件
                    cls._components[namespace] = {"default": {}}
                    logger.debug(f"已清空命名空间 '{namespace}' 中的所有组件")
        else:
            # 清空所有命名空间中的组件
            cls._components = {"default": {"default": {}}}
            logger.debug("已清空所有命名空间中的组件")