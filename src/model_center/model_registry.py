from typing import Dict, List, Optional, Type, Any, Callable
import torch
from torch import nn
from src.model_center.base_registry import ComponentRegistry, logger


class ModelRegistry(ComponentRegistry):
    """模型注册中心：专门用于管理深度学习模型"""
    
    # 存储模型配置的字典，格式：{namespace: {type: {model_name: config}}}
    _model_configs: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {"default": {"default": {}}}
    
    # 存储模型测试函数的字典，格式：{namespace: {type: {model_name: test_func}}}
    _model_test_funcs: Dict[str, Dict[str, Dict[str, Callable]]] = {"default": {"default": {}}}
    
    @classmethod
    def register_model(cls, model_name: str = None, config: Optional[Dict[str, Any]] = None, 
                      namespace: str = "default", type: str = "default", test_func: Optional[Callable] = None):
        """模型注册装饰器：用于注册模型类及其默认配置和测试函数
        
        Args:
            model_name: 模型名称，将作为注册的唯一标识，如果不提供则使用类名
            config: 可选的模型默认配置字典
            namespace: 命名空间，默认为"default"
            type: 模型类型，默认为"default"
            test_func: 可选的模型测试函数
            
        Returns:
            装饰器函数
        """
        def decorator(model_class: Type[nn.Module]):
            # 确定模型名称
            name = model_name if model_name else model_class.__name__
            
            # 注册模型类
            cls.register(model_class, namespace=namespace, type=type, name=name)
            
            # 注册模型配置（如果提供）
            if config:
                if namespace not in cls._model_configs:
                    cls._model_configs[namespace] = {}
                if type not in cls._model_configs[namespace]:
                    cls._model_configs[namespace][type] = {}
                cls._model_configs[namespace][type][name] = config
                logger.debug(f"模型 '{name}' 的配置已在命名空间 '{namespace}' 类型 '{type}' 中注册")
            
            # 注册模型测试函数（如果提供）
            if test_func:
                if namespace not in cls._model_test_funcs:
                    cls._model_test_funcs[namespace] = {}
                if type not in cls._model_test_funcs[namespace]:
                    cls._model_test_funcs[namespace][type] = {}
                cls._model_test_funcs[namespace][type][name] = test_func
                logger.debug(f"模型 '{name}' 的测试函数已在命名空间 '{namespace}' 类型 '{type}' 中注册")
            
            return model_class
            
        return decorator
    
    @classmethod
    def register_config(cls, model_name: str, config: Dict[str, Any], namespace: str = "default", type: str = "default"):
        """注册或更新模型配置
        
        Args:
            model_name: 模型名称
            config: 模型配置字典
            namespace: 命名空间，默认为"default"
            type: 模型类型，默认为"default"
        """
        if namespace not in cls._model_configs:
            cls._model_configs[namespace] = {}
        if type not in cls._model_configs[namespace]:
            cls._model_configs[namespace][type] = {}
        cls._model_configs[namespace][type][model_name] = config
        logger.debug(f"模型 '{model_name}' 的配置已在命名空间 '{namespace}' 类型 '{type}' 中注册")
    
    @classmethod
    def register_test_func(cls, model_name: str, test_func: Callable, namespace: str = "default", type: str = "default"):
        """注册模型专用测试函数
        
        Args:
            model_name: 模型名称
            test_func: 测试函数，接受模型实例和输入尺寸参数
            namespace: 命名空间，默认为"default"
            type: 模型类型，默认为"default"
        """
        if namespace not in cls._model_test_funcs:
            cls._model_test_funcs[namespace] = {}
        if type not in cls._model_test_funcs[namespace]:
            cls._model_test_funcs[namespace][type] = {}
        cls._model_test_funcs[namespace][type][model_name] = test_func
        logger.debug(f"模型 '{model_name}' 的测试函数已在命名空间 '{namespace}' 类型 '{type}' 中注册")
    
    @classmethod
    def create_model(cls, model_name: str, namespace: str = "default", type: str = "default", **kwargs) -> nn.Module:
        """根据模型名称创建模型实例
        
        Args:
            model_name: 模型名称
            namespace: 命名空间，默认为"default"
            type: 模型类型，默认为"default"
            **kwargs: 传递给模型构造函数的参数
            
        Returns:
            模型实例
            
        Raises:
            ValueError: 如果模型名称未注册
        """
        model_class = cls.get(model_name, namespace, type)
        if not model_class:
            registered_models = []
            # 如果指定了type，只列出该类型的模型
            if type:
                registered_models = cls.list_components(namespace, type)
            # 如果未找到或未指定type，列出所有模型
            if not registered_models:
                registered_models = cls.list_components(namespace)
            registered_models_str = ', '.join(registered_models)
            raise ValueError(
                f"不支持的模型类型: {model_name}，命名空间 '{namespace}' 中已注册的模型有: {registered_models_str}"
            )
        
        # 使用提供的参数创建模型实例
        model_instance = model_class(**kwargs)
        logger.debug(f"已创建模型 '{model_name}' (命名空间: '{namespace}', 类型: '{type}') 的实例")
        return model_instance
    
    @classmethod
    def get_config(cls, model_name: str, namespace: str = "default", type: str = "default") -> Optional[Dict[str, Any]]:
        """获取模型的默认配置
        
        Args:
            model_name: 模型名称
            namespace: 命名空间，默认为"default"
            type: 模型类型，默认为"default"
            
        Returns:
            模型配置字典，如果未找到则返回None
        """
        # 首先在指定命名空间和类型中查找
        if (namespace in cls._model_configs and 
            type in cls._model_configs[namespace] and 
            model_name in cls._model_configs[namespace][type]):
            return cls._model_configs[namespace][type][model_name].copy()  # 返回副本避免修改原始配置
        
        # 如果指定命名空间和类型中未找到，尝试在默认命名空间中查找
        if (namespace != "default" and 
            "default" in cls._model_configs and 
            type in cls._model_configs["default"] and 
            model_name in cls._model_configs["default"][type]):
            logger.debug(f"在命名空间 '{namespace}' 类型 '{type}' 中未找到模型 '{model_name}' 的配置，使用默认命名空间中的配置")
            return cls._model_configs["default"][type][model_name].copy()
        
        logger.warning(f"未找到模型 '{model_name}' 的配置（命名空间: '{namespace}', 类型: '{type}'）")
        return None
    
    @classmethod
    def get_test_func(cls, model_name: str, namespace: str = "default", type: str = "default") -> Optional[Callable]:
        """获取模型的测试函数
        
        Args:
            model_name: 模型名称
            namespace: 命名空间，默认为"default"
            type: 模型类型，默认为"default"
            
        Returns:
            测试函数，如果未找到则返回None
        """
        # 首先在指定命名空间和类型中查找
        if (namespace in cls._model_test_funcs and 
            type in cls._model_test_funcs[namespace] and 
            model_name in cls._model_test_funcs[namespace][type]):
            return cls._model_test_funcs[namespace][type][model_name]
        
        # 如果指定命名空间和类型中未找到，尝试在默认命名空间中查找
        if (namespace != "default" and 
            "default" in cls._model_test_funcs and 
            type in cls._model_test_funcs["default"] and 
            model_name in cls._model_test_funcs["default"][type]):
            logger.debug(f"在命名空间 '{namespace}' 类型 '{type}' 中未找到模型 '{model_name}' 的测试函数，使用默认命名空间中的测试函数")
            return cls._model_test_funcs["default"][type][model_name]
        
        # 如果没有注册专用测试函数，返回None
        return None
    
    @classmethod
    def is_registered(cls, model_name: str, namespace: str = "default", type: str = "default") -> bool:
        """检查模型是否已注册
        
        Args:
            model_name: 模型名称
            namespace: 命名空间，默认为"default"
            type: 模型类型，默认为"default"
            
        Returns:
            是否已注册
        """
        # 检查模型类是否已注册
        if cls.get(model_name, namespace, type):
            return True
        return False
    
    @classmethod
    def get_model_info(cls, model_name: str, namespace: str = "default", type: str = "default") -> Dict[str, Any]:
        """获取模型的完整信息
        
        Args:
            model_name: 模型名称
            namespace: 命名空间，默认为"default"
            type: 模型类型，默认为"default"
            
        Returns:
            包含模型类、配置和测试函数的字典
        """
        return {
            "model_class": cls.get(model_name, namespace, type),
            "config": cls.get_config(model_name, namespace, type),
            "test_func": cls.get_test_func(model_name, namespace, type),
            "is_registered": cls.is_registered(model_name, namespace, type)
        }