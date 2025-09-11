"""模型注册中心：提供模型注册、创建和配置管理功能"""
import torch
from typing import Dict, Type, Any, Optional, Callable
from torch import nn


class ModelRegistry:
    """模型注册中心：使用工厂模式统一管理所有深度学习模型"""
    
    # 存储模型类的字典，键为模型名称，值为模型类
    _model_registry: Dict[str, Type[nn.Module]] = {}
    
    # 存储模型配置的字典，键为模型名称，值为配置字典
    _model_configs: Dict[str, Dict[str, Any]] = {}
    
    # 存储模型测试函数的字典，键为模型名称，值为测试函数
    _model_test_funcs: Dict[str, Callable] = {}
    
    @classmethod
    def register_model(cls, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        模型注册装饰器：用于注册模型类及其默认配置
        
        Args:
            model_name: 模型名称，将作为注册的唯一标识
            config: 可选的模型默认配置字典
        
        Returns:
            装饰器函数
        """
        def decorator(model_class: Type[nn.Module]):
            # 注册模型类
            cls._model_registry[model_name] = model_class
            
            # 注册模型配置（如果提供）
            if config:
                cls._model_configs[model_name] = config
            
            # 默认测试函数：使用通用网络形状测试
            from src.utils.network_utils import NetworkUtils
            cls._model_test_funcs[model_name] = lambda net, input_size: NetworkUtils.test_network_shape(
                net, input_size=input_size
            )
            
            # 不打印注册信息，避免过多输出
            return model_class
        
        return decorator
        
    @classmethod
    def auto_register_model(cls, model_class: Type[nn.Module], model_name: Optional[str] = None, 
                          config: Optional[Dict[str, Any]] = None):
        """
        自动注册模型（适用于在运行时动态注册模型）
        
        Args:
            model_class: 模型类
            model_name: 可选的模型名称，如果未提供则使用类名
            config: 可选的模型配置
        """
        if model_name is None:
            model_name = model_class.__name__
        
        cls._model_registry[model_name] = model_class
        
        if config:
            cls._model_configs[model_name] = config
        
        # 设置默认测试函数
        from src.utils.network_utils import NetworkUtils
        cls._model_test_funcs[model_name] = lambda net, input_size: NetworkUtils.test_network_shape(
            net, input_size=input_size
        )
    
    @classmethod
    def register_config(cls, model_name: str, config: Dict[str, Any]):
        """
        注册或更新模型配置
        
        Args:
            model_name: 模型名称
            config: 模型配置字典
        """
        cls._model_configs[model_name] = config
        
    @classmethod
    def register_test_func(cls, model_name: str, test_func: Callable):
        """
        注册模型专用测试函数
        
        Args:
            model_name: 模型名称
            test_func: 测试函数，接受模型实例和输入尺寸参数
        """
        cls._model_test_funcs[model_name] = test_func
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> nn.Module:
        """
        根据模型名称创建模型实例
        
        Args:
            model_name: 模型名称
            **kwargs: 传递给模型构造函数的参数
        
        Returns:
            模型实例
        
        Raises:
            ValueError: 如果模型名称未注册
        """
        if model_name not in cls._model_registry:
            registered_models = ', '.join(cls._model_registry.keys())
            raise ValueError(
                f"不支持的模型类型: {model_name}，已注册的模型有: {registered_models}"
            )
        
        model_class = cls._model_registry[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def get_config(cls, model_name: str) -> Dict[str, Any]:
        """
        获取模型的默认配置
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型配置字典
        
        Raises:
            ValueError: 如果模型名称未注册
        """
        if model_name not in cls._model_configs:
            raise ValueError(f"未找到模型 '{model_name}' 的配置")
        
        return cls._model_configs[model_name].copy()  # 返回副本避免修改原始配置
    
    @classmethod
    def get_test_func(cls, model_name: str) -> Callable:
        """
        获取模型的测试函数
        
        Args:
            model_name: 模型名称
        
        Returns:
            测试函数
        
        Raises:
            ValueError: 如果模型名称未注册
        """
        if model_name not in cls._model_test_funcs:
            # 如果没有注册专用测试函数，返回通用测试函数
            from src.utils.network_utils import NetworkUtils
            return lambda net, input_size: NetworkUtils.test_network_shape(net, input_size=input_size)
        
        return cls._model_test_funcs[model_name]
    
    @classmethod
    def list_models(cls) -> list:
        """
        列出所有已注册的模型名称
        
        Returns:
            模型名称列表
        """
        return list(cls._model_registry.keys())
    
    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """
        检查模型是否已注册
        
        Args:
            model_name: 模型名称
        
        Returns:
            是否已注册
        """
        return model_name in cls._model_registry