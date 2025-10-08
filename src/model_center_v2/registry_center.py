from typing import Dict, Any, Tuple, Optional, Callable
from abc import ABC, abstractmethod

# 抽象基类
class AbstractRegistry(ABC):
    @abstractmethod
    def register(self, namespace: str, type_: str, name: str, component: Any = None) -> Any:
        """注册组件，支持直接调用和装饰器用法"""
        pass

    @abstractmethod
    def get(self, namespace: str, type_: str, name: str) -> Any:
        """获取组件"""
        pass

    @abstractmethod
    def list_components(self, namespace: Optional[str] = None, type_: Optional[str] = None) -> Dict[Tuple[str, str, str], Any]:
        """列出组件"""
        pass

    @abstractmethod
    def unregister(self, namespace: str, type_: str, name: str) -> None:
        """注销组件"""
        pass


# 重命名为ComponentRegistry，并支持两种注册方式
class ComponentRegistry(AbstractRegistry):
    def __init__(self):
        self._registry: Dict[Tuple[str, str, str], Any] = {}
    
    def register(self, namespace: str, type_: str, name: str, component: Any = None) -> Any:
        """
        支持两种用法：
        1. 直接调用：register(namespace, type_, name, component)
        2. 装饰器用法：@register(namespace, type_, name)
        """
        # 如果未提供component，则返回一个装饰器
        if component is None:
            def decorator(component_cls: Any) -> Any:
                self._register_impl(namespace, type_, name, component_cls)
                return component_cls  # 返回类本身，不影响其使用
            return decorator
        # 否则直接注册组件
        else:
            self._register_impl(namespace, type_, name, component)
            return component
    
    def _register_impl(self, namespace: str, type_: str, name: str, component: Any) -> None:
        """实际执行注册的内部方法"""
        key = (namespace, type_, name)
        if key in self._registry:
            # 组件已存在时不再重复注册和打印信息
            return
        self._registry[key] = component
        # 可选：如需保留注册信息，可以添加日志级别控制或使用日志库而非直接打印到控制台
        # print(f"已注册组件: namespace={namespace}, type={type_}, name={name}")
    
    def get(self, namespace: str, type_: str, name: str) -> Any:
        key = (namespace, type_, name)
        if key not in self._registry:
            raise KeyError(f"组件不存在: namespace={namespace}, type={type_}, name={name}")
        return self._registry[key]
    
    def list_components(self, namespace: Optional[str] = None, type_: Optional[str] = None) -> Dict[Tuple[str, str, str], Any]:
        result = {}
        for key, component in self._registry.items():
            ns, t, n = key
            if (namespace is None or ns == namespace) and (type_ is None or t == type_):
                result[key] = component
        return result
    
    def unregister(self, namespace: str, type_: str, name: str) -> None:
        key = (namespace, type_, name)
        if key not in self._registry:
            raise KeyError(f"组件不存在，无法注销: namespace={namespace}, type={type_}, name={name}")
        del self._registry[key]
        print(f"已注销组件: namespace={namespace}, type={type_}, name={name}")


# 数据加载注册中心
class DataLoaderRegistry(ComponentRegistry):
    def register_data_loader(self, namespace: str, name: str, data_loader: Any = None) -> Any:
        if data_loader is None:
            def decorator(cls: Any) -> Any:
                self.register(namespace, "data_loader", name, cls)
                return cls
            return decorator
        else:
            self.register(namespace, "data_loader", name, data_loader)
            return data_loader
    
    def get_data_loader(self, namespace: str, name: str) -> Any:
        return self.get(namespace, "data_loader", name)
    
    def unregister_data_loader(self, namespace: str, name: str) -> None:
        self.unregister(namespace, "data_loader", name)


# 模型注册中心
class ModelRegistry(ComponentRegistry):
    def register_model(self, namespace: str, name: str, model_type: str, model: Any = None) -> Any:
        full_name = f"{model_type}::{name}"
        if model is None:
            def decorator(cls: Any) -> Any:
                self.register(namespace, "model", full_name, cls)
                # 移除重复的打印语句
                return cls
            return decorator
        else:
            self.register(namespace, "model", full_name, model)
            # 移除重复的打印语句
            return model
    
    def get_model(self, namespace: str, name: str, model_type: str) -> Any:
        full_name = f"{model_type}::{name}"
        return self.get(namespace, "model", full_name)
    
    def unregister_model(self, namespace: str, name: str, model_type: str) -> None:
        full_name = f"{model_type}::{name}"
        self.unregister(namespace, "model", full_name)
    

# 配置注册中心
class ConfigRegistry(ComponentRegistry):
    def register_config(self, namespace: str, name: str, config: Any = None) -> Any:
        if config is None:
            def decorator(config_obj: Any) -> Any:
                self.register(namespace, "config", name, config_obj)
                return config_obj
            return decorator
        else:
            self.register(namespace, "config", name, config)
            return config
    
    def get_config(self, namespace: str, name: str) -> Any:
        return self.get(namespace, "config", name)
    
    def unregister_config(self, namespace: str, name: str) -> None:
        self.unregister(namespace, "config", name)


# 训练器注册中心
class TrainerRegistry(ComponentRegistry):
    def register_trainer(self, namespace: str, name: str, trainer: Any = None) -> Any:
        if trainer is None:
            def decorator(cls: Any) -> Any:
                self.register(namespace, "trainer", name, cls)
                return cls
            return decorator
        else:
            self.register(namespace, "trainer", name, trainer)
            return trainer
    
    def get_trainer(self, namespace: str, name: str) -> Any:
        return self.get(namespace, "trainer", name)
    
    def unregister_trainer(self, namespace: str, name: str) -> None:
        self.unregister(namespace, "trainer", name)


# 预测器注册中心（独立拆分）
class PredictorRegistry(ComponentRegistry):
    def register_predictor(self, namespace: str, name: str, predictor: Any = None) -> Any:
        if predictor is None:
            def decorator(cls: Any) -> Any:
                self.register(namespace, "predictor", name, cls)
                return cls
            return decorator
        else:
            self.register(namespace, "predictor", name, predictor)
            return predictor
    
    def get_predictor(self, namespace: str, name: str) -> Any:
        return self.get(namespace, "predictor", name)
    
    def unregister_predictor(self, namespace: str, name: str) -> None:
        self.unregister(namespace, "predictor", name)

