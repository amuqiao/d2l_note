"""
模型注册统一入口：
1. 集中导入所有模型模块，触发装饰器自动注册
2. 对外暴露核心类和便捷函数，简化外部调用
3. 确保应用启动时所有模型都已正确注册
"""
from typing import Optional, Dict, Any, Type, List
import torch.nn as nn
import logging

# ------------------------------
# 1. 统一注册入口 - 为了避免循环导入，这里不直接导入模型
# ------------------------------
# 注意：模型应通过装饰器自行注册，而不是在此处集中导入
# 这样可以避免循环导入问题，同时保持模块的解耦

# ------------------------------
# 2. 对外暴露核心类，简化外部调用
# ------------------------------
from src.model_center.model_registry import ModelRegistry
from src.model_center.base_registry import ComponentRegistry
from src.model_center.data_loader_registry import DataLoaderRegistry
from src.model_center.config_registry import ConfigRegistry

# ------------------------------
# 3. 提供便捷函数（进一步简化外部使用）
# ------------------------------

def create_model(
    model_name: str,
    namespace: str = "default",
    type: str = "default",
    **kwargs
) -> Optional[nn.Module]:
    """
    便捷创建模型函数：直接调用注册中心创建模型
    
    Args:
        model_name: 模型名称
        namespace: 模型命名空间
        type: 模型类型，默认为"default"
        **kwargs: 模型初始化参数
        
    Returns:
        创建的模型实例，失败则返回None
    """
    try:
        return ModelRegistry.create_model(
            model_name=model_name,
            namespace=namespace,
            type=type,
            **kwargs
        )
    except Exception as e:
        logging.error(f"创建模型 '{model_name}' 失败: {e}")
        return None


def get_model_config(
    model_name: str,
    namespace: str = "default",
    type: str = "default"
) -> Optional[Dict[str, Any]]:
    """
    获取模型配置
    
    Args:
        model_name: 模型名称
        namespace: 模型命名空间
        type: 模型类型，默认为"default"
        
    Returns:
        模型配置字典，失败则返回None
    """
    try:
        return ModelRegistry.get_config(
            model_name=model_name,
            namespace=namespace,
            type=type
        )
    except Exception as e:
        logging.error(f"获取模型 '{model_name}' 配置失败: {e}")
        return None


def get_all_registered_models(namespace: str = None, model_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    获取所有已注册的模型信息
    
    Args:
        namespace: 可选，指定命名空间，None表示所有命名空间
        model_type: 可选，指定模型类型，None表示所有类型
        
    Returns:
        模型信息字典，键为命名空间，值为模型信息列表
    """
    result = {}
    
    # 确定要遍历的命名空间
    namespaces_to_check = [namespace] if namespace and namespace in ModelRegistry._components else \
                         ModelRegistry._components.keys() if not namespace else []
    
    for ns in namespaces_to_check:
        result[ns] = []
        # 确定要遍历的类型
        types_to_check = [model_type] if model_type and model_type in ModelRegistry._components[ns] else \
                         ModelRegistry._components[ns].keys() if not model_type else []
        
        for t in types_to_check:
            for model_name, model_class in ModelRegistry._components[ns][t].items():
                result[ns].append({
                    'name': model_name,
                    'type': t,
                    'class_name': model_class.__name__ if isinstance(model_class, type(model_class)) else model_class.__class__.__name__
                })
    
    return result


def is_model_registered(model_name: str, namespace: str = "default", type: str = "default") -> bool:
    """检查模型是否已注册
    
    Args:
        model_name: 模型名称
        namespace: 模型命名空间
        type: 模型类型，默认为"default"
        
    Returns:
        是否已注册
    """
    return ModelRegistry.is_registered(
        model_name=model_name,
        namespace=namespace,
        type=type
    )

# ------------------------------
# 4. 自动注册机制：简化模型注册流程的装饰器
# ------------------------------

def register_model(
    name: str,
    namespace: str = "default",
    type: str = "general",
    config: Dict[str, Any] = None
):
    """
    自动注册模型的装饰器：简化模型注册流程
    
    Args:
        name: 模型名称
        namespace: 命名空间
        type: 模型类型
        config: 模型默认配置
    
    Returns:
        装饰后的模型类
    """
    def decorator(model_class: Type[nn.Module]):
        try:
            # 使用ModelRegistry的register_model方法注册模型和配置
            ModelRegistry.register_model(
                model_name=name,
                namespace=namespace,
                type=type,
                config=config or {}
            )(model_class)
            
            # 记录注册信息
            logging.debug(f"模型 '{name}' (类型: {type}) 已在命名空间 '{namespace}' 中注册")
        except Exception as e:
            # 改进错误处理，避免装饰器异常影响模型定义
            logging.error(f"注册模型 '{name}' 失败: {e}")
            
        return model_class
    
    return decorator

# ------------------------------
# 5. 初始化函数（确保模型正确注册）
# ------------------------------

def initialize_models():
    """初始化模型系统，确保所有模型都已注册"""
    # 获取所有已注册的模型，验证注册是否成功
    all_models = get_all_registered_models()
    total_models = sum(len(models) for models in all_models.values())
    
    # 打印注册信息，便于调试
    model_info = []
    for ns, models in all_models.items():
        model_names = [f"{m['name']}({m['type']})" for m in models]
        model_info.append(f"命名空间 '{ns}' 包含 {len(models)} 个模型: {', '.join(model_names)}")
    
    logging.info(f"模型系统初始化完成，共注册 {total_models} 个模型")
    for info in model_info:
        logging.debug(info)

# ------------------------------
# 6. 延迟初始化
# ------------------------------
# 注意：不再在模块导入时自动初始化，以避免循环导入问题
# 用户可以在需要时手动调用initialize_models()