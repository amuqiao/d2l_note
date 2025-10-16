"""
配置注册统一入口：
1. 集中导入所有配置模块，触发装饰器自动注册
2. 对外暴露核心类和便捷函数，简化外部调用
3. 确保应用启动时所有配置都已正确注册
"""
from typing import Optional, Dict, Any, Type, List, Union, Callable
import logging

# ------------------------------
# 1. 统一注册入口 - 为了避免循环导入，这里不直接导入配置
# ------------------------------
# 注意：配置应通过装饰器自行注册，而不是在此处集中导入
# 这样可以避免循环导入问题，同时保持模块的解耦

# ------------------------------
# 2. 对外暴露核心类，简化外部调用
# ------------------------------
from .config_registry import ConfigRegistry

# ------------------------------
# 3. 提供便捷函数（进一步简化外部使用）
# ------------------------------

def get_config(
    config_name: str,
    namespace: str = "default",
    file_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷获取配置函数：直接调用注册中心获取配置
    
    Args:
        config_name: 配置名称
        namespace: 配置命名空间
        file_path: 可选的配置文件路径
        **kwargs: 要覆盖的配置项
        
    Returns:
        配置字典
    """
    try:
        return ConfigRegistry.get_config(
            config_name=config_name,
            namespace=namespace,
            file_path=file_path,
            **kwargs
        )
    except Exception as e:
        logging.error(f"获取配置 '{config_name}' 失败: {e}")
        return kwargs


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    便捷加载配置文件函数
    
    Args:
        file_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        return ConfigRegistry.load_config_from_file(file_path)
    except Exception as e:
        logging.error(f"从文件 '{file_path}' 加载配置失败: {e}")
        return {}


def merge_configs(*config_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    便捷合并配置函数
    
    Args:
        *config_dicts: 要合并的配置字典
        
    Returns:
        合并后的配置字典
    """
    return ConfigRegistry.merge_configs(*config_dicts)


def get_default_config(namespace: str = "default") -> Dict[str, Any]:
    """
    获取默认配置模板
    
    Args:
        namespace: 命名空间，默认为"default"
        
    Returns:
        默认配置字典
    """
    return ConfigRegistry.get_default_config(namespace)


def get_all_registered_configs(namespace: str = None) -> Dict[str, List[str]]:
    """
    获取所有已注册的配置信息
    
    Args:
        namespace: 可选，指定命名空间，None表示所有命名空间
        
    Returns:
        配置信息字典，键为命名空间，值为配置名称列表
    """
    result = {}
    
    # 确定要遍历的命名空间
    namespaces_to_check = [namespace] if namespace and namespace in ConfigRegistry._components else \
                         ConfigRegistry._components.keys() if not namespace else []
    
    for ns in namespaces_to_check:
        result[ns] = list(ConfigRegistry._components[ns].keys())
    
    return result


def is_config_registered(config_name: str, namespace: str = "default") -> bool:
    """检查配置是否已注册
    
    Args:
        config_name: 配置名称
        namespace: 配置命名空间
        
    Returns:
        是否已注册
    """
    return ConfigRegistry.get(config_name, namespace=namespace) is not None

# ------------------------------
# 4. 自动注册机制：简化配置注册流程的装饰器
# ------------------------------

def register_config(
    name: str = None,
    namespace: str = "default",
    template: Optional[Dict[str, Any]] = None,
    file_path: Optional[str] = None
):
    """
    自动注册配置的装饰器：简化配置注册流程
    
    Args:
        name: 配置名称，如果不提供则使用类名或函数名
        namespace: 命名空间
        template: 可选的配置默认模板
        file_path: 可选的配置文件路径
    
    Returns:
        装饰后的配置类或函数
    """
    def decorator(config_source: Union[Callable, Any]):
        try:
            # 使用ConfigRegistry的register_config方法注册配置
            ConfigRegistry.register_config(
                config_name=name,
                namespace=namespace,
                template=template,
                file_path=file_path
            )(config_source)
            
            # 记录注册信息
            config_name_final = name if name else getattr(config_source, '__name__', str(config_source))
            logging.debug(f"配置 '{config_name_final}' 已在命名空间 '{namespace}' 中注册")
        except Exception as e:
            # 改进错误处理，避免装饰器异常影响配置定义
            logging.error(f"注册配置失败: {e}")
            
        return config_source
    
    return decorator


def register_template(
    config_name: str,
    template: Dict[str, Any],
    namespace: str = "default"
):
    """
    便捷注册配置模板函数
    
    Args:
        config_name: 配置名称
        template: 配置模板字典
        namespace: 命名空间
    
    Returns:
        None
    """
    ConfigRegistry.register_template(
        config_name=config_name,
        template=template,
        namespace=namespace
    )


def register_file(
    config_name: str,
    file_path: str,
    namespace: str = "default"
):
    """
    便捷注册配置文件路径函数
    
    Args:
        config_name: 配置名称
        file_path: 配置文件路径
        namespace: 命名空间
    
    Returns:
        None
    """
    ConfigRegistry.register_file(
        config_name=config_name,
        file_path=file_path,
        namespace=namespace
    )

# ------------------------------
# 5. 初始化函数（确保配置正确注册）
# ------------------------------

def initialize_configs():
    """初始化配置系统，确保所有配置都已注册"""
    # 获取所有已注册的配置，验证注册是否成功
    all_configs = get_all_registered_configs()
    total_configs = sum(len(configs) for configs in all_configs.values())
    
    # 打印注册信息，便于调试
    config_info = []
    for ns, configs in all_configs.items():
        config_info.append(f"命名空间 '{ns}' 包含 {len(configs)} 个配置: {', '.join(configs)}")
    
    logging.info(f"配置系统初始化完成，共注册 {total_configs} 个配置")
    for info in config_info:
        logging.debug(info)

# ------------------------------
# 6. 延迟初始化
# ------------------------------
# 注意：不再在模块导入时自动初始化，以避免循环导入问题
# 用户可以在需要时手动调用initialize_configs()