from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import os
import json
import yaml
from src.model_center.base_registry import ComponentRegistry, logger


class ConfigRegistry(ComponentRegistry):
    """参数配置注册中心：专门用于管理训练参数配置"""
    
    # 存储配置模板的字典，格式：{namespace: {config_name: template}}
    _config_templates: Dict[str, Dict[str, Dict[str, Any]]] = {"default": {}}
    
    # 存储配置文件路径的字典，格式：{namespace: {config_name: file_path}}
    _config_files: Dict[str, Dict[str, str]] = {"default": {}}
    
    @classmethod
    def register_config(cls, config_name: str = None, template: Optional[Dict[str, Any]] = None, 
                       namespace: str = "default", file_path: Optional[str] = None):
        """注册配置模板装饰器：用于注册配置类或函数及其默认模板
        
        Args:
            config_name: 配置名称，将作为注册的唯一标识，如果不提供则使用类名或函数名
            template: 可选的配置默认模板字典
            namespace: 命名空间，默认为"default"
            file_path: 可选的配置文件路径
            
        Returns:
            装饰器函数
        """
        def decorator(config_source: Union[Callable, Any]):
            # 确定配置名称
            name = config_name if config_name else getattr(config_source, '__name__', str(config_source))
            
            # 注册配置源
            cls.register(config_source, namespace=namespace, name=name)
            
            # 注册配置模板（如果提供）
            if template:
                if namespace not in cls._config_templates:
                    cls._config_templates[namespace] = {}
                cls._config_templates[namespace][name] = template
                logger.debug(f"配置模板 '{name}' 已在命名空间 '{namespace}' 中注册")
            
            # 注册配置文件路径（如果提供）
            if file_path:
                if namespace not in cls._config_files:
                    cls._config_files[namespace] = {}
                cls._config_files[namespace][name] = file_path
                logger.debug(f"配置文件 '{name}' 的路径已在命名空间 '{namespace}' 中注册")
            
            return config_source
            
        return decorator
    
    @classmethod
    def register_template(cls, config_name: str, template: Dict[str, Any], namespace: str = "default"):
        """注册或更新配置模板
        
        Args:
            config_name: 配置名称
            template: 配置模板字典
            namespace: 命名空间，默认为"default"
        """
        if namespace not in cls._config_templates:
            cls._config_templates[namespace] = {}
        cls._config_templates[namespace][config_name] = template
        logger.debug(f"配置模板 '{config_name}' 已在命名空间 '{namespace}' 中注册")
    
    @classmethod
    def register_file(cls, config_name: str, file_path: str, namespace: str = "default"):
        """注册配置文件路径
        
        Args:
            config_name: 配置名称
            file_path: 配置文件路径
            namespace: 命名空间，默认为"default"
        """
        if namespace not in cls._config_files:
            cls._config_files[namespace] = {}
        cls._config_files[namespace][config_name] = file_path
        logger.debug(f"配置文件 '{config_name}' 的路径已在命名空间 '{namespace}' 中注册")
    
    @classmethod
    def load_config_from_file(cls, file_path: str) -> Dict[str, Any]:
        """从配置文件中加载配置
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            配置字典
            
        Raises:
            ValueError: 如果文件格式不支持或文件不存在
        """
        if not os.path.exists(file_path):
            raise ValueError(f"配置文件不存在: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_ext == '.json':
                    config = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_ext}")
            
            logger.debug(f"已成功从文件 '{file_path}' 加载配置")
            return config
        except Exception as e:
            logger.error(f"从文件 '{file_path}' 加载配置失败: {str(e)}")
            raise ValueError(f"从文件 '{file_path}' 加载配置失败: {str(e)}")
    
    @classmethod
    def get_config(cls, config_name: str, namespace: str = "default", 
                 file_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """获取配置，优先级：传入的file_path > 注册的file_path > 注册的template
        
        Args:
            config_name: 配置名称
            namespace: 命名空间，默认为"default"
            file_path: 可选的配置文件路径
            **kwargs: 要覆盖的配置项
            
        Returns:
            配置字典
        """
        # 1. 检查是否提供了文件路径
        if file_path:
            return cls.load_config_from_file(file_path)
        
        # 2. 检查是否注册了文件路径
        if namespace in cls._config_files and config_name in cls._config_files[namespace]:
            return cls.load_config_from_file(cls._config_files[namespace][config_name])
        
        # 3. 检查是否在指定命名空间中注册了模板
        if namespace in cls._config_templates and config_name in cls._config_templates[namespace]:
            config = cls._config_templates[namespace][config_name].copy()
            config.update(kwargs)
            return config
        
        # 4. 如果指定命名空间中未找到，尝试在默认命名空间中查找
        if namespace != "default" and "default" in cls._config_templates and config_name in cls._config_templates["default"]:
            logger.debug(f"在命名空间 '{namespace}' 中未找到配置模板 '{config_name}'，使用默认命名空间中的模板")
            config = cls._config_templates["default"][config_name].copy()
            config.update(kwargs)
            return config
        
        # 5. 如果什么都没找到，返回空字典和传入的参数
        logger.warning(f"未找到配置 '{config_name}'（命名空间: '{namespace}'），返回默认配置")
        return kwargs
    
    @classmethod
    def merge_configs(cls, *config_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个配置字典
        
        Args:
            *config_dicts: 要合并的配置字典，后面的字典会覆盖前面的字典中的同名项
            
        Returns:
            合并后的配置字典
        """
        merged_config = {}
        for config in config_dicts:
            if config:
                merged_config.update(config)
        return merged_config
    
    @classmethod
    def get_default_config(cls, namespace: str = "default") -> Dict[str, Any]:
        """获取默认配置模板
        
        Args:
            namespace: 命名空间，默认为"default"
            
        Returns:
            默认配置字典
        """
        default_config = {
            "model": {
                "name": "",
                "args": {}
            },
            "data": {
                "train_dataset": {
                    "name": "",
                    "args": {}
                },
                "test_dataset": {
                    "name": "",
                    "args": {}
                },
                "batch_size": 32,
                "shuffle": True,
                "num_workers": 0
            },
            "training": {
                "optimizer": {
                    "name": "adam",
                    "args": {
                        "lr": 0.001,
                        "betas": [0.9, 0.999],
                        "eps": 1e-08,
                        "weight_decay": 0
                    }
                },
                "scheduler": {
                    "name": "",
                    "args": {}
                },
                "loss_function": {
                    "name": "cross_entropy",
                    "args": {}
                },
                "epochs": 10,
                "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu",
                "checkpoint_dir": "./checkpoints",
                "log_interval": 10,
                "early_stopping": {
                    "enabled": False,
                    "patience": 5
                }
            }
        }
        
        # 如果有命名空间特定的默认配置，可以在这里进行合并
        if namespace in cls._config_templates and "default" in cls._config_templates[namespace]:
            default_config = cls.merge_configs(default_config, cls._config_templates[namespace]["default"])
        
        return default_config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], file_path: str) -> None:
        """保存配置到文件
        
        Args:
            config: 配置字典
            file_path: 要保存的文件路径
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_ext == '.json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                elif file_ext in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_ext}")
            
            logger.debug(f"已成功将配置保存到文件 '{file_path}'")
        except Exception as e:
            logger.error(f"保存配置到文件 '{file_path}' 失败: {str(e)}")
            raise ValueError(f"保存配置到文件 '{file_path}' 失败: {str(e)}")
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any], required_keys: List[str] = None) -> bool:
        """验证配置是否包含所有必需的键
        
        Args:
            config: 要验证的配置字典
            required_keys: 必需的键列表
            
        Returns:
            是否有效
        """
        if not required_keys:
            return True
            
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logger.error(f"配置缺少必需的键: {', '.join(missing_keys)}")
            return False
        
        return True
    
    @classmethod
    def get_config_names(cls, namespace: str = "default") -> List[str]:
        """获取指定命名空间中所有已注册的配置名称
        
        Args:
            namespace: 命名空间，默认为"default"
            
        Returns:
            配置名称列表
        """
        config_names = set()
        
        # 收集配置模板中的名称
        if namespace in cls._config_templates:
            config_names.update(cls._config_templates[namespace].keys())
        
        # 收集配置文件中的名称
        if namespace in cls._config_files:
            config_names.update(cls._config_files[namespace].keys())
        
        # 收集通过register注册的组件名称
        config_names.update(cls.list_components(namespace))
        
        return list(config_names)