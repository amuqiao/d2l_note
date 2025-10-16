from typing import Dict, List, Optional, Type, Any, Callable, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Type, Any, Callable, Tuple
from src.model_center.base_registry import ComponentRegistry, logger


class DataLoaderRegistry(ComponentRegistry):
    """数据加载器注册中心：专门用于管理数据加载器"""
    
    # 存储数据集配置的字典，格式：{namespace: {dataset_name: config}}
    _dataset_configs: Dict[str, Dict[str, Dict[str, Any]]] = {"default": {}}
    
    # 存储数据集预处理函数的字典，格式：{namespace: {dataset_name: preprocess_func}}
    _dataset_preprocess_funcs: Dict[str, Dict[str, Callable]] = {"default": {}}
    
    @classmethod
    def register_dataset(cls, dataset_name: str = None, config: Optional[Dict[str, Any]] = None, 
                        namespace: str = "default", preprocess_func: Optional[Callable] = None):
        """数据集注册装饰器：用于注册数据集类及其配置和预处理函数
        
        Args:
            dataset_name: 数据集名称，将作为注册的唯一标识，如果不提供则使用类名
            config: 可选的数据集默认配置字典
            namespace: 命名空间，默认为"default"
            preprocess_func: 可选的数据集预处理函数
            
        Returns:
            装饰器函数
        """
        def decorator(dataset_class: Type[Dataset]):
            # 确定数据集名称
            name = dataset_name if dataset_name else dataset_class.__name__
            
            # 注册数据集类
            cls.register(dataset_class, namespace=namespace, name=name)
            
            # 注册数据集配置（如果提供）
            if config:
                if namespace not in cls._dataset_configs:
                    cls._dataset_configs[namespace] = {}
                cls._dataset_configs[namespace][name] = config
                logger.debug(f"数据集 '{name}' 的配置已在命名空间 '{namespace}' 中注册")
            
            # 注册数据集预处理函数（如果提供）
            if preprocess_func:
                if namespace not in cls._dataset_preprocess_funcs:
                    cls._dataset_preprocess_funcs[namespace] = {}
                cls._dataset_preprocess_funcs[namespace][name] = preprocess_func
                logger.debug(f"数据集 '{name}' 的预处理函数已在命名空间 '{namespace}' 中注册")
            
            return dataset_class
            
        return decorator
    
    @classmethod
    def register_config(cls, dataset_name: str, config: Dict[str, Any], namespace: str = "default"):
        """注册或更新数据集配置
        
        Args:
            dataset_name: 数据集名称
            config: 数据集配置字典
            namespace: 命名空间，默认为"default"
        """
        if namespace not in cls._dataset_configs:
            cls._dataset_configs[namespace] = {}
        cls._dataset_configs[namespace][dataset_name] = config
        logger.debug(f"数据集 '{dataset_name}' 的配置已在命名空间 '{namespace}' 中注册")
    
    @classmethod
    def register_preprocess_func(cls, dataset_name: str, preprocess_func: Callable, namespace: str = "default"):
        """注册数据集预处理函数
        
        Args:
            dataset_name: 数据集名称
            preprocess_func: 预处理函数
            namespace: 命名空间，默认为"default"
        """
        if namespace not in cls._dataset_preprocess_funcs:
            cls._dataset_preprocess_funcs[namespace] = {}
        cls._dataset_preprocess_funcs[namespace][dataset_name] = preprocess_func
        logger.debug(f"数据集 '{dataset_name}' 的预处理函数已在命名空间 '{namespace}' 中注册")
    
    @classmethod
    def create_dataset(cls, dataset_name: str, namespace: str = "default", **kwargs) -> Dataset:
        """根据数据集名称创建数据集实例
        
        Args:
            dataset_name: 数据集名称
            namespace: 命名空间，默认为"default"
            **kwargs: 传递给数据集构造函数的参数
            
        Returns:
            数据集实例
            
        Raises:
            ValueError: 如果数据集名称未注册
        """
        dataset_class = cls.get(dataset_name, namespace)
        if not dataset_class:
            registered_datasets = ', '.join(cls.list_components(namespace))
            raise ValueError(
                f"不支持的数据集类型: {dataset_name}，命名空间 '{namespace}' 中已注册的数据集有: {registered_datasets}"
            )
        
        # 获取默认配置
        default_config = cls.get_config(dataset_name, namespace) or {}
        # 合并默认配置和传入的参数
        combined_config = {**default_config, **kwargs}
        
        # 创建数据集实例
        dataset_instance = dataset_class(**combined_config)
        logger.debug(f"已创建数据集 '{dataset_name}' 的实例")
        
        # 应用预处理函数（如果有）
        preprocess_func = cls.get_preprocess_func(dataset_name, namespace)
        if preprocess_func:
            try:
                dataset_instance = preprocess_func(dataset_instance)
                logger.debug(f"已应用数据集 '{dataset_name}' 的预处理函数")
            except Exception as e:
                logger.error(f"应用数据集 '{dataset_name}' 的预处理函数失败: {str(e)}")
        
        return dataset_instance
    
    @classmethod
    def create_data_loader(cls, dataset_name: str, namespace: str = "default", 
                         batch_size: int = 32, shuffle: bool = True, 
                         num_workers: int = 0, **kwargs) -> DataLoader:
        """创建数据加载器
        
        Args:
            dataset_name: 数据集名称
            namespace: 命名空间，默认为"default"
            batch_size: 批次大小，默认为32
            shuffle: 是否打乱数据，默认为True
            num_workers: 加载数据的进程数，默认为0
            **kwargs: 传递给Dataset构造函数的参数
            
        Returns:
            DataLoader实例
        """
        # 创建数据集实例
        dataset = cls.create_dataset(dataset_name, namespace, **kwargs)
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.debug(f"已创建数据加载器，数据集: '{dataset_name}'，批次大小: {batch_size}")
        return data_loader
    
    @classmethod
    def create_train_test_loaders(cls, dataset_name: str, namespace: str = "default", 
                                 train_batch_size: int = 32, test_batch_size: int = 32, 
                                 train_shuffle: bool = True, test_shuffle: bool = False, 
                                 num_workers: int = 0, **kwargs) -> Tuple[DataLoader, DataLoader]:
        """创建训练和测试数据加载器
        
        Args:
            dataset_name: 数据集名称
            namespace: 命名空间，默认为"default"
            train_batch_size: 训练数据批次大小，默认为32
            test_batch_size: 测试数据批次大小，默认为32
            train_shuffle: 训练数据是否打乱，默认为True
            test_shuffle: 测试数据是否打乱，默认为False
            num_workers: 加载数据的进程数，默认为0
            **kwargs: 传递给Dataset构造函数的参数
            
        Returns:
            (train_loader, test_loader) 元组
        """
        # 创建训练数据加载器
        train_loader = cls.create_data_loader(
            dataset_name=dataset_name,
            namespace=namespace,
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            train=True,  # 传递train=True标记
            **kwargs
        )
        
        # 创建测试数据加载器
        test_loader = cls.create_data_loader(
            dataset_name=dataset_name,
            namespace=namespace,
            batch_size=test_batch_size,
            shuffle=test_shuffle,
            num_workers=num_workers,
            train=False,  # 传递train=False标记
            **kwargs
        )
        
        return train_loader, test_loader
    
    @classmethod
    def get_config(cls, dataset_name: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """获取数据集的默认配置
        
        Args:
            dataset_name: 数据集名称
            namespace: 命名空间，默认为"default"
            
        Returns:
            数据集配置字典，如果未找到则返回None
        """
        # 首先在指定命名空间中查找
        if namespace in cls._dataset_configs and dataset_name in cls._dataset_configs[namespace]:
            return cls._dataset_configs[namespace][dataset_name].copy()  # 返回副本避免修改原始配置
        
        # 如果指定命名空间中未找到，尝试在默认命名空间中查找
        if namespace != "default" and "default" in cls._dataset_configs and dataset_name in cls._dataset_configs["default"]:
            logger.debug(f"在命名空间 '{namespace}' 中未找到数据集 '{dataset_name}' 的配置，使用默认命名空间中的配置")
            return cls._dataset_configs["default"][dataset_name].copy()
        
        logger.warning(f"未找到数据集 '{dataset_name}' 的配置（命名空间: '{namespace}'）")
        return None
    
    @classmethod
    def get_preprocess_func(cls, dataset_name: str, namespace: str = "default") -> Optional[Callable]:
        """获取数据集的预处理函数
        
        Args:
            dataset_name: 数据集名称
            namespace: 命名空间，默认为"default"
            
        Returns:
            预处理函数，如果未找到则返回None
        """
        # 首先在指定命名空间中查找
        if namespace in cls._dataset_preprocess_funcs and dataset_name in cls._dataset_preprocess_funcs[namespace]:
            return cls._dataset_preprocess_funcs[namespace][dataset_name]
        
        # 如果指定命名空间中未找到，尝试在默认命名空间中查找
        if namespace != "default" and "default" in cls._dataset_preprocess_funcs and dataset_name in cls._dataset_preprocess_funcs["default"]:
            logger.debug(f"在命名空间 '{namespace}' 中未找到数据集 '{dataset_name}' 的预处理函数，使用默认命名空间中的预处理函数")
            return cls._dataset_preprocess_funcs["default"][dataset_name]
        
        # 如果没有注册预处理函数，返回None
        return None
    
    @classmethod
    def is_registered(cls, dataset_name: str, namespace: str = "default") -> bool:
        """检查数据集是否已注册
        
        Args:
            dataset_name: 数据集名称
            namespace: 命名空间，默认为"default"
            
        Returns:
            是否已注册
        """
        # 检查数据集类是否已注册
        if cls.get(dataset_name, namespace):
            return True
        return False