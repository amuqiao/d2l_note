"""
数据加载器注册统一入口：
1. 集中导入所有数据加载器模块，触发装饰器自动注册
2. 对外暴露核心类和便捷函数，简化外部调用
3. 确保应用启动时所有数据加载器都已正确注册
"""
from typing import Optional, Dict, Any, Type, List, Callable, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
import logging

# ------------------------------
# 1. 统一注册入口 - 为了避免循环导入，这里不直接导入数据加载器
# ------------------------------
# 注意：数据加载器应通过装饰器自行注册，而不是在此处集中导入
# 这样可以避免循环导入问题，同时保持模块的解耦

# ------------------------------
# 2. 对外暴露核心类，简化外部调用
# ------------------------------
from .data_loader_registry import DataLoaderRegistry

# ------------------------------
# 3. 提供便捷函数（进一步简化外部使用）
# ------------------------------

def create_dataset(
    dataset_name: str,
    namespace: str = "default",
    **kwargs
) -> Dataset:
    """
    便捷创建数据集函数：直接调用注册中心创建数据集
    
    Args:
        dataset_name: 数据集名称
        namespace: 数据集命名空间
        **kwargs: 传递给数据集构造函数的参数
        
    Returns:
        创建的数据集实例
    """
    try:
        return DataLoaderRegistry.create_dataset(
            dataset_name=dataset_name,
            namespace=namespace,
            **kwargs
        )
    except Exception as e:
        logging.error(f"创建数据集 '{dataset_name}' 失败: {e}")
        raise


def create_data_loader(
    dataset_name: str,
    namespace: str = "default",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    便捷创建数据加载器函数
    
    Args:
        dataset_name: 数据集名称
        namespace: 数据集命名空间
        batch_size: 批次大小，默认为32
        shuffle: 是否打乱数据，默认为True
        num_workers: 加载数据的进程数，默认为0
        **kwargs: 传递给Dataset构造函数的参数
        
    Returns:
        DataLoader实例
    """
    try:
        return DataLoaderRegistry.create_data_loader(
            dataset_name=dataset_name,
            namespace=namespace,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
    except Exception as e:
        logging.error(f"创建数据加载器失败: {e}")
        raise


def create_train_test_loaders(
    dataset_name: str,
    namespace: str = "default",
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    train_shuffle: bool = True,
    test_shuffle: bool = False,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    便捷创建训练和测试数据加载器函数
    
    Args:
        dataset_name: 数据集名称
        namespace: 数据集命名空间
        train_batch_size: 训练数据批次大小，默认为32
        test_batch_size: 测试数据批次大小，默认为32
        train_shuffle: 训练数据是否打乱，默认为True
        test_shuffle: 测试数据是否打乱，默认为False
        num_workers: 加载数据的进程数，默认为0
        **kwargs: 传递给Dataset构造函数的参数
        
    Returns:
        (train_loader, test_loader) 元组
    """
    try:
        return DataLoaderRegistry.create_train_test_loaders(
            dataset_name=dataset_name,
            namespace=namespace,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_shuffle=train_shuffle,
            test_shuffle=test_shuffle,
            num_workers=num_workers,
            **kwargs
        )
    except Exception as e:
        logging.error(f"创建训练和测试数据加载器失败: {e}")
        raise


def get_dataset_config(dataset_name: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
    """
    获取数据集配置
    
    Args:
        dataset_name: 数据集名称
        namespace: 数据集命名空间
        
    Returns:
        数据集配置字典，如果不存在则返回None
    """
    try:
        if namespace in DataLoaderRegistry._dataset_configs and dataset_name in DataLoaderRegistry._dataset_configs[namespace]:
            return DataLoaderRegistry._dataset_configs[namespace][dataset_name]
        return None
    except Exception as e:
        logging.error(f"获取数据集配置失败: {e}")
        return None


def get_dataset_preprocess_func(dataset_name: str, namespace: str = "default") -> Optional[Callable]:
    """
    获取数据集预处理函数
    
    Args:
        dataset_name: 数据集名称
        namespace: 数据集命名空间
        
    Returns:
        预处理函数，如果不存在则返回None
    """
    try:
        if namespace in DataLoaderRegistry._dataset_preprocess_funcs and dataset_name in DataLoaderRegistry._dataset_preprocess_funcs[namespace]:
            return DataLoaderRegistry._dataset_preprocess_funcs[namespace][dataset_name]
        return None
    except Exception as e:
        logging.error(f"获取数据集预处理函数失败: {e}")
        return None


def get_all_registered_datasets(namespace: str = None) -> Dict[str, List[str]]:
    """
    获取所有已注册的数据集信息
    
    Args:
        namespace: 可选，指定命名空间，None表示所有命名空间
        
    Returns:
        数据集信息字典，键为命名空间，值为数据集名称列表
    """
    result = {}
    
    # 确定要遍历的命名空间
    namespaces_to_check = [namespace] if namespace and namespace in DataLoaderRegistry._components else \
                         DataLoaderRegistry._components.keys() if not namespace else []
    
    for ns in namespaces_to_check:
        result[ns] = list(DataLoaderRegistry._components[ns].keys())
    
    return result


def is_dataset_registered(dataset_name: str, namespace: str = "default") -> bool:
    """
    检查数据集是否已注册
    
    Args:
        dataset_name: 数据集名称
        namespace: 数据集命名空间
        
    Returns:
        是否已注册
    """
    return DataLoaderRegistry.get(dataset_name, namespace=namespace) is not None

# ------------------------------
# 4. 自动注册机制：简化数据集注册流程的装饰器
# ------------------------------

def register_dataset(
    name: str = None,
    namespace: str = "default",
    config: Optional[Dict[str, Any]] = None,
    preprocess_func: Optional[Callable] = None
):
    """
    自动注册数据集的装饰器：简化数据集注册流程
    
    Args:
        name: 数据集名称，如果不提供则使用类名
        namespace: 命名空间
        config: 可选的数据集默认配置
        preprocess_func: 可选的数据集预处理函数
    
    Returns:
        装饰后的数据集类
    """
    def decorator(dataset_class: Type[Dataset]):
        try:
            # 使用DataLoaderRegistry的register_dataset方法注册数据集
            DataLoaderRegistry.register_dataset(
                dataset_name=name,
                namespace=namespace,
                config=config,
                preprocess_func=preprocess_func
            )(dataset_class)
            
            # 记录注册信息
            dataset_name_final = name if name else dataset_class.__name__
            logging.debug(f"数据集 '{dataset_name_final}' 已在命名空间 '{namespace}' 中注册")
        except Exception as e:
            # 改进错误处理，避免装饰器异常影响数据集定义
            logging.error(f"注册数据集失败: {e}")
            
        return dataset_class
    
    return decorator


def register_dataset_config(
    dataset_name: str,
    config: Dict[str, Any],
    namespace: str = "default"
):
    """
    便捷注册数据集配置函数
    
    Args:
        dataset_name: 数据集名称
        config: 数据集配置字典
        namespace: 命名空间
    
    Returns:
        None
    """
    DataLoaderRegistry.register_config(
        dataset_name=dataset_name,
        config=config,
        namespace=namespace
    )


def register_dataset_preprocess_func(
    dataset_name: str,
    preprocess_func: Callable,
    namespace: str = "default"
):
    """
    便捷注册数据集预处理函数
    
    Args:
        dataset_name: 数据集名称
        preprocess_func: 预处理函数
        namespace: 命名空间
    
    Returns:
        None
    """
    DataLoaderRegistry.register_preprocess_func(
        dataset_name=dataset_name,
        preprocess_func=preprocess_func,
        namespace=namespace
    )

# ------------------------------
# 5. 初始化函数（确保数据加载器正确注册）
# ------------------------------

def initialize_data_loaders():
    """初始化数据加载器系统，确保所有数据加载器都已注册"""
    # 获取所有已注册的数据集，验证注册是否成功
    all_datasets = get_all_registered_datasets()
    total_datasets = sum(len(datasets) for datasets in all_datasets.values())
    
    # 打印注册信息，便于调试
    dataset_info = []
    for ns, datasets in all_datasets.items():
        dataset_info.append(f"命名空间 '{ns}' 包含 {len(datasets)} 个数据集: {', '.join(datasets)}")
    
    logging.info(f"数据加载器系统初始化完成，共注册 {total_datasets} 个数据集")
    for info in dataset_info:
        logging.debug(info)

# ------------------------------
# 6. 延迟初始化
# ------------------------------
# 注意：不再在模块导入时自动初始化，以避免循环导入问题
# 用户可以在需要时手动调用initialize_data_loaders()