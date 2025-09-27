"""
可视化器注册统一入口：
1. 集中导入所有可视化器模块，触发装饰器自动注册
2. 对外暴露核心类和便捷函数，简化外部调用
3. 确保应用启动时所有可视化器都已正确注册
"""
from typing import Optional, Dict, Any

# ------------------------------
# 1. 集中导入所有可视化器模块（关键：导入即触发装饰器注册）
# ------------------------------
# 导入配置文件可视化器（触发 ConfigFileVisualizer 的装饰器注册）
from .visualizers.config_file_visualizer import ConfigFileVisualizer

# ------------------------------
# 2. 导入核心类，简化外部调用
# ------------------------------
from .visualizers.base_model_visualizers import (
    BaseModelVisualizer,  # 可视化器基类（供外部自定义可视化器用）
    ModelVisualizerRegistry  # 注册中心（供外部调用可视化逻辑用）
)

# 导入数据模型，方便外部使用
from .data_models import ModelInfoData

# ------------------------------
# 提供便捷函数（进一步简化外部使用）
# ------------------------------
def visualize_model_info(
    model_info: ModelInfoData,
    namespace: str = "default"
) -> Optional[Dict[str, Any]]:
    """
    便捷可视化函数：直接调用注册中心可视化逻辑
    
    Args:
        model_info: 模型信息数据
        namespace: 可视化器命名空间
        
    Returns:
        可视化结果字典，可视化失败则返回None
    """
    # 复用注册中心的visualize_model方法
    return ModelVisualizerRegistry.visualize_model(
        model_info=model_info,
        namespace=namespace
    )

def compare_model_infos(
    model_infos: list[ModelInfoData],
    namespace: str = "default"
) -> Optional[Dict[str, Any]]:
    """
    便捷比较函数：直接调用注册中心比较多个模型的逻辑
    
    Args:
        model_infos: 模型信息数据列表
        namespace: 可视化器命名空间
        
    Returns:
        比较结果字典，比较失败则返回None
    """
    # 复用注册中心的compare_models方法
    return ModelVisualizerRegistry.compare_models(
        model_infos=model_infos,
        namespace=namespace
    )

def get_all_registered_visualizers(namespace: str = None) -> list[BaseModelVisualizer]:
    """
    获取所有已注册的可视化器
    
    Args:
        namespace: 可选，指定命名空间，None表示所有命名空间
        
    Returns:
        可视化器实例列表
    """
    if namespace:
        return ModelVisualizerRegistry._visualizers.get(namespace, [])
    
    all_visualizers = []
    for visualizers in ModelVisualizerRegistry._visualizers.values():
        all_visualizers.extend(visualizers)
    return all_visualizers

# ------------------------------
# 初始化函数（确保可视化器正确注册）
# ------------------------------
def initialize_visualizers():
    """初始化可视化器系统，确保所有可视化器都已注册"""
    from src.utils.log_utils import get_logger
    logger = get_logger(name="visualizer_registry")
    
    # 获取所有已注册的可视化器，验证注册是否成功
    all_visualizers = get_all_registered_visualizers()
    if not all_visualizers:
        logger.info("未检测到任何注册的可视化器")
        return
    
    # 打印注册信息，便于调试
    visualizer_info = []
    for ns, visualizers in ModelVisualizerRegistry._visualizers.items():
        visualizer_info.append(f"命名空间 '{ns}' 包含 {len(visualizers)} 个可视化器: {[v.__class__.__name__ for v in visualizers]}")
    
    from src.utils.log_utils import get_logger
    logger = get_logger(name="visualizer_registry")
    logger.info(f"可视化器系统初始化完成，共注册 {len(all_visualizers)} 个可视化器")
    for info in visualizer_info:
        logger.debug(info)