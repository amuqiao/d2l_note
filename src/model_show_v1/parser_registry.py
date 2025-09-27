"""
解析器注册统一入口：
1. 集中导入所有解析器模块，触发装饰器自动注册
2. 对外暴露核心类和便捷函数，简化外部调用
3. 确保应用启动时所有解析器都已正确注册
"""
from typing import Optional

# ------------------------------
# 1. 集中导入所有解析器模块（关键：导入即触发装饰器注册）
# ------------------------------
# 导入配置文件解析器（触发 ConfigFileParser 的装饰器注册）

from .parsers.model_config_parser import ConfigFileParser


# 导入指标文件解析器（触发 MetricsFileParser 的装饰器注册）
from .parsers.model_metrics_parser import MetricsFileParser


# 导入模型文件解析器（触发 ModelFileParser 的装饰器注册）
from .parsers.model_file_parser import ModelFileParser


# ------------------------------
# 2. 对外暴露核心类，简化外部调用
# ------------------------------
from .parsers.base_model_parsers import (
    BaseModelInfoParser,  # 解析器基类（供外部自定义解析器用）
    ModelInfoParserRegistry  # 注册中心（供外部调用解析逻辑用）
)

# 导入数据模型，方便外部使用
from .data_models import ModelInfoData

# ------------------------------
# 3. 提供便捷函数（进一步简化外部使用）
# ------------------------------
def parse_model_info(
    file_path: str,
    namespace: str = "default"
) -> Optional[ModelInfoData]:
    """
    便捷解析函数：直接调用注册中心解析逻辑
    
    Args:
        file_path: 模型信息文件路径
        namespace: 解析器命名空间
        
    Returns:
        解析后的ModelInfoData对象，解析失败则返回None
    """
    # 复用注册中心的parse_file方法
    return ModelInfoParserRegistry.parse_file(
        file_path=file_path,
        namespace=namespace
    )

def get_all_registered_parsers(namespace: str = None) -> list[BaseModelInfoParser]:
    """
    获取所有已注册的解析器
    
    Args:
        namespace: 可选，指定命名空间，None表示所有命名空间
        
    Returns:
        解析器实例列表
    """
    if namespace:
        return ModelInfoParserRegistry._parsers.get(namespace, [])
    
    all_parsers = []
    for parsers in ModelInfoParserRegistry._parsers.values():
        all_parsers.extend(parsers)
    return all_parsers

# ------------------------------
# 4. 初始化函数（确保解析器正确注册）
# ------------------------------
def initialize_parsers():
    """初始化解析器系统，确保所有解析器都已注册"""
    # 获取所有已注册的解析器，验证注册是否成功
    all_parsers = get_all_registered_parsers()
    if not all_parsers:
        raise RuntimeError("未检测到任何注册的解析器，请检查解析器实现和注册逻辑")
    
    # 打印注册信息，便于调试
    parser_info = []
    for ns, parsers in ModelInfoParserRegistry._parsers.items():
        parser_info.append(f"命名空间 '{ns}' 包含 {len(parsers)} 个解析器: {[p.__class__.__name__ for p in parsers]}")
    
    from src.utils.log_utils import get_logger
    logger = get_logger(name="parser_registry")
    logger.info(f"解析器系统初始化完成，共注册 {len(all_parsers)} 个解析器")
    for info in parser_info:
        logger.debug(info)
