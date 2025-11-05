from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import os
import re
import torch
from src.model_visualization.data_models import ModelInfoData
from src.model_visualization.data_access import DataAccessor
from src.utils.log_utils.log_utils import get_logger


# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_parser.log", global_level="INFO")


# ==================================================
# ModelInfoData解析器层：负责解析不同格式的模型信息文件
# ==================================================
class BaseModelInfoParser(ABC):
    """模型信息解析器抽象接口：定义所有解析器的统一规范"""
    
    @abstractmethod
    def support(self, file_path: str) -> bool:
        """判断当前解析器是否支持该文件"""
        pass
    
    @abstractmethod
    def parse(self, file_path: str) -> Optional[ModelInfoData]:
        """解析文件为标准化ModelInfoData"""
        pass


class ModelInfoParserRegistry:
    """模型信息解析器注册中心：管理所有解析器，实现自动匹配"""
    
    _parsers: Dict[str, List[BaseModelInfoParser]] = {"default": []}  # 按命名空间组织解析器
    
    @classmethod
    def register(cls, parser: BaseModelInfoParser, namespace: str = "default"):
        """注册解析器实例"""
        if namespace not in cls._parsers:
            cls._parsers[namespace] = []
        cls._parsers[namespace].append(parser)
    
    @classmethod
    def get_matched_parser(cls, file_path: str, namespace: str = "default") -> Optional[BaseModelInfoParser]:
        """根据文件路径匹配最合适的解析器"""
        # 首先在指定命名空间中查找
        if namespace in cls._parsers:
            for parser in cls._parsers[namespace]:
                if parser.support(file_path):
                    return parser
        
        # 如果指定命名空间中未找到，尝试在默认命名空间中查找
        if namespace != "default" and "default" in cls._parsers:
            for parser in cls._parsers["default"]:
                if parser.support(file_path):
                    return parser
        
        return None
    
    @classmethod
    def parse_file(cls, file_path: str, namespace: str = "default") -> Optional[ModelInfoData]:
        """自动解析文件的入口方法"""
        parser = cls.get_matched_parser(file_path, namespace)
        if not parser:
            logger.warning(f"未找到匹配的模型信息解析器: {file_path}")
            return None
        
        try:
            model_info = parser.parse(file_path)
            if model_info:
                model_info.namespace = namespace  # 设置命名空间
            return model_info
        except Exception as e:
            logger.error(f"解析模型信息文件失败 {file_path}: {str(e)}")
            return None


# ------------------------------
# 注册解析器函数（延迟导入以避免循环依赖）
# ------------------------------
def register_parsers():
    """注册所有解析器，应在应用程序启动时调用"""
    # 延迟导入以避免循环依赖
    from .model_info_parsers_impl import ConfigFileParser, MetricsFileParser, ModelFileParser
    
    # 注册所有解析器
    ModelInfoParserRegistry.register(ConfigFileParser(), namespace="default")
    ModelInfoParserRegistry.register(MetricsFileParser(), namespace="default")
    ModelInfoParserRegistry.register(ModelFileParser(), namespace="default")


# 在模块导入时不自动注册，而是提供注册函数供调用方使用
# 如果需要自动注册（不推荐），可以取消下面这行的注释
# register_parsers()
