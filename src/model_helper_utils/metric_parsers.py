import os
import sys
import json
from typing import List, Dict, Optional, Any, Callable, Type
from functools import lru_cache
from abc import ABC, abstractmethod
from dataclasses import dataclass


# 导入自定义日志模块
from src.utils.log_utils import get_logger

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_analysis.log", global_level="INFO")

# ==================================================
# 数据访问层：标准化指标数据与解析器接口
# ==================================================
@dataclass
class MetricData:
    """标准化指标数据模型：统一不同指标文件的内存表示"""
    metric_type: str  # 指标类型："epoch_curve" / "confusion_matrix" / "lr_curve" 等
    data: Dict[str, Any]  # 结构化指标数据
    source_path: str  # 数据来源文件路径
    timestamp: float  # 数据生成时间戳


class BaseMetricParser(ABC):
    """指标解析器抽象接口：定义所有解析器的统一规范"""
    @abstractmethod
    def support(self, file_path: str) -> bool:
        """判断当前解析器是否支持该文件"""
        pass

    @abstractmethod
    def parse(self, file_path: str) -> Optional[MetricData]:
        """解析文件为标准化MetricData"""
        pass


class MetricParserRegistry:
    """解析器注册中心：管理所有解析器，实现自动匹配"""
    _parsers: List[BaseMetricParser] = []

    @classmethod
    def register(cls, parser: BaseMetricParser):
        """注册解析器实例"""
        cls._parsers.append(parser)

    @classmethod
    def get_matched_parser(cls, file_path: str) -> Optional[BaseMetricParser]:
        """根据文件路径匹配最合适的解析器"""
        for parser in cls._parsers:
            if parser.support(file_path):
                return parser
        return None

    @classmethod
    def parse_file(cls, file_path: str) -> Optional[MetricData]:
        """自动解析文件的入口方法"""
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
            
        parser = cls.get_matched_parser(file_path)
        if not parser:
            logger.warning(f"未找到匹配的解析器: {file_path}")
            return None
            
        try:
            return parser.parse(file_path)
        except Exception as e:
            logger.error(f"解析文件失败 {file_path}: {str(e)}")
            return None

