from typing import List, Optional, Dict, Any, Type
from abc import ABC, abstractmethod
import os
import re
from src.model_show.data_models import ModelInfoData
from src.utils.log_utils.log_utils import get_logger


# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_parser.log", global_level="INFO")


class BaseModelInfoParser(ABC):
    """模型信息解析器抽象接口：定义所有解析器的统一规范"""
    
    # 解析器优先级，数值越高优先级越高
    priority: int = 50
    
    @abstractmethod
    def support(self, file_path: str, namespace: str = "default") -> bool:
        """判断当前解析器是否支持该文件
        
        Args:
            file_path: 文件路径
            namespace: 命名空间，默认为"default"
        
        Returns:
            bool: 是否支持该文件
        """
        pass
    
    @abstractmethod
    def parse(self, file_path: str, namespace: str = "default") -> Optional[ModelInfoData]:
        """解析文件为标准化ModelInfoData
        
        Args:
            file_path: 文件路径
            namespace: 命名空间，默认为"default"
        
        Returns:
            ModelInfoData: 解析后的模型信息数据
        """
        pass


class ModelInfoParserRegistry:
    """模型信息解析器注册中心：管理所有解析器，实现自动匹配"""
    
    _parsers: Dict[str, List[BaseModelInfoParser]] = {"default": []}  # 按命名空间组织解析器
    
    @classmethod
    def register(cls, parser_cls=None, namespace: str = "default"):
        """注册解析器类（支持直接调用和装饰器用法）"""
        # 如果parser_cls为None，说明是作为带参数的装饰器使用
        if parser_cls is None:
            # 返回一个可以接受parser_cls的装饰器函数
            def decorator(cls_to_register):
                return cls.register(cls_to_register, namespace)
            return decorator
        
        # 正常注册逻辑
        if namespace not in cls._parsers:
            cls._parsers[namespace] = []
        
        # 实例化解析器并添加到注册表
        parser_instance = parser_cls()
        cls._parsers[namespace].append(parser_instance)
        
        # 按优先级排序
        cls._parsers[namespace].sort(key=lambda p: p.priority, reverse=True)
        
        return parser_cls
    
    @classmethod
    def unregister(cls, parser_cls: Type[BaseModelInfoParser], namespace: str = "default"):
        """注销解析器"""
        if namespace in cls._parsers:
            cls._parsers[namespace] = [
                p for p in cls._parsers[namespace] 
                if not isinstance(p, parser_cls)
            ]
    
    @classmethod
    def get_matched_parser(cls, file_path: str, namespace: str = "default") -> Optional[BaseModelInfoParser]:
        """根据文件路径匹配最合适的解析器（考虑优先级）"""
        # 首先在指定命名空间中查找
        if namespace in cls._parsers:
            for parser in cls._parsers[namespace]:
                if parser.support(file_path, namespace):
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
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
            
        parser = cls.get_matched_parser(file_path, namespace)
        if not parser:
            logger.warning(f"未找到匹配的模型信息解析器: {file_path}")
            return None
        
        try:
            model_info = parser.parse(file_path, namespace)
            if model_info:
                model_info.namespace = namespace  # 设置命名空间
            return model_info
        except Exception as e:
            logger.error(f"解析模型信息文件失败 {file_path}: {str(e)}")
            return None
