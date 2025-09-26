from typing import Optional
import os
import configparser
from .base_parsers import BaseModelInfoParser
from src.model_visualization.data_models import ModelInfoData


class ConfigFileParser(BaseModelInfoParser):
    """配置文件解析器：解析模型的配置文件"""
    
    def support(self, file_path: str) -> bool:
        """判断是否为支持的配置文件格式"""
        if not os.path.exists(file_path):
            return False
            
        # 支持.ini、.conf、.config等配置文件
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.ini', '.conf', '.config']
    
    def parse(self, file_path: str) -> Optional[ModelInfoData]:
        """解析配置文件为ModelInfoData对象"""
        try:
            config = configparser.ConfigParser()
            config.read(file_path)
            
            # 提取模型基本信息
            model_name = config.get('model', 'name', fallback='unknown')
            model_version = config.get('model', 'version', fallback='unknown')
            input_shape = config.get('model', 'input_shape', fallback='unknown')
            
            # 创建并返回ModelInfoData对象
            return ModelInfoData(
                file_path=file_path,
                model_name=model_name,
                model_version=model_version,
                input_shape=input_shape,
                # 其他字段根据实际情况填充
            )
            
        except Exception as e:
            # 具体日志记录由调用方处理
            raise ValueError(f"解析配置文件失败: {str(e)}")
