from typing import Optional
import os
import json
from typing import Optional
from datetime import datetime
from src.model_show_v2.data_models import ModelInfoData
from src.model_show_v2.data_access import DataAccessor
from src.utils.log_utils import get_logger
from .base_model_parsers import BaseModelInfoParser, ModelInfoParserRegistry

logger = get_logger(name=__name__)


@ModelInfoParserRegistry.register(namespace="config")
class ConfigFileParser(BaseModelInfoParser):
    """配置文件解析器：解析模型的配置文件"""

    def __init__(self):
        """初始化配置文件解析器"""
        pass

    def support(self, file_path: str, namespace: str = "default") -> bool:
        """判断是否为支持的配置文件格式

        Args:
            file_path: 文件路径
            namespace: 命名空间，默认为"default"

        Returns:
            bool: 是否支持该文件
        """
        if not os.path.exists(file_path):
            return False

        # 支持.json格式的配置文件
        ext = os.path.splitext(file_path)[1].lower()
        return ext == ".json" and "config" in file_path.lower()

    def parse(
        self, file_path: str, namespace: str = "default"
    ) -> Optional[ModelInfoData]:
        """解析配置文件为ModelInfoData对象

        Args:
            file_path: 文件路径
            namespace: 命名空间，默认为"default"

        Returns:
            ModelInfoData: 解析后的模型信息数据
        """
        try:
            # 使用DataAccessor读取JSON文件
            config_data = DataAccessor.read_file(file_path)
            if config_data is None:
                raise ValueError(f"配置文件读取失败: {file_path}")

            # 提取模型基本信息
            model_name = config_data.get("model_name", "unknown")

            # 从timestamp字段解析时间戳，如果没有则使用文件修改时间
            timestamp_str = config_data.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.strptime(
                        timestamp_str, "%Y-%m-%d %H:%M:%S"
                    ).timestamp()
                except ValueError:
                    timestamp = DataAccessor.get_file_timestamp(file_path)
            else:
                timestamp = DataAccessor.get_file_timestamp(file_path)

            # 创建并返回ModelInfoData对象（适配新的数据结构）
            return ModelInfoData(
                name=model_name,
                path=file_path,
                model_type=config_data.get("model_type", model_name),  # 优先使用配置中的模型类型
                timestamp=timestamp,
                params=config_data,  # 整个配置数据作为参数
                framework=config_data.get("framework", "PyTorch"),  # 从配置中获取框架
                task_type=config_data.get("task_type", "classification"),  # 从配置中获取任务类型
                version=config_data.get("version", "1.0"),  # 从配置中获取版本
                # 移除了metrics参数，因为新的数据结构使用metric_list
                # 移除了namespace参数，因为新的数据结构中没有该字段
            )

        except Exception as e:
            logger.error(f"解析配置文件失败 {file_path}: {str(e)}")
            raise ValueError(f"解析配置文件失败: {str(e)}")
