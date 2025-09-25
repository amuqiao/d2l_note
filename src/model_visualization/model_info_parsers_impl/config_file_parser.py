from typing import Optional
import os
from abc import ABC, abstractmethod
from src.model_visualization.data_models import ModelInfoData
from src.model_visualization.data_access import DataAccessor
from src.utils.log_utils import get_logger
from src.model_visualization.model_info_parsers import BaseModelInfoParser

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_parser.log", global_level="INFO")


class ConfigFileParser(BaseModelInfoParser):
    """配置文件解析器：处理config.json文件"""
    
    def support(self, file_path: str) -> bool:
        return file_path.endswith("config.json")
    
    def parse(self, file_path: str) -> Optional[ModelInfoData]:
        try:
            raw_data = DataAccessor.read_file(file_path)
            if not raw_data:
                return None
            
            # 从文件路径推断运行类型
            run_dir = os.path.dirname(file_path)
            run_name = os.path.basename(run_dir)
            
            # 提取模型参数
            params = {
                "model_name": raw_data.get("model_name", "Unknown"),
                "device": raw_data.get("device", "cpu"),
                "num_epochs": raw_data.get("num_epochs", 0),
                "learning_rate": raw_data.get("learning_rate", 0.0),
                "batch_size": raw_data.get("batch_size", 0)
            }
            
            # 初始化metrics为空，后续可能会合并其他文件的metrics
            metrics = {
                "train_samples": raw_data.get("train_samples", 0),
                "test_samples": raw_data.get("test_samples", 0)
            }
            
            # 尝试从时间戳字段或文件路径中提取时间戳
            timestamp = DataAccessor.get_file_timestamp(file_path)
            
            return ModelInfoData(
                type="run",
                path=run_dir,
                model_type=params["model_name"],
                params=params,
                metrics=metrics,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"解析配置文件失败 {file_path}: {str(e)}")
            return None


# 导入基类（需要放在文件末尾以避免循环导入）
from src.model_visualization.model_info_parsers import BaseModelInfoParser