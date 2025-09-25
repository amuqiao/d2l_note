from typing import Optional
import os
from abc import ABC, abstractmethod
from src.model_visualization.data_models import ModelInfoData
from src.model_visualization.data_access import DataAccessor
from src.utils.log_utils import get_logger
from src.model_visualization.model_info_parsers import BaseModelInfoParser
from .config_file_parser import ConfigFileParser

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_parser.log", global_level="INFO")


class MetricsFileParser(BaseModelInfoParser):
    """指标文件解析器：处理metrics.json文件"""
    
    def support(self, file_path: str) -> bool:
        return file_path.endswith("metrics.json") and not file_path.endswith("epoch_metrics.json")
    
    def parse(self, file_path: str) -> Optional[ModelInfoData]:
        try:
            raw_data = DataAccessor.read_file(file_path)
            if not raw_data:
                return None
            
            # 从文件路径推断运行类型
            run_dir = os.path.dirname(file_path)
            
            # 提取性能指标
            metrics = {
                "final_train_loss": raw_data.get("final_train_loss", 0.0),
                "final_train_acc": raw_data.get("final_train_acc", 0.0),
                "final_test_acc": raw_data.get("final_test_acc", 0.0),
                "best_test_acc": raw_data.get("best_test_acc", 0.0),
                "total_training_time": raw_data.get("total_training_time", "0s"),
                "samples_per_second": raw_data.get("samples_per_second", "0"),
                "training_start_time": raw_data.get("training_start_time", ""),
                "training_end_time": raw_data.get("training_end_time", "")
            }
            
            # 从epoch_metrics中提取关键信息
            if "epoch_metrics" in raw_data:
                metrics["num_epochs"] = len(raw_data["epoch_metrics"])
            
            # 尝试从配置文件获取更多信息
            config_path = os.path.join(run_dir, "config.json")
            config_parser = ConfigFileParser()
            config_data = config_parser.parse(config_path) if os.path.exists(config_path) else None
            
            if config_data:
                # 合并配置文件的参数
                params = config_data.params
                model_type = config_data.model_type
                # 合并指标
                merged_metrics = {**config_data.metrics, **metrics}
            else:
                params = {"model_name": "Unknown"}
                model_type = "Unknown"
                merged_metrics = metrics
            
            timestamp = DataAccessor.get_file_timestamp(file_path)
            
            return ModelInfoData(
                type="run",
                path=run_dir,
                model_type=model_type,
                params=params,
                metrics=merged_metrics,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"解析指标文件失败 {file_path}: {str(e)}")
            return None


# 导入基类（需要放在文件末尾以避免循环导入）
from src.model_visualization.model_info_parsers import BaseModelInfoParser