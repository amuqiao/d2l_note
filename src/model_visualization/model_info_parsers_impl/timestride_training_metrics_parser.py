from typing import Optional
import os
from abc import ABC, abstractmethod
from src.model_visualization.data_models import ModelInfoData
from src.model_visualization.data_access import DataAccessor
from src.utils.log_utils import get_logger
from src.model_visualization.model_info_parsers import BaseModelInfoParser

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_parser.log", global_level="INFO")


class TimestrideTrainingMetricsParser(BaseModelInfoParser):
    """Timestride训练指标解析器：处理training_metrics.json文件"""
    
    def support(self, file_path: str) -> bool:
        return "timestride" in file_path and file_path.endswith("training_metrics.json")
    
    def parse(self, file_path: str) -> Optional[ModelInfoData]:
        try:
            raw_data = DataAccessor.read_file(file_path)
            if not raw_data or not isinstance(raw_data, list):
                return None
            
            # 从文件路径推断运行类型
            run_dir = os.path.dirname(file_path)
            
            # 提取性能指标
            final_epoch = raw_data[-1] if raw_data else {}
            metrics = {
                "final_train_loss": final_epoch.get("train_loss", 0.0),
                "final_val_loss": final_epoch.get("vali_loss", 0.0),
                "final_test_loss": final_epoch.get("test_loss", 0.0),
                "final_val_accuracy": final_epoch.get("val_accuracy", 0.0),
                "final_test_accuracy": final_epoch.get("test_accuracy", 0.0),
                "num_epochs": len(raw_data),
                "best_test_accuracy": max([epoch.get("test_accuracy", 0.0) for epoch in raw_data]) if raw_data else 0.0
            }
            
            # 尝试从args.json获取更多信息
            args_path = os.path.join(run_dir, "args.json")
            args_data = DataAccessor.read_file(args_path) if os.path.exists(args_path) else {}
            
            params = {
                "model_name": args_data.get("model", "Unknown"),
                "task_name": args_data.get("task_name", "Unknown"),
                "batch_size": args_data.get("batch_size", 0),
                "learning_rate": args_data.get("learning_rate", 0.0),
                "train_epochs": args_data.get("train_epochs", 0),
                "device": args_data.get("device", "cpu"),
                "num_class": args_data.get("num_class", 0)
            }
            
            model_type = params["model_name"]
            
            timestamp = DataAccessor.get_file_timestamp(file_path)
            
            return ModelInfoData(
                type="run",
                path=run_dir,
                model_type=model_type,
                params=params,
                metrics=metrics,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"解析Timestride训练指标文件失败 {file_path}: {str(e)}")
            return None


# 导入基类（需要放在文件末尾以避免循环导入）
from src.model_visualization.model_info_parsers import BaseModelInfoParser