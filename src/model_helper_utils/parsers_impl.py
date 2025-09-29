import os
import json
from typing import Optional

# 导入自定义日志模块
from src.utils.log_utils.log_utils import get_logger

# 导入基础数据结构和接口
from src.model_helper_utils.metric_parsers import MetricData, BaseMetricParser

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_analysis.log", global_level="INFO")


class EpochMetricsParser(BaseMetricParser):
    """epoch_metrics.json解析器：处理训练过程中的loss和acc曲线数据"""
    def support(self, file_path: str) -> bool:
        return file_path.endswith("epoch_metrics.json")

    def parse(self, file_path: str) -> Optional[MetricData]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            return MetricData(
                metric_type="epoch_curve",
                data={
                    "epochs": [item["epoch"] for item in raw_data],
                    "train_loss": [item["train_loss"] for item in raw_data],
                    "train_acc": [item["train_acc"] for item in raw_data],
                    "test_acc": [item["test_acc"] for item in raw_data],
                },
                source_path=file_path,
                timestamp=os.path.getmtime(file_path)
            )
        except Exception as e:
            logger.error(f"解析epoch指标文件失败 {file_path}: {str(e)}")
            return None


class FullMetricsParser(BaseMetricParser):
    """metrics.json解析器：处理包含完整训练信息的指标文件"""
    def support(self, file_path: str) -> bool:
        return file_path.endswith("metrics.json") and not file_path.endswith("epoch_metrics.json")

    def parse(self, file_path: str) -> Optional[MetricData]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            if "epoch_metrics" in raw_data:
                epoch_data = raw_data["epoch_metrics"]
                return MetricData(
                    metric_type="epoch_curve",
                    data={
                        "epochs": [item["epoch"] for item in epoch_data],
                        "train_loss": [item["train_loss"] for item in epoch_data],
                        "train_acc": [item["train_acc"] for item in epoch_data],
                        "test_acc": [item["test_acc"] for item in epoch_data],
                    },
                    source_path=file_path,
                    timestamp=os.path.getmtime(file_path)
                )
            return None
        except Exception as e:
            logger.error(f"解析完整指标文件失败 {file_path}: {str(e)}")
            return None


class ConfusionMatrixParser(BaseMetricParser):
    """混淆矩阵解析器：处理confusion_matrix.json文件"""
    def support(self, file_path: str) -> bool:
        return file_path.endswith("confusion_matrix.json")

    def parse(self, file_path: str) -> Optional[MetricData]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            return MetricData(
                metric_type="confusion_matrix",
                data={
                    "classes": raw_data.get("classes", []),
                    "matrix": raw_data.get("matrix", []),
                    "accuracy": raw_data.get("accuracy", 0.0)
                },
                source_path=file_path,
                timestamp=os.path.getmtime(file_path)
            )
        except Exception as e:
            logger.error(f"解析混淆矩阵文件失败 {file_path}: {str(e)}")
            return None