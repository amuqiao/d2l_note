from abc import ABC, abstractmethod
from typing import List, Optional
from src.model_visualization.data_models import MetricData
from src.model_visualization.data_access import DataAccessor
from src.utils.log_utils.log_utils import get_logger


# 初始化日志器（解析器专属日志，可选）
logger = get_logger(name=__name__, log_file="logs/parser.log", global_level="INFO")


# ==================================================
# 解析器层：负责解析不同格式的指标文件，转换为标准数据模型
# ==================================================
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
    """指标解析器注册中心：管理所有解析器，实现自动匹配"""

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
        parser = cls.get_matched_parser(file_path)
        if not parser:
            logger.warning(f"未找到匹配的解析器: {file_path}")
            return None

        try:
            return parser.parse(file_path)
        except Exception as e:
            logger.error(f"解析文件失败 {file_path}: {str(e)}")
            return None


# ==================================================
# 具体解析器实现
# ==================================================
class EpochMetricsParser(BaseMetricParser):
    """epoch_metrics.json解析器：处理训练过程中的loss和acc曲线数据"""

    def support(self, file_path: str) -> bool:
        return file_path.endswith("epoch_metrics.json")

    def parse(self, file_path: str) -> Optional[MetricData]:
        try:
            raw_data = DataAccessor.read_file(file_path)
            if not raw_data:
                return None

            return MetricData(
                metric_type="epoch_curve",
                data={
                    "epochs": [item["epoch"] for item in raw_data],
                    "train_loss": [item["train_loss"] for item in raw_data],
                    "train_acc": [item["train_acc"] for item in raw_data],
                    "test_acc": [item["test_acc"] for item in raw_data],
                },
                source_path=file_path,
                timestamp=DataAccessor.get_file_timestamp(file_path),
            )
        except Exception as e:
            logger.error(f"解析epoch指标文件失败 {file_path}: {str(e)}")
            return None


class FullMetricsParser(BaseMetricParser):
    """metrics.json解析器：处理包含完整训练信息的指标文件"""

    def support(self, file_path: str) -> bool:
        return file_path.endswith("metrics.json") and not file_path.endswith(
            "epoch_metrics.json"
        )

    def parse(self, file_path: str) -> Optional[MetricData]:
        try:
            raw_data = DataAccessor.read_file(file_path)
            if not raw_data or "epoch_metrics" not in raw_data:
                return None

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
                timestamp=DataAccessor.get_file_timestamp(file_path),
            )
        except Exception as e:
            logger.error(f"解析完整指标文件失败 {file_path}: {str(e)}")
            return None


class ConfusionMatrixParser(BaseMetricParser):
    """混淆矩阵解析器：处理confusion_matrix.json文件"""

    def support(self, file_path: str) -> bool:
        return file_path.endswith("confusion_matrix.json")

    def parse(self, file_path: str) -> Optional[MetricData]:
        try:
            raw_data = DataAccessor.read_file(file_path)
            if not raw_data:
                return None

            return MetricData(
                metric_type="confusion_matrix",
                data={
                    "classes": raw_data.get("classes", []),
                    "matrix": raw_data.get("matrix", []),
                    "accuracy": raw_data.get("accuracy", 0.0),
                },
                source_path=file_path,
                timestamp=DataAccessor.get_file_timestamp(file_path),
            )
        except Exception as e:
            logger.error(f"解析混淆矩阵文件失败 {file_path}: {str(e)}")
            return None


class LabelStatsParser(BaseMetricParser):
    """标签统计解析器：处理训练集、测试集、验证集的标签统计文件"""

    def support(self, file_path: str) -> bool:
        return file_path.endswith("_label_stats.json")

    def parse(self, file_path: str) -> Optional[MetricData]:
        try:
            raw_data = DataAccessor.read_file(file_path)
            if not raw_data:
                return None

            # 确定数据集类型（训练集、测试集或验证集）
            dataset_type = "unknown"
            if "train" in file_path.lower():
                dataset_type = "train"                                                                                                                                                      
            elif "test" in file_path.lower():
                dataset_type = "test"
            elif "val" in file_path.lower():
                dataset_type = "validation"

            return MetricData(
                metric_type="label_stats",
                data={
                    "dataset_type": dataset_type,
                    "total_samples": raw_data.get("total_samples", 0),
                    "label_counts": raw_data.get("label_counts", {}),
                    "named_label_counts": raw_data.get("named_label_counts", {})
                },
                source_path=file_path,
                timestamp=DataAccessor.get_file_timestamp(file_path),
            )
        except Exception as e:
            logger.error(f"解析标签统计文件失败 {file_path}: {str(e)}")
            return None


# ------------------------------
# 自动注册解析器（关键：导入时完成注册，无需主脚本手动调用）
# ------------------------------
MetricParserRegistry.register(EpochMetricsParser())
MetricParserRegistry.register(FullMetricsParser())
MetricParserRegistry.register(ConfusionMatrixParser())
MetricParserRegistry.register(LabelStatsParser())
