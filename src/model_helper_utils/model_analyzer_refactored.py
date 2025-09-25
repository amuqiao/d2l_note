import os
import json
import glob
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np
from typing import List, Dict, Optional, Any, Callable, Type
from functools import lru_cache
from abc import ABC, abstractmethod
from dataclasses import dataclass

# 解决OpenMP运行时库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入自定义日志模块
from src.utils.log_utils import get_logger

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_analysis.log", global_level="INFO")

# 导入工具注册中心
from src.helper_utils.helper_tools_registry import ToolRegistry


# ==================================================
# 核心数据模型层：标准化数据表示，增强核心数据模型地位
# ==================================================
@dataclass
class MetricData:
    """标准化指标数据模型：统一不同指标文件的内存表示"""
    metric_type: str  # 指标类型："epoch_curve" / "confusion_matrix" / "lr_curve" 等
    data: Dict[str, Any]  # 结构化指标数据
    source_path: str  # 数据来源文件路径
    timestamp: float  # 数据生成时间戳

@dataclass
class ModelInfoData:
    """模型信息数据模型：统一模型和训练任务的信息表示"""
    type: str  # "run" 或 "model"
    path: str  # 路径（目录或文件）
    model_type: str  # 模型类型
    params: Dict[str, Any]  # 模型参数
    metrics: Dict[str, Any]  # 性能指标
    timestamp: float  # 时间戳


# ==================================================
# 数据访问层：负责数据的读取和解析，提供统一的访问接口
# ==================================================
class DataAccessor:
    """数据访问器：提供统一的数据读取和解析接口"""
    
    @staticmethod
    def read_file(file_path: str) -> Optional[Any]:
        """通用文件读取方法"""
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_path.endswith('.pth'):
                # 检查PyTorch版本是否支持weights_only参数
                try:
                    return torch.load(file_path, map_location="cpu", weights_only=True)
                except TypeError:
                    return torch.load(file_path, map_location="cpu")
            else:
                logger.warning(f"不支持的文件类型: {file_path}")
                return None
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {str(e)}")
            return None

    @staticmethod
    def get_file_timestamp(file_path: str) -> float:
        """获取文件的修改时间戳"""
        if os.path.exists(file_path):
            return os.path.getmtime(file_path)
        return 0.0


class PathScanner:
    """路径扫描模块：负责查找训练目录和各类文件"""
    
    @staticmethod
    def find_run_directories(
        pattern: str = "run_\*", root_dir: str = "."  # 支持自定义根目录
    ) -> List[str]:
        """根据模式查找训练目录"""
        dir_pattern = os.path.join(root_dir, pattern)
        all_entries = glob.glob(dir_pattern)
        return [entry for entry in all_entries if os.path.isdir(entry)]

    @staticmethod
    def find_model_files(directory: str, pattern: str = "\*.pth") -> List[str]:
        """在指定目录中查找模型文件（非递归）"""
        if not os.path.exists(directory):
            return []
        return glob.glob(os.path.join(directory, pattern))

    @staticmethod
    def find_metric_files(directory: str, pattern: str = "\*.json") -> List[str]:
        """在指定目录中查找指标文件（非递归）"""
        if not os.path.exists(directory):
            return []
        return glob.glob(os.path.join(directory, pattern))

    @staticmethod
    def get_latest_run_directory(
        pattern: str = "run_\*", root_dir: str = "."
    ) -> Optional[str]:
        """获取最新修改的训练目录"""
        run_dirs = PathScanner.find_run_directories(pattern, root_dir)
        if not run_dirs:
            return None
        return max(run_dirs, key=lambda x: os.path.getmtime(x))


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
                timestamp=DataAccessor.get_file_timestamp(file_path)
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
                timestamp=DataAccessor.get_file_timestamp(file_path)
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
                    "accuracy": raw_data.get("accuracy", 0.0)
                },
                source_path=file_path,
                timestamp=DataAccessor.get_file_timestamp(file_path)
            )
        except Exception as e:
            logger.error(f"解析混淆矩阵文件失败 {file_path}: {str(e)}")
            return None


# ==================================================
# 可视化层：负责将标准化数据模型转换为可视化结果
# ==================================================
class BaseVisualizer(ABC):
    """可视化器抽象接口：定义所有可视化器的统一规范"""
    @abstractmethod
    def support(self, metric_data: MetricData) -> bool:
        """判断当前可视化器是否支持该指标数据"""
        pass

    @abstractmethod
    def visualize(self, metric_data: MetricData, show: bool = True) -> Any:
        """绘制可视化结果（返回绘图对象）"""
        pass


class VisualizerRegistry:
    """可视化器注册中心：管理所有可视化器，实现自动匹配"""
    _visualizers: List[BaseVisualizer] = []

    @classmethod
    def register(cls, visualizer: BaseVisualizer):
        """注册可视化器实例"""
        cls._visualizers.append(visualizer)

    @classmethod
    def get_matched_visualizer(cls, metric_data: MetricData) -> Optional[BaseVisualizer]:
        """根据指标数据匹配最合适的可视化器"""
        for vis in cls._visualizers:
            if vis.support(metric_data):
                return vis
        return None

    @classmethod
    def draw(cls, metric_data: MetricData, show: bool = True) -> Any:
        """自动绘制可视化结果的入口方法"""
        vis = cls.get_matched_visualizer(metric_data)
        if not vis:
            logger.warning(f"未找到匹配的可视化器: {metric_data.metric_type}")
            return None
            
        try:
            return vis.visualize(metric_data, show)
        except Exception as e:
            logger.error(f"可视化失败 {metric_data.source_path}: {str(e)}")
            return None


# ==================================================
# 具体可视化器实现
# ==================================================
class CurveVisualizer(BaseVisualizer):
    """曲线可视化器：展示训练过程中的loss和acc变化曲线"""
    def support(self, metric_data: MetricData) -> bool:
        return metric_data.metric_type == "epoch_curve"

    def visualize(self, metric_data: MetricData, show: bool = True) -> Any:
        try:
            from d2l import torch as d2l  # 延迟导入，避免不必要的依赖加载
            
            # 使用工具注册中心设置中文字体
            ToolRegistry.call("setup_font")
            data = metric_data.data
            
            animator = d2l.Animator(
                xlabel="迭代周期",
                xlim=[1, len(data["epochs"])],
                legend=["训练损失", "训练准确率", "测试准确率"],
                title=f"训练曲线 (来源: {os.path.basename(metric_data.source_path)})"
            )
            
            for i in range(len(data["epochs"])):
                animator.add(
                    data["epochs"][i],
                    (data["train_loss"][i], data["train_acc"][i], data["test_acc"][i])
                )
            
            if show:
                plt.show()
            return animator
        except Exception as e:
            logger.error(f"绘制曲线可视化失败: {str(e)}")
            return None


class ConfusionMatrixVisualizer(BaseVisualizer):
    """混淆矩阵可视化器：展示模型分类结果的混淆矩阵"""
    def support(self, metric_data: MetricData) -> bool:
        return metric_data.metric_type == "confusion_matrix"

    def visualize(self, metric_data: MetricData, show: bool = True) -> plt.Figure:
        try:
            # 使用工具注册中心设置中文字体
            ToolRegistry.call("setup_font")
            data = metric_data.data
            matrix = np.array(data["matrix"])
            classes = data["classes"]
            
            # 确保矩阵和类别数量匹配
            if len(matrix) != len(classes) or len(matrix[0]) != len(classes):
                logger.error("混淆矩阵维度与类别数量不匹配")
                return None
                
            # 绘制混淆矩阵
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(matrix, cmap="Blues")
            
            # 设置坐标轴
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            ax.set_xlabel("预测类别")
            ax.set_ylabel("真实类别")
            ax.set_title(f"混淆矩阵 (准确率: {data['accuracy']:.4f})")
            
            # 添加数值标注
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
            
            plt.colorbar(im)
            
            if show:
                plt.show()
            return fig
        except Exception as e:
            logger.error(f"绘制混淆矩阵可视化失败: {str(e)}")
            return None


# ==================================================
# 业务逻辑层：整合各模块功能，提供完整的业务逻辑处理
# ==================================================
class ModelDataProcessor:
    """模型数据处理器：负责处理模型和训练数据的核心业务逻辑"""
    
    @staticmethod
    def extract_run_metrics(run_dir: str) -> Optional[Dict[str, Any]]:
        """从metrics.json提取训练指标"""
        metrics_path = os.path.join(run_dir, "metrics.json")
        metrics = DataAccessor.read_file(metrics_path)
        
        if not metrics:
            return None

        return {
            "best_acc": float(metrics.get("best_test_acc", 0.0)),
            "final_acc": float(metrics.get("final_test_acc", 0.0)),
            "time_cost": metrics.get("total_training_time", "N/A"),
        }

    @staticmethod
    def extract_model_metrics(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """从模型检查点提取指标（兼容不同格式）"""
        return {
            "best_acc": float(
                checkpoint.get("best_test_acc", checkpoint.get("test_acc", 0.0))
            ),
            "epoch": int(checkpoint.get("epoch", 0)),
            "device": checkpoint.get("device", "N/A"),
        }

    @staticmethod
    def load_run_config(run_dir: str) -> Optional[Dict[str, Any]]:
        """从训练目录加载config.json"""
        config_path = os.path.join(run_dir, "config.json")
        return DataAccessor.read_file(config_path)

    @staticmethod
    def get_model_type(config: Optional[Dict[str, Any]]) -> str:
        """从配置中提取模型类型"""
        if not config:
            return "Unknown"
        return config.get("model_name", "Unknown")

    @staticmethod
    def create_run_info(run_dir: str) -> Optional[ModelInfoData]:
        """从训练目录生成任务级信息"""
        config = ModelDataProcessor.load_run_config(run_dir)
        metrics = ModelDataProcessor.extract_run_metrics(run_dir)

        if not config or not metrics:
            return None

        return ModelInfoData(
            type="run",
            path=run_dir,
            model_type=ModelDataProcessor.get_model_type(config),
            params={
                "lr": config.get("learning_rate"),
                "batch_size": config.get("batch_size"),
                "epochs": config.get("num_epochs"),
                "timestamp": config.get("timestamp"),
            },
            metrics=metrics,
            timestamp=DataAccessor.get_file_timestamp(run_dir)
        )

    @staticmethod
    def create_model_info(model_path: str) -> Optional[ModelInfoData]:
        """从模型文件生成快照级信息"""
        checkpoint = DataAccessor.read_file(model_path)
        if not checkpoint:
            return None

        # 尝试从模型所在目录加载配置
        model_dir = os.path.dirname(model_path)
        config = ModelDataProcessor.load_run_config(model_dir)

        return ModelInfoData(
            type="model",
            path=model_path,
            model_type=ModelDataProcessor.get_model_type(config),
            params={
                "lr": config.get("learning_rate") if config else None,
                "batch_size": config.get("batch_size") if config else None,
            },
            metrics=ModelDataProcessor.extract_model_metrics(checkpoint),
            timestamp=DataAccessor.get_file_timestamp(model_path)
        )


class ResultVisualizer:
    """结果展示模块：负责排序和格式化输出结果"""

    @staticmethod
    def sort_by_metric(
        items: List[ModelInfoData], metric_key: str = "best_acc", reverse: bool = True
    ) -> List[ModelInfoData]:
        """按指定指标排序（默认按最佳准确率降序）"""
        valid_items = [item for item in items if metric_key in item.metrics]
        return sorted(
            valid_items, key=lambda x: x.metrics[metric_key], reverse=reverse
        )

    @staticmethod
    def print_summary_table(items: List[ModelInfoData], top_n: int = 10) -> None:
        """打印格式化汇总表格（适配任务级/模型级信息）"""
        if not items:
            logger.info("❌ 没有有效数据可展示")
            return

        is_run_summary = items[0].type == "run"
        display_items = items[:top_n]

        # 表格标题
        logger.info("\n" + "=" * 120)
        logger.info(f"📊 分析结果汇总（共 {len(items)} 项，显示前 {len(display_items)} 项）")
        logger.info("=" * 120)

        # 表头（按信息类型区分）
        if is_run_summary:
            headers = [
                "排名", "目录名", "模型类型", "最佳准确率", 
                "最终准确率", "学习率", "批次大小", "训练轮次", "耗时"
            ]
            logger.info(
                f"{headers[0]:<6} {headers[1]:<22} {headers[2]:<10} {headers[3]:<12} "
                f"{headers[4]:<12} {headers[5]:<8} {headers[6]:<8} {headers[7]:<8} {headers[8]:<10}"
            )
        else:
            headers = ["排名", "文件名", "模型类型", "最佳准确率", "训练轮次", "路径"]
            logger.info(
                f"{headers[0]:<6} {headers[1]:<40} {headers[2]:<10} {headers[3]:<12} "
                f"{headers[4]:<8} {headers[5]:<30}"
            )

        logger.info("-" * 120)

        # 表格内容（带排名标记）
        for i, item in enumerate(display_items, 1):
            rank_mark = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""

            if is_run_summary:
                logger.info(
                    f"{i:<6} {os.path.basename(item.path)[:20]:<22} {item.model_type:<10} "
                    f"{item.metrics['best_acc']:.4f}    {item.metrics['final_acc']:.4f}    "
                    f"{str(item.params['lr']):<8} {str(item.params['batch_size']):<8} "
                    f"{str(item.params['epochs']):<8} {item.metrics['time_cost']:<10} {rank_mark}"
                )
            else:
                # 路径过长时截断
                path_display = (
                    item.path[:28] + "..."
                    if len(item.path) > 30
                    else item.path
                )
                filename = os.path.basename(item.path)
                logger.info(
                    f"{i:<6} {filename[:38]:<40} {item.model_type:<10} "
                    f"{item.metrics['best_acc']:.4f}    {item.metrics['epoch']:<8} "
                    f"{path_display:<30} {rank_mark}"
                )

        logger.info("=" * 120)

    @staticmethod
    def print_statistics(items: List[ModelInfoData]) -> None:
        """打印关键统计信息（最高准确率、平均值等）"""
        if not items:
            return

        valid_items = [item for item in items if "best_acc" in item.metrics]
        if not valid_items:
            logger.info("\n📈 统计信息: 无有效准确率数据")
            return

        # 计算统计指标
        best_item = max(valid_items, key=lambda x: x.metrics["best_acc"])
        avg_acc = sum(item.metrics["best_acc"] for item in valid_items) / len(valid_items)
        acc_std = (sum((item.metrics["best_acc"] - avg_acc) **2 for item in valid_items)
                  / len(valid_items))** 0.5

        logger.info("\n📈 统计信息:")
        logger.info(f"  ├─ 最高准确率: {best_item.metrics['best_acc']:.4f}")
        logger.info(f"  ├─ 平均最佳准确率: {avg_acc:.4f}")
        logger.info(f"  ├─ 准确率标准差: {acc_std:.4f}")
        if items[0].type == "run":
            logger.info(f"  └─ 最高准确率目录: {os.path.basename(best_item.path)}")
        else:
            logger.info(f"  └─ 最高准确率模型: {os.path.basename(best_item.path)}")


# ==================================================
# 服务编排层：整合各模块提供完整分析服务
# ==================================================
class ModelAnalysisService:
    """模型分析服务类：统一管理核心分析功能"""

    @staticmethod
    def summarize_runs(
        run_dir_pattern: str = "run_\*", top_n: int = 10, root_dir: str = "."
    ) -> List[ModelInfoData]:
        """汇总多个训练任务的结果"""
        logger.info(f"📊 开始汇总训练结果 (模式: {run_dir_pattern}, 根目录: {root_dir})")

        # 查找匹配目录
        run_dirs = PathScanner.find_run_directories(run_dir_pattern, root_dir)
        if not run_dirs:
            logger.info(f"❌ 未找到匹配 '{run_dir_pattern}' 的训练目录")
            return []

        logger.info(f"✅ 找到 {len(run_dirs)} 个匹配的训练目录")

        # 提取目录信息
        run_infos = []
        for dir_path in run_dirs:
            info = ModelDataProcessor.create_run_info(dir_path)
            if info:
                run_infos.append(info)
            else:
                logger.info(f"⚠️ 跳过无效目录: {os.path.basename(dir_path)}")

        if not run_infos:
            logger.info("❌ 没有有效的训练信息可汇总")
            return []

        # 排序并展示
        sorted_runs = ResultVisualizer.sort_by_metric(run_infos)
        ResultVisualizer.print_summary_table(sorted_runs, top_n)
        ResultVisualizer.print_statistics(sorted_runs)

        return sorted_runs

    @staticmethod
    def compare_models_by_dir(
        dir_pattern: str = "run_\*",
        root_dir: str = ".",
        top_n: int = 10,
        model_file_pattern: str = "\*.pth",
    ) -> List[ModelInfoData]:
        """按目录模式自动查找模型文件并比较"""
        logger.info(
            f"🔄 开始比较目录下的模型文件 "
            f"(目录模式: {dir_pattern}, 根目录: {root_dir}, 模型规则: {model_file_pattern})"
        )

        # 1. 查找符合模式的目录
        target_dirs = PathScanner.find_run_directories(dir_pattern, root_dir)
        if not target_dirs:
            logger.info(f"❌ 未找到匹配 '{dir_pattern}' 的目录（根目录: {root_dir}）")
            return []
        logger.info(f"✅ 找到 {len(target_dirs)} 个匹配目录")

        # 2. 收集所有目录下的模型文件（去重）
        model_files = set()
        for dir_path in target_dirs:
            pth_files = PathScanner.find_model_files(dir_path, model_file_pattern)
            if pth_files:
                abs_pths = [os.path.abspath(pth) for pth in pth_files]
                model_files.update(abs_pths)
                logger.info(
                    f"  ├─ 目录 {os.path.basename(dir_path)}: 找到 {len(pth_files)} 个模型文件"
                )
            else:
                logger.info(f"  ├─ 目录 {os.path.basename(dir_path)}: 未找到模型文件，跳过")

        model_files_list = list(model_files)
        if not model_files_list:
            logger.info("❌ 未收集到任何有效模型文件")
            return []
        logger.info(f"✅ 共收集到 {len(model_files_list)} 个唯一模型文件")

        # 3. 提取模型信息并展示
        model_infos = []
        for pth_path in model_files_list:
            info = ModelDataProcessor.create_model_info(pth_path)
            if info:
                model_infos.append(info)
            else:
                logger.info(f"⚠️ 跳过无效模型: {os.path.basename(pth_path)}")

        if not model_infos:
            logger.info("❌ 没有可比较的有效模型信息")
            return []

        # 4. 排序并展示
        sorted_models = ResultVisualizer.sort_by_metric(model_infos)
        ResultVisualizer.print_summary_table(sorted_models, top_n)
        ResultVisualizer.print_statistics(sorted_models)

        return sorted_models

    @staticmethod
    def compare_latest_models(
        pattern: str = "run_\*",  # 目录过滤模式
        num_latest: int = 5,  # 取最新N个目录
        root_dir: str = ".",  # 根搜索目录
    ) -> List[ModelInfoData]:
        """比较指定模式下最新N个训练目录中的最佳模型"""
        logger.info(
            f"🔍 比较最新的 {num_latest} 个训练目录中的最佳模型 "
            f"(目录模式: {pattern}, 根目录: {root_dir})"
        )

        # 1. 查找符合模式的目录（按修改时间倒序）
        matched_dirs = PathScanner.find_run_directories(pattern, root_dir)
        if not matched_dirs:
            logger.info(f"❌ 未找到匹配 '{pattern}' 的训练目录（根目录: {root_dir}）")
            return []

        # 按修改时间排序，取最新的num_latest个目录
        sorted_dirs = sorted(matched_dirs, key=lambda x: os.path.getmtime(x), reverse=True)
        latest_dirs = sorted_dirs[:num_latest]

        if not latest_dirs:
            logger.info("❌ 没有符合条件的最新目录")
            return []
        logger.info(f"✅ 找到最新的 {len(latest_dirs)} 个目录")

        # 2. 提取每个目录的最佳模型（优先找best_model*.pth）
        model_files = []
        for dir_path in latest_dirs:
            best_models = PathScanner.find_model_files(dir_path, "best_model\*.pth")
            if best_models:
                # 取目录中最新修改的最佳模型
                latest_model = max(best_models, key=lambda x: os.path.getmtime(x))
                model_files.append(latest_model)
                logger.info(
                    f"  ├─ 目录 {os.path.basename(dir_path)}: 最新最佳模型 {os.path.basename(latest_model)}"
                )
            else:
                logger.info(f"⚠️ 目录 {os.path.basename(dir_path)}: 未找到best_model*.pth，跳过")

        if not model_files:
            logger.info("❌ 没有找到可比较的最佳模型文件")
            return []

        # 3. 提取模型信息并展示
        model_infos = []
        for model_path in model_files:
            info = ModelDataProcessor.create_model_info(model_path)
            if info:
                model_infos.append(info)
            else:
                logger.info(f"⚠️ 跳过无效模型文件: {os.path.basename(model_path)}")

        if not model_infos:
            logger.info("❌ 没有可比较的有效模型信息")
            return []

        # 4. 排序并展示结果
        sorted_models = ResultVisualizer.sort_by_metric(model_infos)
        ResultVisualizer.print_summary_table(sorted_models, len(sorted_models))
        ResultVisualizer.print_statistics(sorted_models)

        return sorted_models

    @staticmethod
    def visualize_training_metrics(run_dir=None, metrics_path=None, root_dir="."):
        """通过注册中心实现自动化可视化"""
        logger.info("\n🎨 开始可视化训练指标...")

        # 1. 确定目标文件
        target_files = []
        
        if metrics_path:
            # 使用指定的指标文件
            if not os.path.isabs(metrics_path):
                metrics_path = os.path.join(root_dir, metrics_path)
            target_files = [metrics_path]
            logger.info(f"  ├─ 使用指定的指标文件: {metrics_path}")
        elif run_dir:
            # 使用指定目录下的所有指标文件
            if not os.path.isabs(run_dir):
                run_dir = os.path.join(root_dir, run_dir)
            if not os.path.exists(run_dir):
                logger.error(f"  └─ 指定的训练目录不存在: {run_dir}")
                return []
                
            target_files = PathScanner.find_metric_files(run_dir)
            logger.info(f"  ├─ 从指定目录加载指标文件: {run_dir} (找到 {len(target_files)} 个)")
        else:
            # 自动查找最新训练目录
            logger.info("  ├─ 未指定目录或文件，自动查找最新训练目录...")
            latest_dir = PathScanner.get_latest_run_directory("run_\*", root_dir)
            
            if not latest_dir:
                logger.info("  └─ ❌ 未找到任何训练目录（需以'run_'开头）")
                return []
                
            target_files = PathScanner.find_metric_files(latest_dir)
            logger.info(f"  └─ 自动选择最新目录: {latest_dir} (找到 {len(target_files)} 个指标文件)")

        if not target_files:
            logger.info("  └─ ❌ 未找到任何指标文件")
            return []

        # 2. 批量解析指标（通过解析器注册中心）
        metric_datas = []
        for file in target_files:
            data = MetricParserRegistry.parse_file(file)
            if data:
                metric_datas.append(data)

        if not metric_datas:
            logger.info("  └─ ❌ 没有解析成功的指标数据")
            return []

        # 3. 批量可视化（通过可视化器注册中心）
        results = []
        for data in metric_datas:
            logger.info(f"  ├─ 可视化指标: {data.metric_type} (来源: {os.path.basename(data.source_path)})")
            vis_result = VisualizerRegistry.draw(data)
            if vis_result:
                results.append(vis_result)

        logger.info(f"  └─ ✅ 完成可视化，共处理 {len(results)} 个指标")
        return results


# ==================================================
# 主函数入口
# ==================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="模型分析工具：汇总训练结果、比较模型性能、可视化训练指标"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="summarize",
        choices=["summarize", "compare", "latest", "analyze"],
        help="运行模式: "
        "summarize(汇总训练目录), "
        "compare(按目录查找模型并比较), "
        "latest(比较最新N个目录的最佳模型), "
        "analyze(可视化训练指标)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="runs/run_*",
        help="目录匹配模式（summarize/compare/latest模式均生效）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="显示的结果数量（summarize/compare模式生效）",
    )
    parser.add_argument(
        "--num-latest", type=int, default=5, help="取最新的目录数量（仅latest模式生效）"
    )
    parser.add_argument(
        "--root-dir", type=str, default=".", help="根搜索目录（所有模式均生效）"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None, help="训练目录路径（仅analyze模式生效）"
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="指标文件路径（仅analyze模式生效）",
    )

    args = parser.parse_args()

    # 按模式调用ModelAnalysisService的对应静态方法
    if args.mode == "summarize":
        ModelAnalysisService.summarize_runs(args.pattern, args.top_n, args.root_dir)
    elif args.mode == "compare":
        ModelAnalysisService.compare_models_by_dir(
            dir_pattern=args.pattern, root_dir=args.root_dir, top_n=args.top_n
        )
    elif args.mode == "latest":
        ModelAnalysisService.compare_latest_models(
            pattern=args.pattern, num_latest=args.num_latest, root_dir=args.root_dir
        )
    elif args.mode == "analyze":
        ModelAnalysisService.visualize_training_metrics(
            run_dir=args.run_dir, metrics_path=args.metrics_path, root_dir=args.root_dir
        )


if __name__ == "__main__":
    main()