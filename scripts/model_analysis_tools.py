import os
import json
import glob
import torch
import matplotlib.pyplot as plt
import sys
from typing import List, Dict, Optional, Any, Callable
from functools import lru_cache  # 用于路径查找缓存

# 解决OpenMP运行时库冲突问题
# 设置环境变量允许多个OpenMP运行时库共存
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BaseTool:
    """基础工具类：提供通用功能，作为其他工具类的基类"""

    @staticmethod
    def setup_font():
        """配置matplotlib中文显示"""
        if sys.platform.startswith("win"):
            plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
        elif sys.platform.startswith("darwin"):
            plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC"]
        elif sys.platform.startswith("linux"):
            plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


class PathScanner:
    """路径扫描模块：负责查找训练目录和模型文件（统一用glob匹配）"""

    @staticmethod
    @lru_cache(maxsize=None)  # 缓存目录查找结果，减少重复IO
    def find_run_directories(
        pattern: str = "run_*", root_dir: str = "."  # 支持自定义根目录
    ) -> List[str]:
        """根据模式查找训练目录"""
        dir_pattern = os.path.join(root_dir, pattern)
        all_entries = glob.glob(dir_pattern)
        return [entry for entry in all_entries if os.path.isdir(entry)]

    @staticmethod
    def find_model_files(directory: str, pattern: str = "*.pth") -> List[str]:
        """在指定目录中查找模型文件（非递归）"""
        if not os.path.exists(directory):
            return []
        return glob.glob(os.path.join(directory, pattern))

    @staticmethod
    def get_latest_run_directory(
        pattern: str = "run_*", root_dir: str = "."
    ) -> Optional[str]:
        """获取最新修改的训练目录"""
        run_dirs = PathScanner.find_run_directories(pattern, root_dir)
        if not run_dirs:
            return None
        return max(run_dirs, key=lambda x: os.path.getmtime(x))


class ConfigLoader:
    """配置加载模块：负责加载和解析配置文件与模型检查点"""

    @staticmethod
    def _load_file_safely(
        loader: Callable[[str], Any], file_path: str, error_prefix: str
    ) -> Optional[Any]:
        """通用安全加载函数：统一处理文件不存在和加载异常"""
        if not os.path.exists(file_path):
            return None
        try:
            return loader(file_path)
        except Exception as e:
            print(f"⚠️ {error_prefix} {file_path}: {str(e)}")
            return None

    @staticmethod
    def load_run_config(run_dir: str) -> Optional[Dict[str, Any]]:
        """从训练目录加载config.json"""
        config_path = os.path.join(run_dir, "config.json")

        def _json_loader(path: str) -> Dict[str, Any]:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        return ConfigLoader._load_file_safely(
            _json_loader, config_path, "加载配置文件失败"
        )

    @staticmethod
    def load_model_checkpoint(model_path: str) -> Optional[Dict[str, Any]]:
        """加载模型检查点（CPU加载避免设备不匹配）"""

        def _model_loader(path: str) -> Dict[str, Any]:
            # 检查PyTorch版本是否支持weights_only参数（PyTorch 1.11.0及以上版本支持）
            try:
                # 尝试使用weights_only参数
                return torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                # 如果报错（不支持该参数），则不使用weights_only
                return torch.load(path, map_location="cpu")

        return ConfigLoader._load_file_safely(
            _model_loader, model_path, "加载模型文件失败"
        )

    @staticmethod
    def get_model_type(config: Optional[Dict[str, Any]]) -> str:
        """从配置中提取模型类型"""
        if not config:
            return "Unknown"
        return config.get("model_name", "Unknown")


class MetricExtractor:
    """指标提取模块：负责从文件中提取模型性能指标"""

    @staticmethod
    def extract_run_metrics(run_dir: str) -> Optional[Dict[str, Any]]:
        """从metrics.json提取训练指标"""
        metrics_path = os.path.join(run_dir, "metrics.json")

        def _metrics_loader(path: str) -> Dict[str, Any]:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        metrics = ConfigLoader._load_file_safely(
            _metrics_loader, metrics_path, "提取运行指标失败"
        )
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


class ModelInfo:
    """模型信息模块：整合路径、配置和指标信息"""

    @staticmethod
    def from_run_directory(run_dir: str) -> Optional[Dict[str, Any]]:
        """从训练目录生成任务级信息"""
        config = ConfigLoader.load_run_config(run_dir)
        metrics = MetricExtractor.extract_run_metrics(run_dir)

        if not config or not metrics:
            return None

        return {
            "type": "run",
            "dir": run_dir,
            "model_type": ConfigLoader.get_model_type(config),
            "params": {
                "lr": config.get("learning_rate"),
                "batch_size": config.get("batch_size"),
                "epochs": config.get("num_epochs"),
                "timestamp": config.get("timestamp"),
            },
            "metrics": metrics,
        }

    @staticmethod
    def from_model_file(model_path: str) -> Optional[Dict[str, Any]]:
        """从模型文件生成快照级信息"""
        checkpoint = ConfigLoader.load_model_checkpoint(model_path)
        if not checkpoint:
            return None

        # 尝试从模型所在目录加载配置
        model_dir = os.path.dirname(model_path)
        config = ConfigLoader.load_run_config(model_dir)

        return {
            "type": "model",
            "path": model_path,
            "filename": os.path.basename(model_path),
            "model_type": ConfigLoader.get_model_type(config),
            "params": {
                "lr": config.get("learning_rate") if config else None,
                "batch_size": config.get("batch_size") if config else None,
            },
            "metrics": MetricExtractor.extract_model_metrics(checkpoint),
        }


class ResultVisualizer:
    """结果展示模块：负责排序和格式化输出结果"""

    @staticmethod
    def sort_by_metric(
        items: List[Dict[str, Any]], metric_key: str = "best_acc", reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """按指定指标排序（默认按最佳准确率降序）"""
        valid_items = [item for item in items if metric_key in item["metrics"]]
        return sorted(
            valid_items, key=lambda x: x["metrics"][metric_key], reverse=reverse
        )

    @staticmethod
    def print_summary_table(items: List[Dict[str, Any]], top_n: int = 10) -> None:
        """打印格式化汇总表格（适配任务级/模型级信息）"""
        if not items:
            print("❌ 没有有效数据可展示")
            return

        is_run_summary = items[0]["type"] == "run"
        display_items = items[:top_n]

        # 表格标题
        print("\n" + "=" * 120)
        print(f"📊 分析结果汇总（共 {len(items)} 项，显示前 {len(display_items)} 项）")
        print("=" * 120)

        # 表头（按信息类型区分）
        if is_run_summary:
            headers = [
                "排名",
                "目录名",
                "模型类型",
                "最佳准确率",
                "最终准确率",
                "学习率",
                "批次大小",
                "训练轮次",
                "耗时",
            ]
            print(
                f"{headers[0]:<6} {headers[1]:<22} {headers[2]:<10} {headers[3]:<12} {headers[4]:<12} {headers[5]:<8} {headers[6]:<8} {headers[7]:<8} {headers[8]:<10}"
            )
        else:
            headers = ["排名", "文件名", "模型类型", "最佳准确率", "训练轮次", "路径"]
            print(
                f"{headers[0]:<6} {headers[1]:<40} {headers[2]:<10} {headers[3]:<12} {headers[4]:<8} {headers[5]:<30}"
            )

        print("-" * 120)

        # 表格内容（带排名标记）
        for i, item in enumerate(display_items, 1):
            rank_mark = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""

            if is_run_summary:
                print(
                    f"{i:<6} {os.path.basename(item['dir'])[:20]:<22} {item['model_type']:<10} "
                    f"{item['metrics']['best_acc']:.4f}    {item['metrics']['final_acc']:.4f}    "
                    f"{str(item['params']['lr']):<8} {str(item['params']['batch_size']):<8} "
                    f"{str(item['params']['epochs']):<8} {item['metrics']['time_cost']:<10} {rank_mark}"
                )
            else:
                # 路径过长时截断
                path_display = (
                    item["path"][:28] + "..."
                    if len(item["path"]) > 30
                    else item["path"]
                )
                print(
                    f"{i:<6} {item['filename'][:38]:<40} {item['model_type']:<10} "
                    f"{item['metrics']['best_acc']:.4f}    {item['metrics']['epoch']:<8} "
                    f"{path_display:<30} {rank_mark}"
                )

        print("=" * 120)

    @staticmethod
    def print_statistics(items: List[Dict[str, Any]]) -> None:
        """打印关键统计信息（最高准确率、平均值等）"""
        if not items:
            return

        valid_items = [item for item in items if "best_acc" in item["metrics"]]
        if not valid_items:
            print("\n📈 统计信息: 无有效准确率数据")
            return

        # 计算统计指标
        best_item = max(valid_items, key=lambda x: x["metrics"]["best_acc"])
        avg_acc = sum(item["metrics"]["best_acc"] for item in valid_items) / len(
            valid_items
        )
        acc_std = (
            sum((item["metrics"]["best_acc"] - avg_acc) ** 2 for item in valid_items)
            / len(valid_items)
        ) ** 0.5

        print("\n📈 统计信息:")
        print(f"  ├─ 最高准确率: {best_item['metrics']['best_acc']:.4f}")
        print(f"  ├─ 平均最佳准确率: {avg_acc:.4f}")
        print(f"  ├─ 准确率标准差: {acc_std:.4f}")
        if items[0]["type"] == "run":
            print(f"  └─ 最高准确率目录: {os.path.basename(best_item['dir'])}")
        else:
            print(f"  └─ 最高准确率模型: {best_item['filename']}")


class MetricsVisualizer(BaseTool):
    """指标可视化模块：负责生成训练指标的可视化图表"""

    @staticmethod
    def visualize_from_epoch_metrics(data, show=True):
        """从epoch_metrics数据或文件生成动画"""
        import matplotlib.pyplot as plt
        from d2l import torch as d2l

        # 设置中文字体
        MetricsVisualizer.setup_font()

        # 加载指标数据
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as f:
                epoch_metrics = json.load(f)
        else:
            epoch_metrics = data

        # 创建动画器
        animator = d2l.Animator(
            xlabel="迭代周期",
            xlim=[1, len(epoch_metrics)],
            legend=["训练损失", "训练准确率", "测试准确率"],
        )

        # 逐轮次添加数据点
        for metric in epoch_metrics:
            animator.add(
                metric["epoch"],
                (metric["train_loss"], metric["train_acc"], metric["test_acc"]),
            )

        # 显示最终结果
        if show:
            plt.title(f"训练指标可视化 (共{len(epoch_metrics)}轮)")
            plt.show()
        return animator

    @staticmethod
    def visualize_from_metrics_json(metrics_path, show=True):
        """从完整的metrics.json文件生成动画"""
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        if "epoch_metrics" in metrics:
            return MetricsVisualizer.visualize_from_epoch_metrics(
                metrics["epoch_metrics"], show
            )
        return None

    @staticmethod
    def visualize_from_run_dir(run_dir, show=True):
        """从训练目录生成动画"""
        print(f"📂 正在从训练目录加载指标: {run_dir}")

        # 优先尝试从metrics.json加载
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            print(f"  ├─ 发现metrics.json文件")
            try:
                return MetricsVisualizer.visualize_from_metrics_json(metrics_path, show)
            except Exception as e:
                print(f"  └─ ⚠️ 从metrics.json加载失败: {str(e)}")

        # 如果失败，尝试从epoch_metrics.json加载
        epoch_metrics_path = os.path.join(run_dir, "epoch_metrics.json")
        if os.path.exists(epoch_metrics_path):
            print(f"  ├─ 发现epoch_metrics.json文件")
            try:
                return MetricsVisualizer.visualize_from_epoch_metrics(
                    epoch_metrics_path, show
                )
            except Exception as e:
                print(f"  └─ ⚠️ 从epoch_metrics.json加载失败: {str(e)}")

        print(f"  └─ ❌ 在目录 {run_dir} 中未找到可用的指标文件")
        return None


class ModelAnalysisService:
    """模型分析服务类：统一管理核心分析功能（汇总、比较、可视化）"""

    @staticmethod
    def summarize_runs(
        run_dir_pattern: str = "run_*", top_n: int = 10, root_dir: str = "."
    ) -> List[Dict[str, Any]]:
        """汇总多个训练任务的结果"""
        print(f"📊 开始汇总训练结果 (模式: {run_dir_pattern}, 根目录: {root_dir})")

        # 查找匹配目录
        run_dirs = PathScanner.find_run_directories(run_dir_pattern, root_dir)
        if not run_dirs:
            print(f"❌ 未找到匹配 '{run_dir_pattern}' 的训练目录")
            return []

        print(f"✅ 找到 {len(run_dirs)} 个匹配的训练目录")

        # 提取目录信息
        run_infos = []
        for dir_path in run_dirs:
            info = ModelInfo.from_run_directory(dir_path)
            if info:
                run_infos.append(info)
            else:
                print(f"⚠️ 跳过无效目录: {os.path.basename(dir_path)}")

        if not run_infos:
            print("❌ 没有有效的训练信息可汇总")
            return []

        # 排序并展示
        sorted_runs = ResultVisualizer.sort_by_metric(run_infos)
        ResultVisualizer.print_summary_table(sorted_runs, top_n)
        ResultVisualizer.print_statistics(sorted_runs)

        return sorted_runs

    @staticmethod
    def compare_models_by_dir(
        dir_pattern: str = "run_*",
        root_dir: str = ".",
        top_n: int = 10,
        model_file_pattern: str = "*.pth",
    ) -> List[Dict[str, Any]]:
        """按目录模式自动查找模型文件并比较（compare模式核心函数）"""
        print(
            f"🔄 开始比较目录下的模型文件 "
            f"(目录模式: {dir_pattern}, 根目录: {root_dir}, 模型规则: {model_file_pattern})"
        )

        # 1. 查找符合模式的目录
        target_dirs = PathScanner.find_run_directories(dir_pattern, root_dir)
        if not target_dirs:
            print(f"❌ 未找到匹配 '{dir_pattern}' 的目录（根目录: {root_dir}）")
            return []
        print(f"✅ 找到 {len(target_dirs)} 个匹配目录:")
        for dir_path in target_dirs:
            print(f"  ├─ {os.path.basename(dir_path)}")

        # 2. 收集所有目录下的模型文件（去重）
        model_files = set()
        for dir_path in target_dirs:
            pth_files = PathScanner.find_model_files(dir_path, model_file_pattern)
            if pth_files:
                abs_pths = [os.path.abspath(pth) for pth in pth_files]
                model_files.update(abs_pths)
                print(
                    f"  ├─ 目录 {os.path.basename(dir_path)}: 找到 {len(pth_files)} 个模型文件"
                )
            else:
                print(f"  ├─ 目录 {os.path.basename(dir_path)}: 未找到模型文件，跳过")

        model_files_list = list(model_files)
        if not model_files_list:
            print("❌ 未收集到任何有效模型文件")
            return []
        print(f"✅ 共收集到 {len(model_files_list)} 个唯一模型文件")

        # 3. 提取模型信息
        model_infos = []
        for pth_path in model_files_list:
            info = ModelInfo.from_model_file(pth_path)
            if info:
                model_infos.append(info)
            else:
                print(f"⚠️ 跳过无效模型: {os.path.basename(pth_path)}")

        if not model_infos:
            print("❌ 没有可比较的有效模型信息")
            return []

        # 4. 排序并展示
        sorted_models = ResultVisualizer.sort_by_metric(model_infos)
        ResultVisualizer.print_summary_table(sorted_models, top_n)
        ResultVisualizer.print_statistics(sorted_models)

        return sorted_models

    @staticmethod
    def compare_latest_models(
        pattern: str = "run_*",  # 目录过滤模式
        num_latest: int = 5,  # 取最新N个目录
        root_dir: str = ".",  # 根搜索目录
    ) -> List[Dict[str, Any]]:
        """比较指定模式下最新N个训练目录中的最佳模型"""
        print(
            f"🔍 比较最新的 {num_latest} 个训练目录中的最佳模型 "
            f"(目录模式: {pattern}, 根目录: {root_dir})"
        )

        # 1. 查找符合模式的目录（按修改时间倒序）
        matched_dirs = PathScanner.find_run_directories(pattern, root_dir)
        if not matched_dirs:
            print(f"❌ 未找到匹配 '{pattern}' 的训练目录（根目录: {root_dir}）")
            return []

        # 按修改时间排序，取最新的num_latest个目录
        sorted_dirs = sorted(matched_dirs, key=lambda x: os.path.getmtime(x), reverse=True)
        latest_dirs = sorted_dirs[:num_latest]

        if not latest_dirs:
            print("❌ 没有符合条件的最新目录")
            return []
        print(f"✅ 找到最新的 {len(latest_dirs)} 个目录:")
        for dir_path in latest_dirs:
            mod_time = os.path.getmtime(dir_path)
            print(
                f"  ├─ {os.path.basename(dir_path)} (最后修改: {os.path.getctime(dir_path):.0f} 时间戳)"
            )

        # 2. 提取每个目录的最佳模型（优先找best_model*.pth）
        model_files = []
        for dir_path in latest_dirs:
            best_models = PathScanner.find_model_files(dir_path, "best_model*.pth")
            if best_models:
                # 取目录中最新修改的最佳模型
                latest_model = max(best_models, key=lambda x: os.path.getmtime(x))
                model_files.append(latest_model)
                print(
                    f"  ├─ 目录 {os.path.basename(dir_path)}: 最新最佳模型 {os.path.basename(latest_model)}"
                )
            else:
                print(f"⚠️ 目录 {os.path.basename(dir_path)}: 未找到best_model*.pth，跳过")

        if not model_files:
            print("❌ 没有找到可比较的最佳模型文件")
            return []

        # 3. 提取模型信息（整合原compare_models逻辑）
        model_infos = []
        for model_path in model_files:
            info = ModelInfo.from_model_file(model_path)
            if info:
                model_infos.append(info)
            else:
                print(f"⚠️ 跳过无效模型文件: {os.path.basename(model_path)}")

        if not model_infos:
            print("❌ 没有可比较的有效模型信息")
            return []

        # 4. 排序并展示结果
        sorted_models = ResultVisualizer.sort_by_metric(model_infos)
        ResultVisualizer.print_summary_table(
            sorted_models, len(sorted_models)
        )  # 显示所有找到的模型
        ResultVisualizer.print_statistics(sorted_models)

        return sorted_models

    @staticmethod
    def visualize_training_metrics(run_dir=None, metrics_path=None, root_dir="."):
        """可视化训练指标，采用优先级策略加载数据

        参数:
            run_dir: 训练目录路径（优先级最高）
            metrics_path: 指标文件路径（优先级次之）
            root_dir: 根搜索目录（当需要自动查找时使用）

        返回:
            d2l.Animator实例或None
        """
        print("\n🎨 开始可视化训练指标...")

        # 根据优先级确定数据源
        data_source = None
        source_type = None

        # 优先级1: 使用指定的训练目录
        if run_dir:
            if not os.path.isabs(run_dir):
                run_dir = os.path.join(root_dir, run_dir)
            data_source = run_dir
            source_type = "dir"
            print(f"  ├─ 优先级1: 使用指定的训练目录: {data_source}")
        # 优先级2: 使用指定的指标文件
        elif metrics_path:
            if not os.path.isabs(metrics_path):
                metrics_path = os.path.join(root_dir, metrics_path)
            data_source = metrics_path
            if metrics_path.endswith("epoch_metrics.json"):
                source_type = "epoch_metrics"
                print(f"  ├─ 优先级2: 使用指定的epoch_metrics.json文件: {data_source}")
            elif metrics_path.endswith("metrics.json"):
                source_type = "metrics"
                print(f"  ├─ 优先级2: 使用指定的metrics.json文件: {data_source}")
            else:
                print(f"  └─ ❌ 不支持的指标文件类型: {metrics_path}")
                return None
        # 优先级3: 自动查找最新训练目录
        else:
            print("  ├─ 优先级3: 未指定目录或文件，自动查找最新训练目录...")
            latest_dir = PathScanner.get_latest_run_directory("run_*", root_dir)
            if not latest_dir:
                print("  └─ ❌ 未找到任何训练目录（需以'run_'开头）")
                return None

            data_source = latest_dir
            source_type = "dir"
            print(f"  └─ ✅ 自动选择最新目录: {data_source}")

        # 根据数据源类型调用对应的可视化方法
        print("  ┌─ 开始加载指标数据...")
        if source_type == "dir":
            animator = MetricsVisualizer.visualize_from_run_dir(data_source)
        elif source_type == "epoch_metrics":
            animator = MetricsVisualizer.visualize_from_epoch_metrics(data_source)
        elif source_type == "metrics":
            animator = MetricsVisualizer.visualize_from_metrics_json(data_source)
        else:
            animator = None

        # 确认结果
        if animator:
            print("  └─ ✅ 训练指标可视化完成")
        else:
            print("  └─ ❌ 训练指标可视化失败")

        return animator


# ==============================================================================
# 主函数入口（调用统一的ModelAnalysisService）
# ==============================================================================
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