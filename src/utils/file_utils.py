"""文件操作工具类"""
import os
import json
import datetime
import glob
import re

class FileUtils:
    """文件操作工具类：提供目录创建、配置保存、指标保存等功能"""

    @staticmethod
    def create_run_dir(prefix="data/run_", root_dir=None):
        """创建时间戳唯一目录（格式：run_年日月_时分秒）
        
        参数:
            prefix: 目录前缀，默认为"run_"
            root_dir: 根目录路径，默认为执行脚本的目录（当前工作目录）
        
        返回:
            创建的目录路径
        """
        # 如果未指定根目录，使用当前工作目录
        if root_dir is None:
            root_dir = os.getcwd()
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(root_dir, f"{prefix}{timestamp}")
        os.makedirs(run_dir, exist_ok=False)  # 目录不存在则创建，避免覆盖
        print(f"✅ 创建训练目录: {run_dir}")
        return run_dir

    @staticmethod
    def save_config(config_dict, save_path):
        """保存配置到JSON文件（便于追溯训练参数）"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        print(f"📝 配置已保存: {os.path.basename(save_path)}")

    @staticmethod
    def save_metrics(metrics_dict, save_path):
        """保存训练指标到JSON文件（便于性能对比）"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
        print(f"📊 指标已保存: {os.path.basename(save_path)}")

    @staticmethod
    def find_best_model_in_dir(run_dir):
        """在训练目录中查找最佳模型文件（匹配best_model开头的.pth）"""
        model_pattern = os.path.join(run_dir, "best_model*.pth")
        model_files = glob.glob(model_pattern)
        if not model_files:
            raise FileNotFoundError(
                f"目录 {run_dir} 中未找到最佳模型（格式：best_model*.pth）"
            )
        return os.path.basename(model_files[-1])  # 默认取最后一个

    @staticmethod
    def list_models_in_dir(run_dir):
        """列出目录中的所有模型文件及其准确率信息"""
        model_pattern = os.path.join(run_dir, "best_model*.pth")
        model_files = glob.glob(model_pattern)
        if not model_files:
            raise FileNotFoundError(
                f"目录 {run_dir} 中未找到模型文件（格式：best_model*.pth）"
            )

        models_info = []
        for model_path in model_files:
            try:
                # 提取文件名中的准确率和轮次信息
                filename = os.path.basename(model_path)
                # 尝试从文件名提取准确率
                acc_match = re.search(r"acc_([0-9.]+)", filename)
                epoch_match = re.search(r"epoch_([0-9]+)", filename)

                acc = float(acc_match.group(1)) if acc_match else 0.0
                epoch = int(epoch_match.group(1)) if epoch_match else 0

                models_info.append(
                    {
                        "path": model_path,
                        "filename": filename,
                        "accuracy": acc,
                        "epoch": epoch,
                    }
                )
            except Exception:
                # 如果解析失败，仍将文件加入列表
                models_info.append(
                    {
                        "path": model_path,
                        "filename": os.path.basename(model_path),
                        "accuracy": 0.0,
                        "epoch": 0,
                    }
                )

        # 按准确率降序排序
        models_info.sort(key=lambda x: x["accuracy"], reverse=True)
        return models_info