"""文件操作工具类"""
import os
import json
import datetime
import glob
import re
from typing import Optional, List, Dict, Any

class FileUtils:
    """文件操作工具类：提供目录创建、配置保存、指标保存等功能"""

    @staticmethod
    def create_run_dir(prefix="runs/run_", root_dir=None):
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

    @staticmethod
    def find_directories_by_pattern(
        dir_prefix: str = "run_",
        search_dirs: List[str] = None,
        root_dir: Optional[str] = None
    ) -> List[str]:
        """
        在指定搜索目录中查找符合前缀的子目录
        
        Args:
            dir_prefix: 目录名称前缀，默认为"run_"
            search_dirs: 可选，自定义搜索目录列表
            root_dir: 可选，根目录路径，当未提供search_dirs时使用
        
        Returns:
            符合条件的目录绝对路径列表
        """
        # 如果未提供搜索目录列表，使用默认值
        if search_dirs is None:
            if root_dir is None:
                root_dir = os.getcwd()
            search_dirs = [
                root_dir,  # 首先在root_dir目录下查找
                os.path.join(root_dir, "data")  # 然后在root_dir/data目录下查找
            ]
        
        # 存储所有找到的指定前缀的目录
        matching_dirs = []
        
        # 遍历搜索目录
        for search_dir in search_dirs:
            try:
                if os.path.exists(search_dir) and os.path.isdir(search_dir):
                    dirs_in_search = [
                        os.path.abspath(os.path.join(search_dir, d))
                        for d in os.listdir(search_dir) 
                        if os.path.isdir(os.path.join(search_dir, d)) and d.startswith(dir_prefix)
                    ]
                    matching_dirs.extend(dirs_in_search)
            except Exception as e:
                print(f"⚠️ 搜索目录 {search_dir} 时出错: {str(e)}")
        
        return matching_dirs
    
    @staticmethod
    def select_latest_directory(directories: List[str]) -> str:
        """
        从目录列表中选择最新创建或修改的目录
        
        Args:
            directories: 目录路径列表
            
        Returns:
            最新目录的路径
            
        Raises:
            ValueError: 如果目录列表为空
        """
        if not directories:
            raise ValueError("目录列表不能为空")
        
        # 按修改时间排序，选择最新的目录
        directories.sort(key=os.path.getmtime, reverse=True)
        return directories[0]
    
    @staticmethod
    def find_latest_run_dir(
        root_dir: Optional[str] = None,
        search_dirs: Optional[List[str]] = None,
        dir_prefix: str = "run_"
    ) -> str:
        """
        自动查找最新的训练目录（整合目录查找和选择功能）
        
        Args:
            root_dir: 根目录路径
            search_dirs: 可选，自定义搜索目录列表
            dir_prefix: 目录名称前缀，默认为"run_"
            
        Returns:
            最新训练目录的路径
            
        Raises:
            FileNotFoundError: 如果未找到任何符合条件的目录
        """
        # 查找符合条件的目录
        all_dirs = FileUtils.find_directories_by_pattern(
            dir_prefix=dir_prefix,
            search_dirs=search_dirs,
            root_dir=root_dir
        )
        
        # 选择最新的目录
        if all_dirs:
            latest_dir = FileUtils.select_latest_directory(all_dirs)
            print(f"✅ 自动选择最新{dir_prefix}目录: {latest_dir}")
            return latest_dir
        else:
            # 如果没找到，提供更详细的错误信息
            if search_dirs is None:
                if root_dir is None:
                    root_dir = os.getcwd()
                search_dirs = [root_dir, os.path.join(root_dir, "data")]
            
            searched_paths = ", ".join([os.path.join(d, f"{dir_prefix}*") for d in search_dirs])
            raise FileNotFoundError(f"未找到任何{dir_prefix}目录\n已搜索路径: {searched_paths}")

    @staticmethod
    def validate_directory(path: str) -> None:
        """
        验证目录是否存在
        Args:
            path: 目录路径
        Raises:
            FileNotFoundError: 如果目录不存在
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"目录不存在: {path}")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"路径不是一个目录: {path}")

    @staticmethod
    def validate_file(path: str, expected_extension: Optional[str] = None) -> None:
        """
        验证文件是否存在以及文件格式是否正确
        Args:
            path: 文件路径
            expected_extension: 预期的文件扩展名
        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果文件格式不正确
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        if not os.path.isfile(path):
            raise IsADirectoryError(f"路径不是一个文件: {path}")
        if expected_extension and not path.endswith(expected_extension):
            raise ValueError(f"无效的文件格式: {path}，应为{expected_extension}文件")

    @staticmethod
    def normalize_path(path: str) -> str:
        """
        规范化文件路径（处理相对路径和绝对路径）
        Args:
            path: 原始路径
        Returns:
            规范化后的绝对路径
        """
        return os.path.abspath(os.path.expanduser(path))

    @staticmethod
    def get_config_path_from_model_path(model_path: str) -> str:
        """
        从模型文件路径获取配置文件路径（假设在同一目录）
        Args:
            model_path: 模型文件路径
        Returns:
            配置文件路径
        """
        model_dir = os.path.dirname(model_path)
        return os.path.join(model_dir, "config.json")


# find_latest_run_dir方法调用示例
if __name__ == "__main__":

    # 示例5: 组合使用多个参数
    try:
        latest_dir = FileUtils.find_latest_run_dir(
            root_dir="../",
            search_dirs=["../runs", "../results"],
            dir_prefix="train_"
        )
        print(f"示例5 - 组合参数找到最新的目录: {latest_dir}")
    except FileNotFoundError as e:
        print(f"示例5 - 错误: {str(e)}")