import os
import glob
from typing import List, Optional


class PathScanner:
    """路径扫描模块：负责查找训练目录和各类文件"""
    @staticmethod
    def find_run_directories(
        pattern: str = "run_*",  # 文件夹命名模式（支持通配符）
        root_dir: str = ".",     # 查找的根目录
        recursive: bool = False  # 是否递归查找所有子目录
    ) -> List[str]:
        """
        根据模式查找所有层级的模型文件夹
        
        参数:
            pattern: 文件夹命名模式（支持 glob 通配符，如 "run_*", "model_?", "exp_[0-9]*"）
            root_dir: 查找的起始根目录
            recursive: 是否递归查找所有子目录（True 时会遍历 root_dir 下所有层级）
        
        返回:
            所有匹配的文件夹绝对路径列表
        """
        # 构造完整的查找模式：根目录 + 递归通配符（如果需要） + 目标文件夹模式
        if recursive:
            # 递归模式：** 表示任意层级目录
            search_pattern = os.path.join(root_dir, "**", pattern)
        else:
            # 非递归模式：只匹配 root_dir 下的直接子目录
            search_pattern = os.path.join(root_dir, pattern)
        
        # 使用 glob 查找所有匹配项（递归需设置 recursive=True）
        all_matches = glob.glob(search_pattern, recursive=recursive)
        
        # 过滤出真实存在的目录（排除文件）
        matched_dirs = [
            os.path.abspath(match) 
            for match in all_matches 
            if os.path.isdir(match)
        ]
        
        return matched_dirs
    
    @staticmethod
    def get_latest_run_directory(
        pattern: str = "run_*", 
        root_dir: str = ".",
        recursive: bool = False  # 新增：与find_run_directories保持一致的递归参数
    ) -> Optional[str]:
        """
        获取最新修改的训练目录
        
        参数:
            pattern: 文件夹命名模式（支持 glob 通配符）
            root_dir: 查找的起始根目录
            recursive: 是否递归查找所有子目录
        
        返回:
            最新修改的文件夹绝对路径，无匹配时返回None
        """
        # 传递recursive参数，与find_run_directories保持一致
        run_dirs = PathScanner.find_run_directories(pattern, root_dir, recursive)
        if not run_dirs:
            return None
        # 按修改时间排序，返回最新的目录
        return max(run_dirs, key=lambda x: os.path.getmtime(x))

    @staticmethod
    def find_model_files(
        pattern: str = "*.pth",  # 模式参数前置，与其他方法保持一致
        root_dir: str = ".",     # 统一参数名为root_dir
        recursive: bool = False  # 新增：支持递归查找
    ) -> List[str]:
        """
        在指定目录中查找模型文件
        
        参数:
            pattern: 模型文件命名模式（支持 glob 通配符，如 "*.pth", "model_*.ckpt"）
            root_dir: 查找的起始根目录
            recursive: 是否递归查找所有子目录
        
        返回:
            所有匹配的模型文件绝对路径列表
        """
        if not os.path.exists(root_dir):
            return []
            
        # 与find_run_directories保持一致的路径拼接逻辑
        if recursive:
            search_pattern = os.path.join(root_dir, "**", pattern)
        else:
            search_pattern = os.path.join(root_dir, pattern)
        
        # 查找所有匹配项
        all_matches = glob.glob(search_pattern, recursive=recursive)
        
        # 过滤出真实存在的文件（排除目录）
        matched_files = [
            os.path.abspath(match) 
            for match in all_matches 
            if os.path.isfile(match)
        ]
        
        return matched_files

    @staticmethod
    def find_metric_files(
        pattern: str = "*.json",  # 模式参数前置，与其他方法保持一致
        root_dir: str = ".",      # 统一参数名为root_dir
        recursive: bool = False   # 新增：支持递归查找
    ) -> List[str]:
        """
        在指定目录中查找指标文件
        
        参数:
            pattern: 指标文件命名模式（支持 glob 通配符，如 "*.json", "metrics_*.log"）
            root_dir: 查找的起始根目录
            recursive: 是否递归查找所有子目录
        
        返回:
            所有匹配的指标文件绝对路径列表
        """
        if not os.path.exists(root_dir):
            return []
            
        # 与find_run_directories保持一致的路径拼接逻辑
        if recursive:
            search_pattern = os.path.join(root_dir, "**", pattern)
        else:
            search_pattern = os.path.join(root_dir, pattern)
        
        # 查找所有匹配项
        all_matches = glob.glob(search_pattern, recursive=recursive)
        
        # 过滤出真实存在的文件（排除目录）
        matched_files = [
            os.path.abspath(match) 
            for match in all_matches 
            if os.path.isfile(match)
        ]
        
        return matched_files



if __name__ == "__main__":
    # 测试使用的基础路径
    test_root = "/data/home/project/d2l_note/runs"
    print(f"测试基础路径: {test_root}\n")
    
    # 检查路径是否存在
    if not os.path.exists(test_root):
        print(f"警告: 路径 {test_root} 不存在，请确认路径正确性")
    else:
        # 1. 测试find_run_directories方法
        print("1. 测试查找运行目录:")
        run_dirs = PathScanner.find_run_directories(root_dir=test_root)
        print(f"  找到 {len(run_dirs)} 个直接子目录 (run_*):")
        for dir_path in run_dirs[:3]:  # 只显示前3个
            print(f"    - {dir_path}")
        if len(run_dirs) > 3:
            print(f"    ... 还有 {len(run_dirs)-3} 个目录未显示")
        
        # 2. 测试递归查找功能
        print("\n2. 测试递归查找:")
        all_run_dirs = PathScanner.find_run_directories(root_dir=test_root, recursive=True)
        print(f"  递归找到 {len(all_run_dirs)} 个目录 (run_*):")
        
        # 3. 测试最新运行目录
        print("\n3. 测试最新运行目录:")
        latest_run = PathScanner.get_latest_run_directory(root_dir=test_root, recursive=True)
        if latest_run:
            print(f"  最新修改的目录: {latest_run}")
            print(f"  修改时间: {os.path.getmtime(latest_run)}")
        else:
            print("  未找到任何运行目录")
        
        # 4. 测试模型文件查找
        print("\n4. 测试模型文件查找:")
        model_files = PathScanner.find_model_files(root_dir=test_root, recursive=True)
        print(f"  找到 {len(model_files)} 个模型文件 (*.pth):")
        for file_path in model_files[:3]:  # 只显示前3个
            print(f"    - {file_path}")
        
        # 5. 测试指标文件查找
        print("\n5. 测试指标文件查找:")
        metric_files = PathScanner.find_metric_files(root_dir=test_root, recursive=True)
        print(f"  找到 {len(metric_files)} 个指标文件 (*.json):")
        for file_path in metric_files[:3]:  # 只显示前3个
            print(f"    - {file_path}")
            
        print("\n测试完成")