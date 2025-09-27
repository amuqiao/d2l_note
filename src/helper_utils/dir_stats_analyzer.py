"""
目录统计分析工具 (dir_stats_analyzer.py)

功能:
- 递归分析指定目录的文件和子目录结构
- 统计文件总数、目录总数和总大小
- 分析文件类型分布及其占比
- 识别最大的文件及其位置
- 按文件数量或大小对目录进行排序
- 生成格式化的统计报告，包括总体摘要、文件类型分布和目录排名

使用示例:
1. 分析当前目录:
   python dir_stats_analyzer.py

2. 分析指定目录:
   python dir_stats_analyzer.py /path/to/directory

3. 详细模式并显示更多条目:
   python dir_stats_analyzer.py /path/to/directory -v --max-entries 20

4. 查看帮助信息:
   python dir_stats_analyzer.py -h
"""

import os
import argparse
from dataclasses import dataclass
from prettytable import PrettyTable
from typing import List, Dict, Tuple, Optional

@dataclass
class FileStats:
    """单个文件的统计信息"""
    name: str
    path: str
    size: int  # 字节数
    extension: str  # 文件扩展名

@dataclass
class DirectoryStats:
    """单个目录的统计信息"""
    path: str
    file_count: int
    dir_count: int
    total_size: int  # 字节数
    file_types: Dict[str, int]  # 键: 扩展名, 值: 数量
    subdirs: List['DirectoryStats']  # 子目录统计
    max_file: Optional[FileStats] = None  # 目录中最大的文件

@dataclass
class OverallStats:
    """总体统计信息"""
    total_files: int
    total_dirs: int
    total_size: int  # 字节数
    file_type_distribution: Dict[str, Tuple[int, float]]  # 键: 扩展名, 值: (数量, 占比)
    dir_stats_list: List[DirectoryStats]  # 所有目录的统计信息列表
    max_file: Optional[FileStats] = None  # 所有文件中最大的文件

# 📁 文件处理工具函数
def get_file_extension(filename: str) -> str:
    """获取文件扩展名，不带点，小写"""
    ext = os.path.splitext(filename)[1].lower()
    return ext[1:] if ext else "no_extension"

def convert_size(size_bytes: int) -> str:
    """将字节数转换为人类可读的格式"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = min(int((size_bytes.bit_length() - 1) // 10), len(size_name) - 1)
    return f"{size_bytes / (1024 **i):.2f} {size_name[i]}"

# 🔍 目录遍历与统计收集
def traverse_directory(path: str, all_dirs: List[DirectoryStats], verbose: bool = False) -> DirectoryStats:
    """
    递归遍历目录并收集统计信息
    
    参数:
        path: 要遍历的目录路径
        all_dirs: 存储所有目录统计信息的列表
        verbose: 是否输出详细信息
    """
    file_count = 0
    dir_count = 0
    total_size = 0
    file_types = {}
    subdirs = []
    max_file = None
    
    # 检查路径是否存在
    if not os.path.exists(path):
        raise ValueError(f"路径不存在: {path}")
    
    # 检查是否是目录
    if not os.path.isdir(path):
        raise ValueError(f"不是目录: {path}")
    
    if verbose:
        print(f"正在分析: {path}")
    
    for entry in os.scandir(path):
        try:
            if entry.is_dir(follow_symlinks=False):
                # 递归处理子目录
                subdir_stats = traverse_directory(entry.path, all_dirs, verbose)
                subdirs.append(subdir_stats)
                dir_count += 1 + subdir_stats.dir_count
                file_count += subdir_stats.file_count
                total_size += subdir_stats.total_size
                
                # 合并文件类型统计
                for ext, count in subdir_stats.file_types.items():
                    if ext in file_types:
                        file_types[ext] += count
                    else:
                        file_types[ext] = count
                
                # 更新最大文件
                if subdir_stats.max_file and (not max_file or subdir_stats.max_file.size > max_file.size):
                    max_file = subdir_stats.max_file
                    
            elif entry.is_file(follow_symlinks=False):
                file_count += 1
                file_size = entry.stat().st_size
                total_size += file_size
                ext = get_file_extension(entry.name)
                
                # 更新文件类型统计
                if ext in file_types:
                    file_types[ext] += 1
                else:
                    file_types[ext] = 1
                
                # 创建文件统计对象
                current_file = FileStats(
                    name=entry.name,
                    path=entry.path,
                    size=file_size,
                    extension=ext
                )
                
                # 更新最大文件
                if not max_file or file_size > max_file.size:
                    max_file = current_file
        except OSError as e:
            print(f"❌ 警告: 无法访问 {entry.path}: {e}")
    
    # 创建当前目录的统计对象
    dir_stats = DirectoryStats(
        path=path,
        file_count=file_count,
        dir_count=dir_count,
        total_size=total_size,
        file_types=file_types,
        subdirs=subdirs,
        max_file=max_file
    )
    
    # 将当前目录统计添加到全局列表
    all_dirs.append(dir_stats)
    
    return dir_stats

# 📊 统计计算
def calculate_overall_stats(root_stats: DirectoryStats, all_dirs: List[DirectoryStats]) -> OverallStats:
    """计算总体统计信息"""
    # 计算文件类型分布及其占比
    file_type_distribution = {}
    total_files = root_stats.file_count
    
    for ext, count in root_stats.file_types.items():
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        file_type_distribution[ext] = (count, percentage)
    
    return OverallStats(
        total_files=root_stats.file_count,
        total_dirs=root_stats.dir_count,
        total_size=root_stats.total_size,
        file_type_distribution=file_type_distribution,
        dir_stats_list=all_dirs,
        max_file=root_stats.max_file
    )

# 🔄 排序功能
def sort_dirs_by_file_count(dir_stats_list: List[DirectoryStats], descending: bool = True) -> List[DirectoryStats]:
    """按文件数量排序目录统计信息"""
    return sorted(dir_stats_list, key=lambda x: x.file_count, reverse=descending)

def sort_dirs_by_size(dir_stats_list: List[DirectoryStats], descending: bool = True) -> List[DirectoryStats]:
    """按总大小排序目录统计信息"""
    return sorted(dir_stats_list, key=lambda x: x.total_size, reverse=descending)

# 📑 报告生成
def generate_summary_table(overall_stats: OverallStats, verbose: bool = False) -> PrettyTable:
    """生成总体统计摘要表格"""
    table = PrettyTable()
    table.field_names = ["统计项", "值"]
    
    table.add_row(["总文件数量", overall_stats.total_files])
    table.add_row(["总目录数量", overall_stats.total_dirs])
    table.add_row(["总大小", convert_size(overall_stats.total_size)])
    
    if overall_stats.max_file:
        table.add_row(["最大文件", overall_stats.max_file.name])
        table.add_row(["最大文件大小", convert_size(overall_stats.max_file.size)])
        table.add_row(["最大文件路径", get_display_path(overall_stats.max_file.path, verbose)])
    else:
        table.add_row(["最大文件", "无"])
    
    return table

def generate_file_type_table(overall_stats: OverallStats) -> PrettyTable:
    """生成文件类型分布表格"""
    table = PrettyTable()
    table.field_names = ["文件类型", "数量", "占比"]
    
    # 按数量排序
    sorted_types = sorted(
        overall_stats.file_type_distribution.items(),
        key=lambda x: x[1][0],
        reverse=True
    )
    
    for ext, (count, percentage) in sorted_types:
        table.add_row([ext, count, f"{percentage:.2f}%"])
    
    return table

def get_display_path(path: str, verbose: bool = False) -> str:
    """根据verbose参数返回完整路径或简短路径"""
    if verbose:
        return path
    # 返回简短路径（仅显示最后两级目录）
    parts = path.split(os.path.sep)
    if len(parts) <= 2:
        return path
    return os.path.sep.join(["..."] + parts[-2:])

def generate_dir_stats_table(dir_stats_list: List[DirectoryStats], max_entries: int = 10, verbose: bool = False) -> PrettyTable:
    """生成目录统计表格"""
    table = PrettyTable()
    table.field_names = ["目录路径", "文件数量", "子目录数量", "总大小", "最大文件"]
    
    # 限制显示的条目数量
    display_stats = dir_stats_list[:max_entries]
    
    for dir_stat in display_stats:
        max_file_name = dir_stat.max_file.name if dir_stat.max_file else "无"
        table.add_row([
            get_display_path(dir_stat.path, verbose),
            dir_stat.file_count,
            dir_stat.dir_count,
            convert_size(dir_stat.total_size),
            max_file_name
        ])
    
    if len(dir_stats_list) > max_entries:
        table.add_row(["...", "...", "...", "...", "..."])
    
    return table

# ⚙️ 参数解析
def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='📊 目录统计分析工具 - 分析指定路径下的文件和目录统计信息',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""使用示例:
  1. 基本使用（分析当前目录）:
     python dir_stats_analyzer.py
     
  2. 分析指定目录:
     python dir_stats_analyzer.py /path/to/directory
     
  3. 显示详细处理过程并增加显示条目数:
     python dir_stats_analyzer.py /path/to/directory -v --max-entries 20
     
  4. 查看帮助信息:
     python dir_stats_analyzer.py -h"""
    )
    
    parser.add_argument(
        'path', 
        nargs='?', 
        default=os.getcwd(), 
        help='要分析的目录路径（默认为当前目录）'
    )
    
    parser.add_argument(
        '--max-entries', 
        type=int, 
        default=10, 
        help='目录表格中显示的最大条目数（默认为10）'
    )
    
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help='显示详细处理过程'
    )
    
    return parser.parse_args()

# 🚀 主函数
def main():
    """主函数：协调目录分析的各个步骤并生成报告"""
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        print(f"🚀 开始分析目录: {args.path}")
        if args.verbose:
            print("📋 详细模式已开启，正在收集统计信息...")
        else:
            print("📋 正在收集统计信息...")
        
        # 收集所有目录的统计信息
        all_dirs = []
        root_stats = traverse_directory(args.path, all_dirs, args.verbose)
        
        # 计算总体统计信息
        overall_stats = calculate_overall_stats(root_stats, all_dirs)
        
        # 按文件数量排序目录
        sorted_by_file_count = sort_dirs_by_file_count(all_dirs)
        
        # 按大小排序目录
        sorted_by_size = sort_dirs_by_size(all_dirs)
        
        # 生成并打印报告
        print("\n" + "="*70)
        print("📊 文件系统统计分析报告")
        print("="*70 + "\n")
        
        print("1. 总体统计摘要:")
        print(generate_summary_table(overall_stats, args.verbose))
        print("\n")
        
        print("2. 文件类型分布:")
        print(generate_file_type_table(overall_stats))
        print("\n")
        
        print(f"3. 按文件数量排序的目录 (前{args.max_entries}个):")
        print(generate_dir_stats_table(sorted_by_file_count, args.max_entries, args.verbose))
        print("\n")
        
        print(f"4. 按大小排序的目录 (前{args.max_entries}个):")
        print(generate_dir_stats_table(sorted_by_size, args.max_entries, args.verbose))
        print("\n")
        
        print("✅ 分析完成。")
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")

if __name__ == "__main__":
    main()
