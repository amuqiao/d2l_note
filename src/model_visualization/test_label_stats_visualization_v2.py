#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标签统计可视化测试脚本

用于测试LabelStatsVisualizer能否正确处理train/test/val三种label_stats.json文件

用法示例:
  # 基本用法：在当前目录查找标签统计文件
  python test_label_stats_visualization.py

  # 指定目录并递归查找
  python test_label_stats_visualization.py --dir ./data --recursive

  # 详细输出模式
  python test_label_stats_visualization.py -v

  # 查看帮助信息
  python test_label_stats_visualization.py -h
"""
import os
import sys
import argparse
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model_visualization.metric_parsers import MetricParserRegistry
from src.model_visualization.visualizers import VisualizerRegistry
from src.model_visualization.path_scanner import PathScanner


def test_label_stats_visualization(root_dir='.', pattern="*_label_stats.json", 
                                  recursive=True, verbose=False):
    """
    📊 测试标签统计可视化功能
    
    参数:
        root_dir: 查找标签统计文件的根目录
        pattern: 标签统计文件的匹配模式
        recursive: 是否递归查找所有子目录
        verbose: 是否输出详细信息
    """
    # 打印配置信息
    if verbose:
        print("📋 配置信息:")
        print(f"  查找目录: {root_dir}")
        print(f"  文件模式: {pattern}")
        print(f"  递归查找: {'是' if recursive else '否'}")
        print(f"  详细输出: {'是' if verbose else '否'}")
    else:
        print(f"🚀 开始查找标签统计文件...")
    
    # 查找所有符合条件的标签统计文件
    label_stats_files = PathScanner.find_metric_files(
        pattern=pattern,
        root_dir=root_dir,
        recursive=recursive
    )
    
    # 检查是否找到文件
    if not label_stats_files:
        print("❌ 未找到标签统计文件")
        return False
    
    print(f"✅ 找到 {len(label_stats_files)} 个标签统计文件")
    if verbose:
        print("📝 找到的文件列表:")
        for file_path in label_stats_files:
            print(f"  - {file_path}")
    
    # 按目录分组文件
    files_by_directory = {}
    for file_path in label_stats_files:
        directory = os.path.dirname(file_path)
        if directory not in files_by_directory:
            files_by_directory[directory] = []
        files_by_directory[directory].append(file_path)
    
    # 处理每个目录下的文件
    success_count = 0
    total_files = len(label_stats_files)
    
    for directory, files_in_dir in files_by_directory.items():
        print(f"\n📂 处理目录: {directory}")
        print(f"  包含 {len(files_in_dir)} 个标签统计文件")
        
        dir_success_count = 0
        
        for file_path in files_in_dir:
            filename = os.path.basename(file_path)
            if verbose:
                print(f"\n🔍 正在处理文件: {filename}")
            else:
                print(f"🔍 处理文件: {filename}\n", end=' ')
            
            try:
                # 解析文件
                metric_data = MetricParserRegistry.parse_file(file_path)
                if not metric_data:
                    print(f"❌ 解析文件失败: {filename}")
                    continue
                
                if verbose:
                    print(f"  ✅ 成功解析文件，指标类型: {metric_data.metric_type}")
                    print(f"  🎨 开始可视化...")
                
                # 可视化数据
                result = VisualizerRegistry.draw(metric_data)
                if result:
                    print(f"\n✅" if not verbose else "  ✅ 可视化成功")
                    success_count += 1
                    dir_success_count += 1
                else:
                    print(f"\n❌" if not verbose else "  ❌ 可视化失败")
                    
            except Exception as e:
                error_msg = f"处理文件时出错: {str(e)}"
                print(f"❌ {error_msg}" if not verbose else f"  ❌ {error_msg}")
                if verbose:
                    traceback.print_exc()
        
        # 目录处理结果
        print(f"  目录处理完成: 成功 {dir_success_count}/{len(files_in_dir)}")
    
    # 总体结果
    print(f"\n📊 测试完成: 共处理 {total_files} 个文件，成功 {success_count} 个")
    return success_count > 0


def parse_arguments():
    """
    🧩 解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="标签统计可视化测试工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""使用示例:
  # 基本用法：在当前目录查找标签统计文件
  python test_label_stats_visualization.py

  # 指定目录并递归查找
  python test_label_stats_visualization.py --dir ./data --recursive

  # 详细输出模式
  python test_label_stats_visualization.py -v

  # 非递归查找
  python test_label_stats_visualization.py --dir ./data --no-recursive
        """
    )
    
    parser.add_argument(
        "--dir", 
        type=str,
        default='.',
        help="查找标签统计文件的根目录 (默认: 当前目录)"
    )
    
    parser.add_argument(
        "--pattern", 
        type=str,
        default="*_label_stats.json",
        help="标签统计文件的匹配模式 (默认: *_label_stats.json)"
    )
    
    parser.add_argument(
        "--recursive", 
        action="store_true",
        default=True,
        help="递归查找所有子目录 (默认: 启用)"
    )
    
    parser.add_argument(
        "--no-recursive", 
        action="store_false",
        dest="recursive",
        help="禁用递归查找"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="启用详细输出模式"
    )
    
    return parser.parse_args()


def main():
    """
    🚀 程序主入口
    """
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 运行测试
        success = test_label_stats_visualization(
            root_dir=args.dir,
            pattern=args.pattern,
            recursive=args.recursive,
            verbose=args.verbose
        )
        
        # 根据测试结果设置退出码
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"💥 测试过程中出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
