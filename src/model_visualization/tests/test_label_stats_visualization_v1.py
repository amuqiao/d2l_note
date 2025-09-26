#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标签统计可视化测试脚本
用于测试LabelStatsVisualizer能否正确处理train/test/val三种label_stats.json文件
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model_visualization.metric_parsers import MetricParserRegistry
from src.model_visualization.visualizers import VisualizerRegistry
from src.model_visualization.path_scanner import PathScanner


def test_label_stats_visualization(root_dir='.', pattern="*_label_stats.json", recursive=True):
    """
    测试标签统计可视化功能
    
    参数:
        root_dir: 查找标签统计文件的根目录
        pattern: 标签统计文件的匹配模式
        recursive: 是否递归查找所有子目录
    """
    # 默认路径设置
    search_dir = root_dir
    
    print(f"开始查找标签统计文件...")
    print(f"查找目录: {search_dir}")
    print(f"文件模式: {pattern}")
    print(f"递归查找: {'是' if recursive else '否'}")
    
    # 使用PathScanner.find_metric_files获取所有标签统计文件（两个方法功能相同）
    label_stats_files = PathScanner.find_metric_files(
        pattern=pattern,
        root_dir=search_dir,
        recursive=recursive
    )
    
    if not label_stats_files:
        print(f"未找到标签统计文件")
        return False
    
    print(f"找到 {len(label_stats_files)} 个标签统计文件")
    
    # 按目录分组文件
    files_by_directory = {}
    for file_path in label_stats_files:
        directory = os.path.dirname(file_path)
        if directory not in files_by_directory:
            files_by_directory[directory] = []
        files_by_directory[directory].append(file_path)
    
    # 依次处理每个目录下的文件
    success_count = 0
    total_files = len(label_stats_files)
    
    # 按目录处理文件
    for directory, files_in_dir in files_by_directory.items():
        print(f"\n\n========== 处理目录: {directory} ==========")
        print(f"目录中包含 {len(files_in_dir)} 个标签统计文件")
        
        dir_success_count = 0
        
        for file_path in files_in_dir:
            print(f"\n--- 处理文件: {os.path.basename(file_path)} ---")
            
            try:
                # 使用解析器解析文件
                metric_data = MetricParserRegistry.parse_file(file_path)
                if not metric_data:
                    print(f"解析文件失败: {file_path}")
                    continue
                
                print(f"成功解析文件，指标类型: {metric_data.metric_type}")
                
                # 使用可视化器可视化数据
                result = VisualizerRegistry.draw(metric_data)
                if result:
                    print(f"可视化成功")
                    success_count += 1
                    dir_success_count += 1
                else:
                    print(f"可视化失败")
            except Exception as e:
                print(f"处理文件时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 打印目录处理结果
        print(f"\n目录 {directory} 处理完成: 共 {len(files_in_dir)} 个文件，成功 {dir_success_count} 个")
    
    print(f"\n\n测试完成: 共处理 {total_files} 个文件，成功 {success_count} 个")
    return success_count > 0


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="标签统计可视化测试工具")
    parser.add_argument(
        "--dir", 
        type=str,
        default='.',
        help="查找标签统计文件的根目录"
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
        help="是否递归查找所有子目录 (默认: 是)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 运行测试
        success = test_label_stats_visualization(
            root_dir=args.dir,
            pattern=args.pattern,
            recursive=args.recursive
        )
        
        # 根据测试结果设置退出码
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)