#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timestride模型信息解析与可视化工具
专门用于解析和展示Timestride项目中的模型信息、训练指标和参数配置
"""
import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional, Union

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入必要的模块
from src.model_visualization.model_info_parsers import ModelInfoParserRegistry, ModelInfoData
from src.model_visualization.model_info_visualizers import ModelInfoVisualizerRegistry, TimestrideModelComparisonVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("timestride_model_info")


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Timestride模型信息解析与可视化工具")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/home/project/d2l_note/timestride/cp_0924_count/pv_pos_1/classification_string_current_1_32_pv_pos_1_TimesNet_TSLBHDF5_ftM_sl96_ll48_pl0_dm32_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0",
        help="模型目录路径，默认指向示例模型"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="timestride",
        help="解析器和可视化器的命名空间，默认为'timestride'"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径，默认为标准输出"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="是否启用模型比较功能"
    )
    return parser.parse_args()


def find_model_files(model_dir: str) -> List[str]:
    """在指定目录中查找模型相关文件"""
    model_files = []
    
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        logger.error(f"模型目录不存在: {model_dir}")
        return []
    
    # 查找.pth文件
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pth"):
                model_files.append(os.path.join(root, file))
        
        # 查找training_metrics.json
        metrics_path = os.path.join(root, "training_metrics.json")
        if os.path.exists(metrics_path):
            model_files.append(metrics_path)
        
        # 查找args.json
        args_path = os.path.join(root, "args.json")
        if os.path.exists(args_path):
            model_files.append(args_path)
    
    # 去重并保持顺序
    unique_files = []
    seen = set()
    for file in model_files:
        if file not in seen:
            seen.add(file)
            unique_files.append(file)
    
    return unique_files


def parse_model_info(files: List[str], namespace: str = "timestride") -> List[ModelInfoData]:
    """解析模型信息文件"""
    results = []
    
    for file_path in files:
        logger.info(f"正在解析文件: {file_path}")
        
        try:
            # 使用Registry的parse_file方法解析文件
            data = ModelInfoParserRegistry.parse_file(file_path, namespace=namespace)
            if data:
                results.append(data)
                logger.info(f"成功解析文件 '{file_path}'，类型: {data.type}")
            else:
                logger.warning(f"无法解析文件: {file_path}")
        except Exception as e:
            logger.error(f"解析文件 '{file_path}' 失败: {str(e)}")
    
    return results


def visualize_model_info(data_list: List[ModelInfoData], output_file: Optional[str] = None, compare: bool = False) -> str:
    """可视化模型信息"""
    results = []
    
    # 打印程序标题
    results.append("=" * 80)
    results.append("📈 Timestride模型信息解析与可视化")
    results.append("=" * 80)
    
    # 存储解析的信息，用于后续处理
    parsed_infos = {}
    
    # 分离模型文件和其他类型的文件
    model_files = []  # 存储.pth文件信息
    other_files = []  # 存储其他类型的文件信息
    
    # 如果启用了比较功能且有多个模型信息
    if compare and len(data_list) >= 2:
        try:
            results.append("\n" + "=" * 80)
            results.append("🔄 Timestride模型比较")
            results.append("=" * 80)
            
            # 使用特殊的比较可视化器
            comparison_visualizer = TimestrideModelComparisonVisualizer()
            for model_info in data_list:
                comparison_visualizer.add_model_info(model_info)
                parsed_infos[model_info.path] = model_info
                if model_info.type == "model" and model_info.path.endswith(".pth"):
                    model_files.append(model_info)
                else:
                    other_files.append(model_info)
            comparison_visualizer.visualize()
        except Exception as e:
            results.append(f"❌ 执行模型比较失败: {str(e)}")
    else:
        # 解析并可视化每个模型信息
        for i, data in enumerate(data_list, 1):
            results.append(f"\n{i}. 🔍 解析结果 - {data.type}:")
            results.append(f"   路径: {data.path}")
            results.append(f"   类型: {data.model_type}")
            
            # 使用Registry的draw方法进行可视化
            try:
                # 注意：draw方法会直接输出到控制台，我们这里不捕获它的返回值
                ModelInfoVisualizerRegistry.draw(data)
                parsed_infos[data.path] = data
                # 分类存储
                if data.type == "model" and data.path.endswith(".pth"):
                    model_files.append(data)
                else:
                    other_files.append(data)
            except Exception as e:
                results.append(f"   ❌ 可视化失败: {str(e)}")
    
    # 添加总结信息
    results.append("\n" + "=" * 80)
    results.append(f"✅ 解析完成! 共解析 {len(data_list)} 个文件")
    
    # 添加详细分类统计
    if model_files:
        results.append(f"   - 模型文件(.pth): {len(model_files)} 个")
        # 统计包含.pth文件的唯一目录
        model_dirs = set(os.path.dirname(model.path) for model in model_files)
        results.append(f"   - 包含模型的目录: {len(model_dirs)} 个")
    if other_files:
        results.append(f"   - 其他文件: {len(other_files)} 个")
        
    results.append("=" * 80)
    
    # 合并所有结果
    output = "\n".join(results)
    
    # 如果指定了输出文件，保存到文件
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            logger.info(f"输出已保存到文件: {output_file}")
        except Exception as e:
            logger.error(f"保存输出到文件失败: {str(e)}")
    
    return output


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"开始解析Timestride模型信息，目录: {args.model_dir}")
    
    # 查找模型文件
    model_files = find_model_files(args.model_dir)
    
    if not model_files:
        logger.error(f"在目录 '{args.model_dir}' 中没有找到模型相关文件")
        sys.exit(1)
    
    logger.info(f"找到 {len(model_files)} 个模型相关文件")
    if args.verbose:
        for file in model_files:
            logger.debug(f"找到文件: {file}")
    
    # 解析模型信息
    model_data_list = parse_model_info(model_files, args.namespace)
    
    if not model_data_list:
        logger.error("解析模型信息失败，没有获取到有效数据")
        sys.exit(1)
    
    logger.info(f"成功解析 {len(model_data_list)} 个模型数据")
    
    # 可视化模型信息
    output = visualize_model_info(model_data_list, args.output, args.compare)
    
    # 如果没有指定输出文件，则打印到控制台
    if not args.output:
        print(output)
    
    logger.info("Timestride模型信息解析与可视化完成")


if __name__ == "__main__":
    main()