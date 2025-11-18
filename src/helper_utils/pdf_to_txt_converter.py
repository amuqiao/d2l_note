#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF转TXT转换器

功能：
    - 将PDF文件转换为纯文本格式
    - 支持单个文件和批量文件夹转换
    - 支持自定义输出目录
    - 包含详细的进度和错误信息

依赖：
    - pdfplumber: 用于提取PDF文本内容
    - argparse: 用于处理命令行参数
"""

import os
import argparse
import logging
from typing import List, Optional
import pdfplumber

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_argparse() -> argparse.ArgumentParser:
    """
    设置命令行参数解析器
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description='PDF转TXT转换器 - 支持单个文件和批量文件夹转换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 转换单个PDF文件
    python pdf_to_txt_converter.py -i input.pdf
    
    # 转换单个PDF文件并指定输出文件名
    python pdf_to_txt_converter.py -i input.pdf -o output.txt
    
    # 批量转换文件夹中的所有PDF文件
    python pdf_to_txt_converter.py -i pdf_folder
    
    # 批量转换文件夹中的所有PDF文件并指定输出目录
    python pdf_to_txt_converter.py -i pdf_folder -o output_folder
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入PDF文件或包含PDF文件的文件夹路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出TXT文件或文件夹路径。如果不指定，将在输入文件同级目录生成'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='递归处理子文件夹中的PDF文件（仅在输入为文件夹时有效）'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细的处理信息'
    )
    
    return parser


def convert_pdf_to_txt(pdf_path: str, txt_path: str) -> bool:
    """
    将单个PDF文件转换为TXT文件
    
    Args:
        pdf_path: 输入PDF文件路径
        txt_path: 输出TXT文件路径
        
    Returns:
        bool: 转换是否成功
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
        
        with pdfplumber.open(pdf_path) as pdf:
            # 提取所有页面的文本
            text_content = []
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"\n=== 第{page_num}页 ===\n")
                    text_content.append(page_text)
            
            # 写入TXT文件
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(''.join(text_content))
        
        logger.info(f"成功转换: {pdf_path} -> {txt_path}")
        return True
        
    except Exception as e:
        logger.error(f"转换失败: {pdf_path}。错误信息: {str(e)}")
        return False


def find_pdf_files(directory: str, recursive: bool = False) -> List[str]:
    """
    查找目录中的所有PDF文件
    
    Args:
        directory: 要搜索的目录路径
        recursive: 是否递归搜索子目录
        
    Returns:
        List[str]: PDF文件路径列表
    """
    pdf_files = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and file.lower().endswith('.pdf'):
                pdf_files.append(file_path)
    
    return pdf_files


def process_single_file(input_path: str, output_path: Optional[str]) -> int:
    """
    处理单个PDF文件
    
    Args:
        input_path: 输入PDF文件路径
        output_path: 输出TXT文件路径（可选）
        
    Returns:
        int: 成功转换的文件数
    """
    # 如果未指定输出路径，使用输入文件名生成
    if not output_path:
        output_path = os.path.splitext(input_path)[0] + '.txt'
    
    # 执行转换
    success = convert_pdf_to_txt(input_path, output_path)
    return 1 if success else 0


def process_directory(input_dir: str, output_dir: Optional[str], recursive: bool = False) -> int:
    """
    处理目录中的所有PDF文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径（可选）
        recursive: 是否递归处理子目录
        
    Returns:
        int: 成功转换的文件数
    """
    # 查找所有PDF文件
    pdf_files = find_pdf_files(input_dir, recursive)
    logger.info(f"找到 {len(pdf_files)} 个PDF文件需要处理")
    
    success_count = 0
    
    for pdf_path in pdf_files:
        # 确定输出文件路径
        if output_dir:
            # 保持相对路径结构
            rel_path = os.path.relpath(pdf_path, input_dir)
            txt_filename = os.path.splitext(rel_path)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)
        else:
            # 如果未指定输出目录，在原文件同级目录生成
            txt_path = os.path.splitext(pdf_path)[0] + '.txt'
        
        # 执行转换
        if convert_pdf_to_txt(pdf_path, txt_path):
            success_count += 1
    
    return success_count


def main() -> None:
    """
    主函数
    """
    # 解析命令行参数
    parser = setup_argparse()
    args = parser.parse_args()
    
    # 根据verbose参数调整日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    input_path = os.path.abspath(args.input)
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        logger.error(f"错误：输入路径 '{input_path}' 不存在")
        return
    
    total_count = 0
    success_count = 0
    
    # 判断是文件还是目录
    if os.path.isfile(input_path):
        # 检查是否为PDF文件
        if not input_path.lower().endswith('.pdf'):
            logger.error(f"错误：输入文件 '{input_path}' 不是PDF文件")
            return
        
        logger.info(f"开始处理单个PDF文件: {input_path}")
        total_count = 1
        success_count = process_single_file(input_path, args.output)
    
    elif os.path.isdir(input_path):
        logger.info(f"开始处理目录: {input_path}")
        logger.info(f"递归模式: {'开启' if args.recursive else '关闭'}")
        
        # 如果指定了输出路径，确保它是一个目录
        if args.output and os.path.exists(args.output) and not os.path.isdir(args.output):
            logger.error(f"错误：指定的输出路径 '{args.output}' 不是一个目录")
            return
        
        success_count = process_directory(input_path, args.output, args.recursive)
        total_count = len(find_pdf_files(input_path, args.recursive))
    
    # 输出总结信息
    logger.info(f"转换完成！成功: {success_count}/{total_count} 个文件")
    
    if success_count < total_count:
        logger.warning(f"有 {total_count - success_count} 个文件转换失败，请查看日志了解详情")


if __name__ == "__main__":
    main()