#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pdfplumber
import logging
import warnings
from typing import List

# 直接抑制warnings模块发出的警告
warnings.filterwarnings("ignore", category=Warning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

# 获取所有可能的日志记录器并设置为CRITICAL级别
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.CRITICAL)

# 设置根日志记录器级别并添加过滤器
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 创建一个过滤器来过滤掉所有不相关的警告
class CustomFilter(logging.Filter):
    def filter(self, record):
        # 只允许我们自己的INFO级别消息通过
        if record.levelno == logging.INFO and '已转换' in record.getMessage() or '转换完成' in record.getMessage() or '找到' in record.getMessage():
            return True
        # 过滤掉所有其他消息，特别是FontBBox相关的警告
        if 'FontBBox' in record.getMessage():
            return False
        # 允许其他级别较高的错误消息
        return record.levelno >= logging.ERROR

# 应用过滤器到所有处理器
for handler in root_logger.handlers:
    handler.addFilter(CustomFilter())

# 创建我们自己的logger，使用我们的命名空间
def get_custom_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

logger = get_custom_logger(__name__)


def validate_input_directory(input_dir: str) -> bool:
    """验证输入目录是否存在且为有效目录"""
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return False
    if not os.path.isdir(input_dir):
        logger.error(f"输入路径不是一个目录: {input_dir}")
        return False
    return True


def create_output_directory(output_dir: str) -> bool:
    """创建输出目录（如果不存在）"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"已创建输出目录: {output_dir}")
        return True
    except OSError as e:
        logger.error(f"创建输出目录失败: {e}")
        return False


def get_pdf_files(input_dir: str) -> List[str]:
    """获取输入目录中所有PDF文件的路径"""
    pdf_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                pdf_files.append(file_path)
    
    if not pdf_files:
        logger.warning(f"在输入目录中未找到任何PDF文件: {input_dir}")
    else:
        logger.info(f"找到 {len(pdf_files)} 个PDF文件待转换")
    
    return pdf_files


def pdf_to_txt(pdf_path: str, output_dir: str) -> bool:
    """将单个PDF文件转换为TXT并保存到输出目录"""
    try:
        # 提取文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        txt_filename = f"{filename}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        # 读取PDF并提取文本
        with pdfplumber.open(pdf_path) as pdf:
            text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            
            # 合并所有页面文本并保存
            if text:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(text))
                logger.info(f"已转换: {pdf_path} -> {txt_path}")
                return True
            else:
                logger.warning(f"PDF文件中未提取到文本: {pdf_path}")
                return False
                
    except Exception as e:
        logger.error(f"转换PDF时出错 {pdf_path}: {str(e)}")
        return False


def batch_convert_pdfs(input_dir: str, output_dir: str) -> None:
    """批量转换输入目录中的所有PDF文件"""
    # 验证输入目录
    if not validate_input_directory(input_dir):
        return
    
    # 创建输出目录
    if not create_output_directory(output_dir):
        return
    
    # 获取所有PDF文件
    pdf_files = get_pdf_files(input_dir)
    if not pdf_files:
        return
    
    # 批量转换
    success_count = 0
    for pdf_file in pdf_files:
        if pdf_to_txt(pdf_file, output_dir):
            success_count += 1
    
    logger.info(f"转换完成 - 成功: {success_count}/{len(pdf_files)}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='将指定目录下的PDF文件批量转换为TXT格式',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input_dir', 
        required=True,
        help='包含PDF文件的输入目录'
    )
    
    parser.add_argument(
        '-o', '--output_dir', 
        required=True,
        help='保存转换后TXT文件的输出目录'
    )
    
    # 添加使用示例
    parser.epilog = """使用示例:
  %(prog)s -i ./pdf_files -o ./txt_output
  %(prog)s --input_dir "C:/Documents/PDFs" --output_dir "C:/Documents/Texts"
    """
    
    args = parser.parse_args()
    
    # 执行批量转换
    batch_convert_pdfs(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()