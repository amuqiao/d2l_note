from ast import main
import sys
import os
import logging
from src.utils.log_utils import LogUtils, logger

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demonstrate_proper_usage():
    """演示LogUtils的正确使用方式"""
    # 1. 基本使用方式 - 直接使用全局logger实例
    print("\n=== 基本使用方式 ===")
    logger.info("这是使用全局logger实例记录的信息")
    
    # 2. 自定义日志配置 - 修改日志文件路径和级别
    print("\n=== 自定义日志配置 ===")
    custom_logger = LogUtils.init_logger(
        log_dir="custom_logs",
        log_file="custom_app.log",
        log_level="DEBUG"
    )
    custom_logger.debug("这是记录到自定义日志文件的调试信息")
    
    # 3. 再次调用init_logger会发生什么?
    print("\n=== 再次调用init_logger ===")
    new_logger = LogUtils.init_logger(
        log_dir="new_logs",
        log_file="new_app.log"
    )
    # 注意：new_logger和之前的logger实际上是同一个实例
    print(f"是否是同一个实例: {logger is new_logger}")
    
    # 4. 验证日志文件路径已更新
    print(f"当前日志文件路径: {logger.get_log_file_path()}")

if __name__ == "__main__":
    demonstrate_proper_usage()
