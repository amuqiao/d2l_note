import os
import logging

# 设置日志配置 - 输出到控制台
logging.basicConfig(
    level=logging.INFO,
    # format='%(message)s',  # 简化格式，主要显示消息内容
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)

# 创建日志记录器
logger = logging.getLogger(__name__)

# 示例日志输出
if __name__ == "__main__":
    logger.debug("这是一条调试信息（不会被输出，因为级别是INFO）")
    logger.info("程序启动成功")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    
    try:
        1 / 0
    except Exception as e:
        logger.exception("发生了异常")