import os
import logging


# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 获取当前脚本的文件名（不含扩展名）
script_name = os.path.splitext(os.path.basename(__file__))[0]
# 日志文件名 = 脚本名 + .log
log_file_name = f"{script_name}.log"

# 设置日志配置 - 同时输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,
    # format='%(message)s',  # 简化格式，主要显示消息内容
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # 日志文件路径（当前目录下的log_file_name）
        logging.FileHandler(os.path.join(script_dir, log_file_name), mode='a'),  # w表示写入文件,a表示追加
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