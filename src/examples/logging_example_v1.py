import logging
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 日志文件路径（当前目录下的app.log）
log_file = os.path.join(script_dir, 'app.log')

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file,  # 指定日志文件路径
    filemode='a'  # 追加模式，默认也是'a'
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
