import logging

# 极简配置：控制台输出+精确时间
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S.%f",  # 优化时间格式
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# 使用示例
logger.info("极简日志配置，仅控制台输出")
