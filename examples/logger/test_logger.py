"""测试新的日志模块功能"""
from logger import Logger, logger, init, debug, info, success, warning, error, critical, exception, log_dict, is_initialized, get_logger, reset_logger


def test_basic_logging():
    """测试基本的日志记录功能"""
    print("=== 测试基本日志记录功能 ===")
    debug("这是一条调试日志")
    info("这是一条信息日志")
    success("这是一条成功日志")
    warning("这是一条警告日志")
    error("这是一条错误日志")
    critical("这是一条严重错误日志")
    
    # 测试异常日志
    try:
        1 / 0
    except Exception:
        exception("捕获到异常")
    
    # 测试字典日志
    data = {"name": "logger", "version": "1.0", "type": "singleton"}
    log_dict("测试字典日志", data)


def test_singleton():
    """测试单例模式"""
    print("\n=== 测试单例模式 ===")
    logger1 = Logger.get_instance()
    logger2 = Logger()
    logger3 = get_logger()
    
    print(f"logger1 id: {id(logger1)}")
    print(f"logger2 id: {id(logger2)}")
    print(f"logger3 id: {id(logger3)}")
    print(f"所有实例是否相同: {logger1 is logger2 is logger3}")


def test_initialization_check():
    """测试初始化检查功能"""
    print("\n=== 测试初始化检查 ===")
    print(f"日志系统是否已初始化: {is_initialized()}")


def test_custom_initialization():
    """测试自定义初始化"""
    print("\n=== 测试自定义初始化 ===")
    custom_logger = init(
        log_dir="custom_logs",
        log_file="test_logger.log",
        log_level="DEBUG",
        console_level="INFO",
        use_timestamp=False
    )
    
    print(f"自定义日志目录: {custom_logger.get_log_dir()}")
    print(f"自定义日志文件路径: {custom_logger.get_log_file_path()}")
    
    # 测试日志级别设置
    custom_logger.set_level("DEBUG")
    custom_logger.set_console_level("WARNING")
    custom_logger.set_file_level("DEBUG")
    
    # 这些日志只会在文件中显示，不会在控制台显示
    custom_logger.debug("这条调试日志只会写入文件")
    custom_logger.info("这条信息日志只会写入文件")
    
    # 这些日志会同时在控制台和文件中显示
    custom_logger.warning("这条警告日志会同时显示在控制台和文件中")
    custom_logger.error("这条错误日志会同时显示在控制台和文件中")


def test_reset():
    """测试重置功能"""
    print("\n=== 测试重置功能 ===")
    print(f"重置前是否已初始化: {is_initialized()}")
    reset_logger()
    print(f"重置后是否已初始化: {is_initialized()}")
    
    # 重置后重新初始化
    new_logger = Logger()
    print(f"重新初始化后是否已初始化: {is_initialized()}")


def main():
    """运行所有测试"""
    test_basic_logging()
    test_singleton()
    test_initialization_check()
    test_custom_initialization()
    test_reset()


if __name__ == "__main__":
    main()