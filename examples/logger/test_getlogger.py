"""测试通过logging.getLogger('d2l_note_new')获取日志实例"""
import logging
from logger import Logger, init


def test_getlogger_directly():
    """测试直接通过logging.getLogger('d2l_note_new')获取日志实例"""
    print("=== 测试通过logging.getLogger('d2l_note_new')获取日志实例 ===")
    
    # 场景1: 先初始化Logger，然后尝试通过logging.getLogger获取
    print("\n场景1: 先初始化Logger，然后通过logging.getLogger获取")
    # 初始化我们的Logger
    logger_instance = Logger()
    logger_instance.info("通过Logger类初始化的日志")
    
    # 尝试通过logging.getLogger获取
    direct_logger = logging.getLogger('d2l_note_new')
    print(f"通过logging.getLogger获取的实例id: {id(direct_logger)}")
    print(f"我们的Logger类中的logger属性id: {id(logger_instance.logger)}")
    print(f"是否是同一个实例: {direct_logger is logger_instance.logger}")
    
    # 使用直接获取的logger记录日志
    direct_logger.info("通过logging.getLogger直接记录的日志")
    
    # 场景2: 重置后，先通过init函数初始化，然后尝试通过logging.getLogger获取
    print("\n场景2: 先通过init函数初始化，然后通过logging.getLogger获取")
    Logger.reset()
    custom_logger = init(log_dir="test_logs", log_file="test_getlogger.log")
    custom_logger.info("通过init函数初始化的日志")
    
    # 尝试通过logging.getLogger获取
    direct_logger2 = logging.getLogger('d2l_note_new')
    print(f"通过logging.getLogger获取的实例id: {id(direct_logger2)}")
    print(f"init函数返回的实例中的logger属性id: {id(custom_logger.logger)}")
    print(f"是否是同一个实例: {direct_logger2 is custom_logger.logger}")
    
    # 使用直接获取的logger记录日志
    direct_logger2.info("通过logging.getLogger直接记录的日志(场景2)")
    
    # 场景3: 不初始化我们的Logger，直接尝试通过logging.getLogger获取
    print("\n场景3: 不初始化我们的Logger，直接通过logging.getLogger获取")
    Logger.reset()
    
    # 直接通过logging.getLogger获取
    direct_logger3 = logging.getLogger('d2l_note_new')
    print(f"直接通过logging.getLogger获取的实例: {direct_logger3}")
    print(f"日志级别: {direct_logger3.level}")
    print(f"处理器数量: {len(direct_logger3.handlers)}")
    
    # 尝试使用这个logger记录日志
    try:
        direct_logger3.info("测试不初始化直接使用")
        print("可以记录日志，但可能不会输出到任何地方，因为没有添加处理器")
    except Exception as e:
        print(f"记录日志时出错: {e}")


if __name__ == "__main__":
    test_getlogger_directly()