"""日志模块使用示例"""

# 导入日志工具
from src.utils.log_utils import (
    init_logger, 
    logger, 
    debug, info, success, warning, error, critical, exception, log_dict
)
import time
import os
import traceback


def basic_logging_example():
    """基本日志记录示例"""
    print("\n=== 基本日志记录示例 ===")
    
    # 使用不同级别的日志
    debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    success("这是一条成功信息")
    warning("这是一条警告信息")
    error("这是一条错误信息")
    critical("这是一条严重错误信息")
    
    # 格式化日志消息
    name = "d2l_note"
    version = "1.0"
    info("项目名称: %s, 版本: %s", name, version)
    
    # 使用f-string格式化
    info(f"项目名称: {name}, 版本: {version}")


def file_logging_example():
    """文件日志记录示例"""
    print("\n=== 文件日志记录示例 ===")
    
    # 初始化日志系统，指定日志文件
    log_dir = "logs"
    log_file = f"example_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # 初始化日志系统
    log_manager = init_logger(
        log_dir=log_dir, 
        log_file=log_file, 
        log_level="DEBUG",  # 设置日志级别为DEBUG
        use_timestamp=False  # 不使用时间戳后缀（因为我们已经在文件名中包含了）
    )
    
    # 获取日志文件路径
    log_file_path = log_manager.get_log_file_path()
    info(f"日志文件已创建: {log_file_path}")
    
    # 记录不同级别的日志，这些日志会同时输出到控制台和文件
    debug("这条调试信息将被写入日志文件")
    info("这条信息将被写入日志文件")
    success("这条成功信息将被写入日志文件")
    warning("这条警告信息将被写入日志文件")
    error("这条错误信息将被写入日志文件")
    
    return log_file_path


def exception_logging_example():
    """异常日志记录示例"""
    print("\n=== 异常日志记录示例 ===")
    
    try:
        # 故意引发异常
        result = 10 / 0
    except Exception as e:
        # 使用exception方法记录异常，会自动包含堆栈信息
        exception("计算过程中发生错误")
        
        # 或者使用error方法手动记录异常信息
        error(f"捕获到异常: {str(e)}")
        error(f"堆栈跟踪: {traceback.format_exc()}")


def structured_logging_example():
    """结构化日志记录示例"""
    print("\n=== 结构化日志记录示例 ===")
    
    # 创建一个包含训练参数的字典
    training_params = {
        "model_name": "LeNet",
        "num_epochs": 10,
        "learning_rate": 0.01,
        "batch_size": 128,
        "device": "cuda",
        "optimizer": "SGD"
    }
    
    # 使用log_dict方法结构化记录字典数据
    log_dict("训练参数配置", training_params)
    
    # 创建一个包含模型性能指标的字典
    metrics = {
        "train_loss": 0.2345,
        "train_accuracy": 0.9213,
        "test_loss": 0.3124,
        "test_accuracy": 0.8976,
        "total_time": "123.45s"
    }
    
    # 使用不同日志级别记录字典数据
    log_dict("模型性能指标", metrics, log_level="INFO")
    log_dict("详细性能分析", metrics, log_level="DEBUG")


def integration_example():
    """在实际项目中的集成示例"""
    print("\n=== 实际项目集成示例 ===")
    
    # 初始化日志系统
    log_manager = init_logger(log_level="INFO")
    log_dir = log_manager.get_log_dir()
    
    # 模拟训练过程
    def run_training(model_name, config):
        info(f"开始训练模型: {model_name}")
        log_dict(f"{model_name} 配置信息", config)
        
        try:
            # 模拟训练过程
            for epoch in range(config["num_epochs"]):
                info(f"Epoch {epoch+1}/{config['num_epochs']} 开始")
                
                # 模拟训练中...
                time.sleep(0.5)  # 模拟耗时操作
                
                # 模拟每个epoch的结果
                train_loss = 0.5 * (1 - epoch/config['num_epochs'])
                train_acc = 0.6 + 0.3 * (epoch/config['num_epochs'])
                
                info(f"Epoch {epoch+1} 完成 - 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
                
                # 随机模拟一个错误
                if epoch == 3:
                    raise ValueError("模拟训练过程中的随机错误")
                
            # 训练成功
            success(f"模型 {model_name} 训练完成！")
            return True
        except Exception as e:
            # 记录异常
            exception(f"模型 {model_name} 训练失败")
            return False
    
    # 模拟不同模型的训练
    models_to_train = [
        {
            "name": "LeNet",
            "config": {
                "num_epochs": 5,
                "lr": 0.01,
                "batch_size": 128,
                "input_size": "1x1x28x28"
            }
        },
        {
            "name": "AlexNet",
            "config": {
                "num_epochs": 3,
                "lr": 0.001,
                "batch_size": 64,
                "input_size": "1x3x224x224"
            }
        }
    ]
    
    for model_info in models_to_train:
        success = run_training(model_info["name"], model_info["config"])
        if not success:
            warning(f"跳过后续模型训练")
            break
    
    info(f"所有训练任务已处理完成，日志保存在: {log_dir}")


def initialization_timing_example():
    """日志初始化时机示例"""
    print("\n=== 日志初始化时机示例 ===")
    
    # 当模块被导入时，全局logger实例已经自动创建
    info("1. 日志模块被导入时，logger已经自动初始化")
    info(f"   当前日志文件路径: {logger.get_log_file_path()}")
    
    # 场景1: 在文件开头初始化（推荐）
    # 优点：确保所有日志都被正确记录，特别是导入模块时产生的日志
    info("\n2. 在文件开头初始化（推荐做法）")
    # 示例：如果这是一个新文件，我们会在顶部添加：
    # from src.utils.log_utils import init_logger, info
    # init_logger(log_level="INFO")
    
    # 场景2: 在main函数中初始化
    # 优点：可以根据命令行参数动态配置日志
    info("\n3. 在main函数中初始化")
    # 示例：
    # def main():
    #     args = parse_args()
    #     init_logger(log_level=args.log_level)
    
    # 场景3: 在需要使用日志的函数内部初始化
    # 注意：这种方式可能会导致之前的日志丢失
    info("\n4. 在函数内部初始化（不推荐）")
    # 示例：
    # def my_function():
    #     init_logger()  # 此时之前的日志可能已丢失
    #     info("这条日志会被记录")


def multiple_initialization_example():
    """重复初始化影响示例"""
    print("\n=== 重复初始化影响示例 ===")
    
    # 1. 首次初始化
    info("1. 首次初始化日志系统")
    first_logger = init_logger(log_dir="logs", log_file="first_init.log", use_timestamp=False)
    first_log_path = first_logger.get_log_file_path()
    info(f"   首次初始化的日志文件: {first_log_path}")
    
    # 2. 再次初始化（会发生什么？）
    info("\n2. 重复初始化日志系统")
    second_logger = init_logger(log_dir="logs", log_file="second_init.log", use_timestamp=False)
    second_log_path = second_logger.get_log_file_path()
    info(f"   第二次初始化的日志文件: {second_log_path}")
    
    # 3. 单例模式的验证
    info("\n3. 单例模式验证")
    info(f"   first_logger 和 second_logger 是同一个实例: {first_logger is second_logger}")
    info(f"   first_logger 和全局logger 是同一个实例: {first_logger is logger}")
    
    # 4. 重复初始化的影响
    info("\n4. 重复初始化的潜在问题:")
    info("   - 多个文件handler可能同时存在")
    info("   - 日志会被写入到所有初始化过的日志文件中")
    info("   - 多次输出初始化完成的日志消息")
    
    # 5. 避免重复初始化的方法
    info("\n5. 避免重复初始化的方法:")
    info("   - 在应用程序入口点（main函数）统一初始化一次")
    info("   - 被其他模块导入时，不要在模块级别调用init_logger()")
    info("   - 使用日志前检查是否已经初始化")
    
    # 检查当前logger的handlers数量
    handler_count = len(logger.logger.handlers)
    info(f"   当前logger的handler数量: {handler_count}")
    for i, handler in enumerate(logger.logger.handlers):
        handler_type = type(handler).__name__
        info(f"   Handler {i+1}: {handler_type}")
        if hasattr(handler, 'baseFilename'):
            info(f"     文件路径: {handler.baseFilename}")


def multi_script_integration_example():
    """多脚本调用场景下的日志管理示例"""
    print("\n=== 多脚本调用场景下的日志管理 ===")
    
    info("1. 推荐的集成模式:")
    info("   - 主入口文件负责统一初始化日志系统")
    info("   - 被调用的模块直接使用全局logger实例")
    
    # 模拟主入口文件的初始化
    info("\n2. 模拟主入口文件初始化日志:")
    main_logger = init_logger(log_dir="logs", log_file="main_script.log", use_timestamp=False)
    info(f"   主脚本日志文件: {main_logger.get_log_file_path()}")
    
    # 模拟被调用模块使用日志
    def simulate_module():
        """模拟被其他模块导入的情况"""
        # 直接使用全局logger，不需要再次初始化
        info("   [模块内] 直接使用已初始化的全局logger")
        debug("   [模块内] 记录一些调试信息")
        success("   [模块内] 记录一些成功信息")
    
    info("\n3. 模拟被调用模块使用日志:")
    simulate_module()
    
    info("\n4. 多脚本环境中的最佳实践:")
    info("   - 只在一个地方（主入口）调用init_logger()")
    info("   - 使用统一的日志配置")
    info("   - 考虑为不同模块使用不同的日志名称（可选）")
    info("   - 避免在模块级别（导入时）初始化日志")


def main():
    """运行所有日志示例"""
    print("=== LogUtils 使用示例 ===")
    
    # 运行基本日志记录示例
    basic_logging_example()
    
    # 运行文件日志记录示例
    log_file_path = file_logging_example()
    
    # 运行异常日志记录示例
    exception_logging_example()
    
    # 运行结构化日志记录示例
    structured_logging_example()
    
    # 运行实际项目集成示例
    integration_example()
    
    # 新增：日志初始化时机示例
    initialization_timing_example()
    
    # 新增：重复初始化影响示例
    multiple_initialization_example()
    
    # 新增：多脚本调用场景示例
    multi_script_integration_example()
    
    print("\n=== 日志示例运行完成 ===")
    print(f"日志文件位置: {log_file_path}")
    print("请查看日志文件了解详细的日志记录内容。")


if __name__ == "__main__":
    main()