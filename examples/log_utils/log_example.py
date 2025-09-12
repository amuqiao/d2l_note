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
    
    print("\n=== 日志示例运行完成 ===")
    print(f"日志文件位置: {log_file_path}")
    print("请查看日志文件了解详细的日志记录内容。")


if __name__ == "__main__":
    main()