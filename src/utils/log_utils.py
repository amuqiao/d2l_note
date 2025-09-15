import logging
import os
import sys
import logging
from typing import Optional, Union
from datetime import datetime

class CustomLogger:
    """
    自定义日志模块，支持控制台和文件输出，带图标标记和异常处理
    """
    
    # 日志级别对应的图标
    LOG_ICONS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🔥'
    }
    
    # 自定义的格式化器类，用于在日志记录时动态添加图标
    class IconFormatter(logging.Formatter):
        def format(self, record):
            # 动态添加图标到日志记录
            record.icon = CustomLogger.LOG_ICONS.get(record.levelname, '')
            return super().format(record)
    
    def __init__(
        self,
        name: str = __name__,
        log_file: Optional[str] = None,
        global_level: Union[str, int] = logging.INFO,
        console_level: Optional[Union[str, int]] = None,
        file_level: Optional[Union[str, int]] = None
    ):
        """
        初始化日志模块
        
        :param name: 日志名称
        :param log_file: 日志文件路径，None则不输出到文件
        :param global_level: 全局日志级别
        :param console_level: 控制台日志级别，None则使用全局级别
        :param file_level: 文件日志级别，None则使用全局级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(global_level)
        self.logger.handlers = []  # 清除已有的处理器
        
        self.global_level = global_level
        self.console_level = console_level or global_level
        self.file_level = file_level or global_level
        
        # 添加控制台处理器
        self._add_console_handler()
        
        # 添加文件处理器（如果指定了文件路径）
        if log_file:
            self.add_file_handler(log_file)
    
    def _add_console_handler(self) -> None:
        """添加控制台日志处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        
        # 控制台日志格式：带图标，简洁格式
        console_formatter = self.IconFormatter(
            '%(asctime)s %(icon)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
        self.console_handler = console_handler
    
    def add_file_handler(self, log_file: str) -> None:
        """
        添加文件日志处理器
        
        :param log_file: 日志文件路径
        """
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.file_level)
        
        # 文件日志格式：更详细，包含模块、函数和行号
        file_formatter = self.IconFormatter(
            '%(asctime)s [%(icon)s] %(levelname)s %(module)s.%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
        self.file_handler = file_handler
    
    def set_global_level(self, level: Union[str, int]) -> None:
        """设置全局日志级别"""
        self.global_level = level
        self.logger.setLevel(level)
    
    def set_console_level(self, level: Union[str, int]) -> None:
        """设置控制台日志级别"""
        self.console_level = level
        self.console_handler.setLevel(level)
    
    def set_file_level(self, level: Union[str, int]) -> None:
        """设置文件日志级别"""
        if hasattr(self, 'file_handler'):
            self.file_level = level
            self.file_handler.setLevel(level)
        else:
            raise ValueError("没有文件处理器，请先添加文件日志处理器")
    
    def debug(self, message: str) -> None:
        """输出DEBUG级别的日志"""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """输出INFO级别的日志"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """输出WARNING级别的日志"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """输出ERROR级别的日志"""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """输出CRITICAL级别的日志"""
        self.logger.critical(message)
    
    def exception(self, message: str, exc_info: bool = True) -> None:
        """
        记录异常信息
        
        :param message: 异常相关消息
        :param exc_info: 是否记录异常堆栈信息
        """
        self.logger.error(message, exc_info=exc_info)


# 全局日志实例，方便其他模块直接导入使用
def get_logger(
    name: str = __name__,
    log_file: Optional[str] = None,
    global_level: Union[str, int] = logging.INFO,
    console_level: Optional[Union[str, int]] = None,
    file_level: Optional[Union[str, int]] = None
) -> CustomLogger:
    """
    获取日志实例
    
    :param name: 日志名称
    :param log_file: 日志文件路径
    :param global_level: 全局日志级别
    :param console_level: 控制台日志级别
    :param file_level: 文件日志级别
    :return: 日志实例
    """
    return CustomLogger(
        name=name,
        log_file=log_file,
        global_level=global_level,
        console_level=console_level,
        file_level=file_level
    )


# 默认日志实例，可直接使用
logger = get_logger()
