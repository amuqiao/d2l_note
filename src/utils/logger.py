"""新的日志管理模块：提供统一、高效的日志记录功能"""
import os
import sys
import logging
import datetime
import traceback
from typing import Optional, Union, Dict, Any


class Logger:
    """日志管理类：支持多种日志级别、控制台输出和文件记录"""
    
    # 类变量，用于实现单例模式
    _instance = None
    _initialized = False
    
    # 项目配置常量
    DEFAULT_LOGGER_NAME = 'd2l_note'
    DEFAULT_LOG_DIR = 'logs'
    DEFAULT_LOG_FILE_PREFIX = 'd2l_note'
    
    # 日志级别映射
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # 日志类型对应的图标
    LOG_ICONS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'SUCCESS': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '💥'
    }
    
    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, auto_init: bool = True):
        """初始化日志工具
        
        参数:
            auto_init: 是否自动初始化，默认为True
        """
        # 防止重复初始化
        if self.__class__._initialized:
            return
        
        # 创建logger实例
        self.logger = logging.getLogger(self.DEFAULT_LOGGER_NAME)
        self.logger.setLevel(logging.DEBUG)  # 默认设置为最低级别，让handler来控制
        self.logger.propagate = False  # 不继承root logger的处理器
        
        # 初始化属性
        self.log_file_path = None
        self.log_dir = None
        self.console_handler = None
        self.file_handler = None
        
        # 清除已有的handler
        self._clear_all_handlers()
        
        # 标记为已初始化
        self.__class__._initialized = True
        
        # 自动初始化（默认启用）
        if auto_init:
            self._setup_default_handlers()
    
    def _clear_all_handlers(self):
        """清除所有处理器"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        self.console_handler = None
        self.file_handler = None
    
    def _setup_default_handlers(self):
        """设置默认的处理器"""
        self.add_console_handler()
        self.add_file_handler(use_timestamp=False)
        # 添加简洁的打印信息
        print(f"[Logger初始化] 默认日志系统已启动，日志文件: {self.log_file_path}")

    @classmethod
    def is_initialized(cls) -> bool:
        """检查日志系统是否已经初始化

        返回:
            bool: 是否已初始化
        """
        return cls._initialized
    
    def add_console_handler(self, log_level: str = 'INFO', formatter: Optional[logging.Formatter] = None):
        """添加控制台输出处理器
        
        参数:
            log_level: 日志级别，默认为INFO
            formatter: 日志格式化器，如果为None则使用默认格式化器
        """
        # 移除已有的控制台处理器
        if self.console_handler:
            self.logger.removeHandler(self.console_handler)
            self.console_handler.close()
            self.console_handler = None
        
        # 创建新的控制台处理器
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(self.LOG_LEVELS.get(log_level.upper(), logging.INFO))
        
        # 设置格式化器
        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        self.console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(self.console_handler)
    
    def add_file_handler(
        self, 
        log_file_path: Optional[str] = None, 
        log_level: str = 'DEBUG',
        formatter: Optional[logging.Formatter] = None,
        use_timestamp: bool = False
    ):
        """添加文件输出处理器
        
        参数:
            log_file_path: 日志文件路径，如果为None则自动生成
            log_level: 日志级别，默认为DEBUG
            formatter: 日志格式化器，如果为None则使用默认格式化器
            use_timestamp: 是否在默认文件名中添加时间戳，默认为False
        """
        # 移除已有的文件处理器
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.file_handler = None
        
        # 生成默认日志文件路径
        if log_file_path is None:
            log_dir = self.DEFAULT_LOG_DIR
            if use_timestamp:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = f'{self.DEFAULT_LOG_FILE_PREFIX}_{timestamp}.log'
            else:
                log_file = f'{self.DEFAULT_LOG_FILE_PREFIX}.log'
            log_file_path = os.path.join(log_dir, log_file)
        
        # 确保日志目录存在
        self.log_dir = os.path.dirname(log_file_path)
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建文件处理器
        self.file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        self.file_handler.setLevel(self.LOG_LEVELS.get(log_level.upper(), logging.DEBUG))
        
        # 设置格式化器
        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        self.file_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(self.file_handler)
        self.log_file_path = log_file_path
    
    @classmethod
    def init(
        cls, 
        log_dir: str = None, 
        log_file: str = None, 
        log_level: str = 'INFO',
        console_level: str = 'INFO',
        use_timestamp: bool = True
    ) -> 'Logger':
        """
        初始化日志系统
        
        参数:
            log_dir: 日志文件保存目录，默认使用DEFAULT_LOG_DIR
            log_file: 日志文件名，如果不指定则自动生成
            log_level: 文件日志级别，默认为INFO
            console_level: 控制台日志级别，默认为INFO
            use_timestamp: 日志文件名是否包含时间戳
            
        返回:
            Logger实例
        """
        instance = cls(auto_init=False)  # 先创建实例但不自动初始化
        
        # 清理所有handler
        instance._clear_all_handlers()
        
        # 添加控制台handler
        instance.add_console_handler(log_level=console_level)
        
        # 生成日志文件路径
        if log_file is None:
            log_dir = log_dir or cls.DEFAULT_LOG_DIR
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if use_timestamp else ''
            log_file = f'{cls.DEFAULT_LOG_FILE_PREFIX}_{timestamp}.log' if timestamp else f'{cls.DEFAULT_LOG_FILE_PREFIX}.log'
        log_file_path = os.path.join(log_dir, log_file)
        
        # 添加文件handler
        instance.add_file_handler(log_file_path, log_level=log_level)
        
        # 添加简洁的打印信息
        print(f"[Logger初始化] 自定义日志系统已启动，日志文件: {log_file_path} (控制台级别:{console_level},文件级别:{log_level})")
        
        instance.info(f"日志系统初始化完成，日志文件: {log_file_path}")
        return instance
    
    def debug(self, message: str, *args, **kwargs):
        """记录调试日志"""
        icon = self.LOG_ICONS.get('DEBUG', '')
        self.logger.debug(f"{icon} {message}", *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """记录信息日志"""
        icon = self.LOG_ICONS.get('INFO', '')
        self.logger.info(f"{icon} {message}", *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """记录成功日志"""
        icon = self.LOG_ICONS.get('SUCCESS', '')
        # 自定义SUCCESS级别（使用INFO的级别值）
        self.logger.info(f"{icon} {message}", *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """记录警告日志"""
        icon = self.LOG_ICONS.get('WARNING', '')
        self.logger.warning(f"{icon} {message}", *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """记录错误日志"""
        icon = self.LOG_ICONS.get('ERROR', '')
        self.logger.error(f"{icon} {message}", *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """记录严重错误日志"""
        icon = self.LOG_ICONS.get('CRITICAL', '')
        self.logger.critical(f"{icon} {message}", *args, **kwargs)
    
    def exception(self, message: str, exc_info: bool = True, *args, **kwargs):
        """记录异常信息"""
        icon = self.LOG_ICONS.get('ERROR', '')
        self.logger.error(f"{icon} {message}", exc_info=exc_info, *args, **kwargs)
    
    def log_dict(self, title: str, data: dict, log_level: str = 'INFO'):
        """以结构化方式记录字典数据"""
        log_func = getattr(self, log_level.lower(), self.info)
        log_func(f"{title}:")
        for key, value in data.items():
            log_func(f"  - {key}: {value}")
    
    def get_log_dir(self) -> Optional[str]:
        """获取当前日志目录"""
        return self.log_dir
    
    def get_log_file_path(self) -> Optional[str]:
        """获取当前日志文件路径"""
        return self.log_file_path
    
    def set_level(self, level: str):
        """设置logger的全局日志级别"""
        self.logger.setLevel(self.LOG_LEVELS.get(level.upper(), logging.DEBUG))
    
    def set_console_level(self, level: str):
        """设置控制台处理器的日志级别"""
        if self.console_handler:
            self.console_handler.setLevel(self.LOG_LEVELS.get(level.upper(), logging.INFO))
    
    def set_file_level(self, level: str):
        """设置文件处理器的日志级别"""
        if self.file_handler:
            self.file_handler.setLevel(self.LOG_LEVELS.get(level.upper(), logging.DEBUG))
    
    @classmethod
    def get_instance(cls) -> 'Logger':
        """获取日志工具实例"""
        return cls()
    
    @classmethod
    def reset(cls):
        """重置日志系统"""
        if cls._instance:
            cls._instance._clear_all_handlers()
        cls._instance = None
        cls._initialized = False


# 创建全局日志实例，方便直接调用
logger = Logger()


# 提供直接使用的函数接口
def init(
    log_dir: str = 'logs', 
    log_file: str = None, 
    log_level: str = 'INFO',
    console_level: str = 'INFO',
    use_timestamp: bool = True
) -> Logger:
    """初始化全局日志系统"""
    return Logger.init(log_dir, log_file, log_level, console_level, use_timestamp)

def debug(message: str, *args, **kwargs):
    """记录调试日志"""
    logger.debug(message, *args, **kwargs)

def info(message: str, *args, **kwargs):
    """记录信息日志"""
    logger.info(message, *args, **kwargs)

def success(message: str, *args, **kwargs):
    """记录成功日志"""
    logger.success(message, *args, **kwargs)

def warning(message: str, *args, **kwargs):
    """记录警告日志"""
    logger.warning(message, *args, **kwargs)

def error(message: str, *args, **kwargs):
    """记录错误日志"""
    logger.error(message, *args, **kwargs)

def critical(message: str, *args, **kwargs):
    """记录严重错误日志"""
    logger.critical(message, *args, **kwargs)

def exception(message: str, exc_info: bool = True, *args, **kwargs):
    """记录异常信息"""
    logger.exception(message, exc_info=exc_info, *args, **kwargs)

def log_dict(title: str, data: dict, log_level: str = 'INFO'):
    """以结构化方式记录字典数据"""
    logger.log_dict(title, data, log_level)

def is_initialized():
    """检查全局日志系统是否已经初始化
    
    返回:
        bool: 是否已初始化
    """
    return Logger.is_initialized()

def get_logger():
    """获取全局日志实例"""
    return logger

def reset_logger():
    """重置全局日志系统"""
    Logger.reset()