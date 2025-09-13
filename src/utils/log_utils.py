"""日志管理工具类：提供统一的日志记录功能"""
import os
import sys
import logging
import datetime
import traceback
from typing import Optional, Union


class LogUtils:
    """日志管理工具类：支持多种日志级别、控制台输出和文件记录"""
    
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
    
    # 单例模式实例
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogUtils, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, auto_add_file_handler: bool = True):
        """初始化日志工具
        
        参数:
            auto_add_file_handler: 是否自动添加文件handler，默认为True
        """
        # 防止多次初始化
        if self._initialized:
            return
        
        self.logger = logging.getLogger('d2l_note')
        self.logger.setLevel(logging.DEBUG)  # 默认设置为最低级别，让handler来控制
        self.log_file_path = None
        self.log_dir = None
        self._initialized = True
        
        # 清空已有的handler（避免重复添加）
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # 添加控制台handler（默认启用）
        self._add_console_handler()
        
        # 自动添加文件handler（默认启用）
        if auto_add_file_handler:
            log_dir = 'logs'
            log_file = None
            log_level = 'INFO'
            use_timestamp = False
            
            # 创建日志目录
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            # 生成日志文件名
            if log_file is None:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if use_timestamp else ''
                log_file = f'd2l_note_{timestamp}.log' if timestamp else 'd2l_note.log'
            
            log_file_path = os.path.join(log_dir, log_file)
            
            # 添加文件handler
            self._add_file_handler(log_file_path, log_level)
            
            self.info(f"日志系统初始化完成，日志文件: {log_file_path}")
    
    def _add_console_handler(self):
        """添加控制台输出handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # 控制台默认显示INFO及以上级别
        console_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self, log_file_path: str, log_level: str = 'DEBUG'):
        """添加文件输出handler"""
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # 创建文件handler
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(self.LOG_LEVELS.get(log_level.upper(), logging.DEBUG))
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file_path = log_file_path
        self.log_dir = log_dir
    
    @classmethod
    def init_logger(
        cls, 
        log_dir: str = 'logs', 
        log_file: str = None, 
        log_level: str = 'INFO',
        use_timestamp: bool = True
    ) -> 'LogUtils':
        """
        初始化日志系统
        
        参数:
            log_dir: 日志文件保存目录
            log_file: 日志文件名，如果不指定则自动生成
            log_level: 日志级别，默认为INFO
            use_timestamp: 日志文件名是否包含时间戳
            
        返回:
            LogUtils实例
        """
        instance = cls()
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # 生成日志文件名
        if log_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if use_timestamp else ''
            log_file = f'd2l_note_{timestamp}.log' if timestamp else 'd2l_note.log'
        
        log_file_path = os.path.join(log_dir, log_file)
        
        # 添加文件handler
        instance._add_file_handler(log_file_path, log_level)
        
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
    
    @classmethod
    def get_logger(cls) -> 'LogUtils':
        """获取日志工具实例"""
        if not cls._instance:
            cls._instance = cls()
        return cls._instance


# 创建全局日志实例，方便直接调用
logger = LogUtils()


# 提供直接使用的函数接口
def init_logger(
    log_dir: str = 'logs', 
    log_file: str = None, 
    log_level: str = 'INFO',
    use_timestamp: bool = False
) -> LogUtils:
    """初始化全局日志系统"""
    return LogUtils.init_logger(log_dir, log_file, log_level, use_timestamp)

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