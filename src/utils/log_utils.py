"""日志管理工具类：提供统一的日志记录功能"""
import os
import sys
import logging
import datetime
import traceback
from typing import Optional, Union

# 利用Python logging模块的内置单例特性，直接使用命名logger作为全局单例
# 这个logger对象在整个Python进程中是唯一的
_global_logger = logging.getLogger('d2l_note')
_global_logger_initialized = False

# 配置全局logger，确保它不会继承root logger的处理器
_global_logger.propagate = False

# 确保在初始化前全局logger没有处理器
for handler in _global_logger.handlers[:]:
    handler.close()
    _global_logger.removeHandler(handler)


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
    
    def __new__(cls):
        # 所有LogUtils实例都共享同一个底层logger对象
        # 这里不使用传统的单例模式，而是让所有实例包装同一个全局logger
        instance = super(LogUtils, cls).__new__(cls)
        instance.logger = _global_logger
        return instance

    def __init__(self, auto_add_file_handler: bool = True):
        """初始化日志工具
        
        参数:
            auto_add_file_handler: 是否自动添加文件handler，默认为True
        """
        global _global_logger_initialized
        
        # 确保所有LogUtils实例都引用同一个全局logger
        self.logger = _global_logger
        self.log_file_path = None
        self.log_dir = None
        
        # 防止重复初始化
        if _global_logger_initialized:
            return
        
        # 设置基础配置
        self.logger.setLevel(logging.DEBUG)  # 默认设置为最低级别，让handler来控制
        
        # 标记为已初始化
        _global_logger_initialized = True
        
        # 清空已有的handler（避免重复添加）
        self._clear_all_handlers()
        
        # 添加控制台handler
        self._add_console_handler()
        
        # 自动添加文件handler（默认启用）
        if auto_add_file_handler:
            self._auto_add_file_handler()

    def _clear_all_handlers(self):
        """清除所有处理器"""
        for handler in self.logger.handlers[:]:
            handler.close()  # 确保关闭handler以释放资源
            self.logger.removeHandler(handler)

    def is_initialized(self) -> bool:
        """检查日志系统是否已经初始化
        
        返回:
            bool: 是否已初始化
        """
        global _global_logger_initialized
        return _global_logger_initialized
        
    def _auto_add_file_handler(self):
        """自动添加文件handler（默认配置）"""
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
        
        # 使用print而不是self.info来避免触发额外的日志处理器创建
        print(f"2025-09-13 20:20:22 INFO ℹ️ 日志系统初始化完成，日志文件: {log_file_path}")

    def _add_console_handler(self):
        """添加控制台输出handler"""
        # 注意：在Python logging模块中，FileHandler是StreamHandler的子类
        # 所以需要特别区分真正的控制台处理器和文件处理器
        
        # 先清理所有真正的控制台处理器（StreamHandler但不是FileHandler）
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
        
        # 添加一个新的控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # 控制台默认显示INFO及以上级别
        console_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # 再次检查，确保没有真正的控制台处理器
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in self.logger.handlers):
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
        
        # 清理所有handler（确保完全重置）
        instance._clear_all_handlers()
        
        # 添加控制台handler
        instance._add_console_handler()
        
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
        return cls()


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

def is_logger_initialized():
    """检查全局日志系统是否已经初始化
    
    返回:
        bool: 是否已初始化
    """
    return logger.is_initialized()