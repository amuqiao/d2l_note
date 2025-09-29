#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MySQL数据库查询脚本

功能：使用pymysql库连接MySQL数据库并执行指定查询
特点：
- 采用类结构封装，分离连接管理和查询执行
- 支持参数化查询，提高安全性
- 结构化日志输出，便于调试和监控
"""
import pymysql
from pymysql import Error
import logging

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """数据库连接管理类，负责数据库连接的创建和关闭"""
    
    def __init__(self, db_config):
        """
        初始化数据库连接配置
        
        :param db_config: 数据库配置字典，包含host、port、user等信息
        """
        self.db_config = db_config
        self.connection = None
    
    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(**self.db_config)
            logger.info("成功连接到MySQL数据库")
            logger.info(
                f"数据库连接信息: 主机={self.db_config['host']}, "
                f"端口={self.db_config['port']}, "
                f"数据库={self.db_config['database']}, "
                f"用户={self.db_config['user']}"
            )
            return True
        except Error as e:
            logger.error(f"数据库连接错误: {e}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")
            self.connection = None

class QueryExecutor:
    """查询执行类，负责执行SQL查询并处理结果"""
    
    def __init__(self, db_connection):
        """
        初始化查询执行器
        
        :param db_connection: DatabaseConnection实例
        """
        self.db_connection = db_connection
    
    def execute_query(self, sql, params=None, max_display=3):
        """
        执行SQL查询
        
        :param sql: SQL查询语句
        :param params: 查询参数，用于参数化查询
        :param max_display: 最大显示的结果数量
        :return: 查询结果
        """
        if not self.db_connection.connection:
            logger.error("没有有效的数据库连接，请先建立连接")
            return None
            
        try:
            with self.db_connection.connection.cursor() as cursor:
                # 执行参数化查询，防止SQL注入
                cursor.execute(sql, params or ())
                result = cursor.fetchall()
                logger.info(f"查询结果数量: {len(result)}")
                
                # 显示部分结果
                if result and max_display > 0:
                    logger.info(f"查询结果示例(前{max_display}条):")
                    for i, row in enumerate(result[:max_display]):
                        logger.info(f"  记录{i+1}: {row}")
                
                return result
        except Error as e:
            logger.error(f"查询执行错误: {e}")
            return None

class RentQueryService:
    """租赁信息查询服务类，封装特定业务查询"""
    
    # SQL查询语句模板
    RENT_QUERY_SQL = """
    SELECT 
       rent.work_plan_id, 
       rent.order_advice, 
       rent.auth_result, 
       rent.new_card_number, 
       rent.new_card_bank, 
       rent.new_card_line_number, 
       rent.picture_card_frontal, 
       rent.create_time, 
       rent.is_delete 
     FROM 
       pomp_base_work_plan_rent rent 
     INNER JOIN 
       pomp_base_work_plan plan 
       ON rent.work_plan_id = plan.id  -- 关联两表 
     WHERE 
       plan.task_id = %s  -- 通过task_id过滤 
     ORDER BY 
       rent.update_time DESC 
     LIMIT 10;
    """
    
    def __init__(self, query_executor):
        """
        初始化租赁查询服务
        
        :param query_executor: QueryExecutor实例
        """
        self.query_executor = query_executor
    
    def query_by_task_id(self, task_id):
        """
        根据任务ID查询租赁信息
        
        :param task_id: 任务ID
        :return: 查询结果
        """
        logger.info(f"\n正在查询task_id: {task_id}")
        return self.query_executor.execute_query(
            self.RENT_QUERY_SQL, 
            (task_id,)
        )

def main():
    """主函数，协调各组件执行查询任务"""
    # 数据库配置
    db_config = {
        'host': '172.18.2.199',
        'port': 3306,
        'user': 'pvuser',
        'password': 'xdgro6ZnYEheGj0R',
        'database': 'iot_platform'
    }
    
    # 任务ID列表
    task_ids = [
        "ZJ2025042111220065", "ZJ2025060512080021", "ZJ2025060516080248",
        "ZJ2025052211110160", "ZJ2025060516090296", "ZJ2025060516080245",
        "ZJ2025012112120014", "ZJ2025060618510204", "ZJ2025060516100318",
        "ZJ2025060618480067", "ZJ2025060516040121", "ZJ2025060516100329",
        "ZJ2025060618520245", "ZJ2025060618510174", "ZJ2025060618500150",
        "ZJ2025060618500151", "ZJ2025060618510187", "ZJ2025051411030035",
        "ZJ2025060516030090", "ZJ2025060516070218", "ZJ2025060516070207",
        "ZJ2025060516110337", "ZJ2025060516090263", "ZJ2025060516100316",
        "ZJ2025060516020036"
    ]
    
    # 创建数据库连接
    db_connection = DatabaseConnection(db_config)
    if not db_connection.connect():
        return
    
    try:
        # 创建查询执行器和查询服务
        query_executor = QueryExecutor(db_connection)
        rent_query_service = RentQueryService(query_executor)
        
        # 执行所有任务ID的查询
        for task_id in task_ids:
            rent_query_service.query_by_task_id(task_id)
            
    finally:
        # 确保连接关闭
        db_connection.close()

if __name__ == "__main__":
    main()
