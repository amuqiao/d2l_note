#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MySQL数据库查询脚本

功能：使用pymysql库连接MySQL数据库并执行指定查询
特点：
v1版本 ：采用三层类结构

- DatabaseConnection ：负责数据库连接的创建和关闭
- QueryExecutor ：负责执行SQL查询
- RentQueryService ：封装特定业务查询

- 采用类结构封装，分离连接管理和查询执行
- 支持参数化查询，提高安全性
- 结构化日志输出，便于调试和监控
"""
import pymysql
from pymysql import Error
import logging

# 极简配置：控制台输出+精确时间
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# 使用示例
logger.info("极简日志配置，仅控制台输出")


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

                return result
        except Error as e:
            logger.error(f"查询执行错误: {e}")
            return None


class RentQueryService:
    """租赁信息查询服务类，封装特定业务查询"""

    # SQL查询语句模板 - 使用IN方式批量查询多个task_id
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
       rent.is_delete, 
       plan.task_id  -- 添加task_id字段便于结果分组
     FROM 
       pomp_base_work_plan_rent rent 
     INNER JOIN 
       pomp_base_work_plan plan 
       ON rent.work_plan_id = plan.id  -- 关联两表 
     WHERE 
       plan.task_id IN ({placeholders})  -- 使用IN批量查询多个task_id
     ORDER BY 
       plan.task_id, rent.update_time DESC;
    """

    def __init__(self, query_executor):
        """
        初始化租赁查询服务

        :param query_executor: QueryExecutor实例
        """
        self.query_executor = query_executor

    def batch_query_by_task_ids(self, task_ids):
        """
        批量查询多个任务ID的租赁信息
        注：单个查询也可以通过传入包含一个ID的列表来实现

        :param task_ids: 任务ID列表
        :return: 包含所有查询结果的字典，key为task_id，value为查询结果
        """
        if not task_ids:
            logger.warning("任务ID列表为空，无需查询")
            return {}
            
        logger.info(f"\n正在批量查询{len(task_ids)}个task_id")
        
        # 构建SQL查询的占位符
        placeholders = ', '.join(['%s'] * len(task_ids))
        sql = self.RENT_QUERY_SQL.format(placeholders=placeholders)
        
        # 执行一次SQL查询获取所有结果
        result = self.query_executor.execute_query(sql, task_ids)
        
        # 将结果按task_id分组
        all_results = {task_id: [] for task_id in task_ids}
        top_5_results = []
        
        if result:
            for row in result:
                # 最后一个字段是task_id
                task_id = row[-1]
                # 移除最后一个task_id字段，保持原有数据结构
                cleaned_row = row[:-1]
                all_results[task_id].append(cleaned_row)
                
                # 收集前5条结果
                if len(top_5_results) < 5:
                    top_5_results.append(cleaned_row)
            
            # 打印前5行结果
            logger.info(f"\n查询结果示例(前5条):")
            for i, row in enumerate(top_5_results):
                logger.info(f"  记录{i+1}: {row}")
        
        return all_results


def main():
    """主函数，协调各组件执行查询任务"""
    # 数据库配置
    db_config = {
        "host": "172.18.2.199",
        "port": 3306,
        "user": "pvuser",
        "password": "xdgro6ZnYEheGj0R",
        "database": "iot_platform",
    }

    # 任务ID列表
    task_ids = [
        "ZJ2025042111220065",
        "ZJ2025060512080021",
        "ZJ2025060516080248",
        "ZJ2025052211110160",
        "ZJ2025060516090296",
        "ZJ2025060516080245",
        "ZJ2025012112120014",
        "ZJ2025060618510204",
        "ZJ2025060516100318",
        "ZJ2025060618480067",
        "ZJ2025060516040121",
        "ZJ2025060516100329",
        "ZJ2025060618520245",
        "ZJ2025060618510174",
        "ZJ2025060618500150",
        "ZJ2025060618500151",
        "ZJ2025060618510187",
        "ZJ2025051411030035",
        "ZJ2025060516030090",
        "ZJ2025060516070218",
        "ZJ2025060516070207",
        "ZJ2025060516110337",
        "ZJ2025060516090263",
        "ZJ2025060516100316",
        "ZJ2025060516020036",
    ]

    # 创建数据库连接
    db_connection = DatabaseConnection(db_config)
    if not db_connection.connect():
        return

    try:
        # 创建查询执行器和查询服务
        query_executor = QueryExecutor(db_connection)
        rent_query_service = RentQueryService(query_executor)

        # 使用批量查询方法执行所有任务ID的查询
        rent_query_service.batch_query_by_task_ids(task_ids)

    finally:
        # 确保连接关闭
        db_connection.close()


if __name__ == "__main__":
    main()
