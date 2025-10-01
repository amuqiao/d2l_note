#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MySQL数据库查询脚本（改进版）

功能：使用pymysql库连接MySQL数据库并执行指定查询
特点：
- 自动管理数据库连接，无需手动创建和关闭
- 采用上下文管理器模式，确保资源正确释放
- 支持参数化查询，提高安全性
- 结构化日志输出，便于调试和监控
"""
import pymysql
from pymysql import Error
import logging
from contextlib import contextmanager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """数据库连接管理类，实现上下文管理器接口"""

    def __init__(self, db_config):
        """初始化数据库连接配置"""
        self.db_config = db_config
        self.connection = None

    def __enter__(self):
        """建立数据库连接（上下文管理器进入时调用）"""
        try:
            self.connection = pymysql.connect(** self.db_config)
            logger.info("成功连接到MySQL数据库")
            logger.info(
                f"数据库连接信息: 主机={self.db_config['host']}, "
                f"端口={self.db_config['port']}, "
                f"数据库={self.db_config['database']}, "
                f"用户={self.db_config['user']}"
            )
            return self
        except Error as e:
            logger.error(f"数据库连接错误: {e}")
            raise  # 抛出异常，让上下文管理器处理

    def __exit__(self, exc_type, exc_val, exc_tb):
        """关闭数据库连接（上下文管理器退出时调用）"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")
        # 如果有异常发生，会在这里自动传播，不需要额外处理


class BaseService:
    """基础服务类，封装数据库连接和查询执行逻辑"""
    
    def __init__(self, db_config):
        """初始化服务，保存数据库配置"""
        self.db_config = db_config

    @contextmanager
    def get_cursor(self):
        """获取数据库游标上下文管理器，自动处理连接和游标生命周期"""
        with DatabaseConnection(self.db_config) as db:
            with db.connection.cursor() as cursor:
                yield cursor

    def execute_query(self, sql, params=None):
        """执行查询并返回结果"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, params or ())
                result = cursor.fetchall()
                logger.info(f"查询结果数量: {len(result)}")
                return result
        except Error as e:
            logger.error(f"查询执行错误: {e}")
            return None


class RentQueryService(BaseService):
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

    def batch_query_by_task_ids(self, task_ids):
        """
        批量查询多个任务ID的租赁信息
        注：单个查询也可以通过传入包含一个ID的列表来实现
        """
        if not task_ids:
            logger.warning("任务ID列表为空，无需查询")
            return {}
            
        logger.info(f"\n正在批量查询{len(task_ids)}个task_id")
        
        # 构建SQL查询的占位符
        placeholders = ', '.join(['%s'] * len(task_ids))
        sql = self.RENT_QUERY_SQL.format(placeholders=placeholders)
        
        # 执行一次SQL查询获取所有结果
        result = self.execute_query(sql, task_ids)
        
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
    """主函数，演示如何使用服务类"""
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
        # 可以继续添加更多task_id...
    ]

    # 创建查询服务实例（用户只需要关心这里）
    rent_service = RentQueryService(db_config)
    
    # 执行查询（用户只需要调用服务方法）
    results = rent_service.batch_query_by_task_ids(task_ids)
    
    # 可以在这里处理查询结果
    # ...


if __name__ == "__main__":
    main()
