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
- 支持将查询结果导出到CSV文件
"""
import pymysql
from pymysql import Error
import logging
from contextlib import contextmanager
import csv
import os
from datetime import datetime

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
        """执行查询并返回字典格式的结果"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, params or ())
                # 获取列名
                columns = [desc[0] for desc in cursor.description]
                # 获取结果并转换为字典列表
                result = []
                for row in cursor.fetchall():
                    result.append(dict(zip(columns, row)))
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
            return []
            
        logger.info(f"\n正在批量查询{len(task_ids)}个task_id")
        
        # 构建SQL查询的占位符
        placeholders = ', '.join(['%s'] * len(task_ids))
        sql = self.RENT_QUERY_SQL.format(placeholders=placeholders)
        
        # 执行一次SQL查询获取所有结果
        result = self.execute_query(sql, task_ids)
        
        processed_results = []
        
        if result:
            for row in result:
                # 创建一个新的字典来存储处理后的数据
                processed_row = row.copy()
                
                # 处理picture_card_frontal字段
                picture_card_frontal = processed_row.get('picture_card_frontal')
                if picture_card_frontal and isinstance(picture_card_frontal, str):
                    # 截断///后面的部分
                    if '///' in picture_card_frontal:
                        truncated_part = picture_card_frontal.split('///')[0]
                        # 拼接前缀
                        processed_row['picture_card_frontal'] = f"https://xuntian-pv.tcl.com/{truncated_part}"
                    else:
                        # 如果没有///，直接拼接前缀
                        processed_row['picture_card_frontal'] = f"https://xuntian-pv.tcl.com/{picture_card_frontal}"
                
                processed_results.append(processed_row)

        return processed_results


def save_results_to_csv(results, output_dir='.', file_name=None):
    """
    将查询结果保存到CSV文件
    
    参数:
    - results: 查询结果列表
    - output_dir: 输出目录，默认为当前目录
    - file_name: 文件名，默认为None（将自动生成包含时间戳的文件名）
    
    返回:
    - 保存的文件路径
    """
    if not results:
        logger.warning("结果为空，无需保存到CSV")
        return None
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果未提供文件名，则生成包含时间戳的文件名
    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"bank_card_query_result_{timestamp}.csv"
    
    # 完整文件路径
    file_path = os.path.join(output_dir, file_name)
    
    try:
        # 获取所有字段名，并确保重要字段排在前面
        first_row = results[0]
        all_fields = list(first_row.keys())
        
        # 定义需要排在前面的字段
        priority_fields = ['task_id', 'new_card_number', 'picture_card_frontal']
        
        # 构建排好序的字段列表：重要字段在前，其余字段按原顺序排列
        sorted_fields = []
        for field in priority_fields:
            if field in all_fields:
                sorted_fields.append(field)
                all_fields.remove(field)
        sorted_fields.extend(all_fields)
        
        # 写入CSV文件
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted_fields)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据
            for row in results:
                writer.writerow(row)
        
        logger.info(f"查询结果已成功保存到CSV文件: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"保存CSV文件时发生错误: {e}")
        return None


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

    # 创建查询服务实例（用户只需要关心这里）
    rent_service = RentQueryService(db_config)
    
    # 执行查询（用户只需要调用服务方法）
    results = rent_service.batch_query_by_task_ids(task_ids)
    
    # 打印所有查询结果
    logger.info(f"\n查询结果总数: {len(results)}")
    print(results)
        
    # 将结果保存到CSV文件（默认保存在当前目录）
    csv_file_path = save_results_to_csv(results)
    
    # 如果需要指定目录，可以使用如下方式：
    # csv_file_path = save_results_to_csv(results, output_dir='/path/to/directory')
    
    if csv_file_path:
        logger.info(f"CSV文件已保存至: {csv_file_path}")


if __name__ == "__main__":
    main()
