#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MySQL数据库参数化查询脚本

功能：
- 支持参数化查询，避免SQL注入风险
- 默认返回字典列表（键为列名），可切换为元组列表
- 手动管理连接生命周期，适用于基础查询场景
- 完善的日志输出和错误处理
"""
import pymysql
from pymysql import Error
from pymysql.cursors import DictCursor
import logging

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 数据库配置（可根据实际情况修改）
DB_CONFIG = {
    'host': '172.18.2.199',
    'port': 3306,
    'user': 'pvuser',
    'password': 'xdgro6ZnYEheGj0R',
    'database': 'iot_platform',
    'charset': 'utf8mb4'  # 增加字符集配置，避免中文乱码
}

def query_mysql(sql, params=None, return_dict=True):
    """
    参数化MySQL查询函数

    Args:
        sql (str): SQL查询语句（使用%s作为占位符）
        params (tuple/list, optional): 查询参数，与SQL中的%s一一对应。默认None（无参数）
        return_dict (bool, optional): 是否返回字典列表。默认True（返回字典），False返回元组列表

    Returns:
        list: 查询结果（字典列表或元组列表），查询失败返回空列表
    """
    connection = None
    result = []
    try:
        # 建立数据库连接
        connection = pymysql.connect(**DB_CONFIG)
        logger.info(f"成功连接数据库: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")

        # 根据return_dict选择游标类型
        cursor_class = DictCursor if return_dict else None
        with connection.cursor(cursor_class) as cursor:
            # 执行参数化查询（params必须是元组或列表，无参数时传None）
            cursor.execute(sql, params)
            result = cursor.fetchall()

            # 日志输出结果信息
            result_type = "字典列表" if return_dict else "元组列表"
            logger.info(f"查询成功，返回{result_type}，共{len(result)}条数据")
            if result:
                logger.info(f"结果示例: {result[0]}")

    except Error as e:
        logger.error(f"数据库操作失败: {str(e)}")
        logger.error(f"SQL语句: {sql}")
        logger.error(f"查询参数: {params}")
    finally:
        # 关闭连接
        if connection:
            connection.close()
            logger.info("数据库连接已关闭")
    return result

if __name__ == "__main__":
    # -------------- 使用示例 --------------
    # 1. 带参数的查询（默认返回字典）
    logger.info("\n=== 示例1：带参数查询（返回字典）===")
    sql1 = "SELECT * FROM t_gec_file_ocr_record WHERE id > %s LIMIT 2"
    params1 = (100,)  # 注意：参数必须是元组（即使只有一个参数，也要加逗号）
    result1 = query_mysql(sql1, params1)

    # 2. 无参数的查询（返回元组列表）
    logger.info("\n=== 示例2：无参数查询（返回元组）===")
    sql2 = "SELECT count(*) AS total_count FROM t_gec_file_ocr_record LIMIT 2"
    result2 = query_mysql(sql2, return_dict=False)

    # 3. 多参数查询（返回字典）
    logger.info("\n=== 示例3：多参数查询（返回字典）===")
    sql3 = "SELECT id FROM t_gec_file_ocr_record WHERE id > %s AND id < %s LIMIT 2"
    params3 = (1, 100000000)  # 多参数按SQL占位符顺序传入
    result3 = query_mysql(sql3, params3)