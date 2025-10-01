#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MySQL数据库直接连接测试脚本

功能：使用pymysql库直接连接MySQL数据库并执行查询
特点：
- 采用直接连接方式，手动管理连接生命周期
- 执行固定SQL查询语句，不支持参数化
- 使用日志输出查询结果，适用于基础连接测试
"""
import pymysql
from pymysql import Error
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库配置全局变量
DB_CONFIG = {
    'host': '172.18.2.199',
    'port': 3306,
    'user': 'pvuser',
    'password': 'xdgro6ZnYEheGj0R',
    'database': 'iot_platform'
}

# SQL查询语句模板
SQL_QUERY_TEMPLATE = """SELECT 
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
 LIMIT 10;"""

# 任务ID列表
zj_ids = [
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
    "ZJ2025060516020036"
]

def query_mysql():
    connection = None
    try:
        # 建立数据库连接
        connection = pymysql.connect(**DB_CONFIG)
        logger.info("成功连接到MySQL数据库")
        logger.info(f"数据库连接信息: 主机={DB_CONFIG['host']}, 端口={DB_CONFIG['port']}, 数据库={DB_CONFIG['database']}, 用户={DB_CONFIG['user']}")

        # 对每个task_id执行查询
        for task_id in zj_ids:
            logger.info(f"\n正在查询task_id: {task_id}")
            # 执行查询
            with connection.cursor() as cursor:
                cursor.execute(SQL_QUERY_TEMPLATE, (task_id,))
                result = cursor.fetchall()
                logger.info(f"查询结果数量: {len(result)}")
                if result:
                    logger.info("查询结果示例:")
                    # 只打印前3条结果，避免输出过多
                    for i, row in enumerate(result[:3]):
                        logger.info(f"  记录{i+1}: {row}")
                else:
                    logger.info("没有找到匹配的记录")

    except Error as e:
        logger.error(f"数据库操作错误: {e}")
    finally:
        # 关闭数据库连接
        if connection:
            connection.close()
            logger.info("数据库连接已关闭")

if __name__ == "__main__":
    query_mysql()