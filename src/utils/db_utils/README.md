# MySQL DAO模块使用文档

本文档详细介绍了两个MySQL数据库访问对象(DAO)模块的设计、功能和使用方法，帮助您理解它们的架构差异和最佳实践。

## 目录结构
```
src/utils/db_utils/
├── test_mysql_direct_dao_example_v1.py  # 基础版本DAO模块
├── test_mysql_direct_dao_example_v2.py  # 改进版DAO模块（推荐使用）
└── README.md                            # 本文档
```

## 模块概述

两个模块均用于连接MySQL数据库并执行特定的租赁信息查询，但采用了不同的架构设计和资源管理方式。两个模块都支持：
- 参数化查询，提高安全性
- 结构化日志输出，便于调试和监控
- 批量查询多个任务ID的租赁信息

## V1版本架构与功能

### 架构设计
V1版本采用传统的三层架构设计，各组件职责明确分离：

```
┌────────────────────┐    ┌───────────────┐    ┌───────────────────┐
│ DatabaseConnection │ ── │ QueryExecutor │ ── │  RentQueryService │
└────────────────────┘    └───────────────┘    └───────────────────┘
      连接管理            SQL查询执行            业务逻辑封装
```

### 核心类说明

#### 1. DatabaseConnection
- **职责**：负责数据库连接的创建和关闭
- **主要方法**：
  - `connect()`: 建立数据库连接
  - `close()`: 关闭数据库连接

#### 2. QueryExecutor
- **职责**：负责执行SQL查询并处理结果
- **主要方法**：
  - `execute_query(sql, params=None, max_display=3)`: 执行参数化查询

#### 3. RentQueryService
- **职责**：封装特定业务查询逻辑
- **主要方法**：
  - `batch_query_by_task_ids(task_ids)`: 批量查询多个任务ID的租赁信息

### V1版本伪代码示例

```python
# V1版本设计模式伪代码
class DatabaseConnection:
    def __init__(self, config):
        self.config = config
        self.connection = None
    
    def connect(self):
        # 手动建立连接
        self.connection = create_connection(self.config)
        return self.connection is not None
    
    def close(self):
        # 手动关闭连接
        if self.connection:
            self.connection.close()

class QueryExecutor:
    def __init__(self, db_connection):
        # 依赖注入
        self.db_connection = db_connection
    
    def execute_query(self, sql, params=None):
        # 需要检查连接状态
        if not self.db_connection.connection:
            raise Error("连接未建立")
        # 执行查询
        with self.db_connection.connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()

# 使用流程
db = DatabaseConnection(config)
db.connect()  # 手动调用连接
try:
    executor = QueryExecutor(db)
    service = RentQueryService(executor)
    results = service.query(...)  # 执行业务查询
finally:
    db.close()  # 手动调用关闭
```

## V2版本架构与功能

### 架构设计
V2版本采用了上下文管理器模式，简化了资源管理，提高了代码的可读性和健壮性：

```
┌────────────────────┐    ┌──────────────┐    ┌──────────────────┐
│ DatabaseConnection │ ── │ BaseService  │ ── │ RentQueryService │
└────────────────────┘    └──────────────┘    └──────────────────┘
      上下文管理           通用查询封装            业务逻辑封装
```

### 核心类说明

#### 1. DatabaseConnection
- **职责**：数据库连接管理，实现上下文管理器接口
- **主要方法**：
  - `__enter__()`: 建立数据库连接（上下文管理器进入时自动调用）
  - `__exit__()`: 关闭数据库连接（上下文管理器退出时自动调用）

#### 2. BaseService
- **职责**：基础服务类，封装数据库连接和查询执行逻辑
- **主要方法**：
  - `get_cursor()`: 获取数据库游标上下文管理器
  - `execute_query(sql, params=None)`: 执行查询并返回结果

#### 3. RentQueryService
- **职责**：租赁信息查询服务类，继承自BaseService
- **主要方法**：
  - `batch_query_by_task_ids(task_ids)`: 批量查询多个任务ID的租赁信息

### V2版本伪代码示例

```python
# V2版本设计模式伪代码
class DatabaseConnection:
    def __init__(self, config):
        self.config = config
        self.connection = None
    
    def __enter__(self):
        # 上下文管理器进入时自动建立连接
        self.connection = create_connection(self.config)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 上下文管理器退出时自动关闭连接
        if self.connection:
            self.connection.close()

class BaseService:
    def __init__(self, config):
        self.config = config
    
    @contextmanager
    def get_cursor(self):
        # 嵌套上下文管理器，简化资源获取
        with DatabaseConnection(self.config) as db:
            with db.connection.cursor() as cursor:
                yield cursor
    
    def execute_query(self, sql, params=None):
        # 自动管理连接和游标生命周期
        with self.get_cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()

# 使用流程
service = RentQueryService(config)  # 直接创建服务
results = service.query(...)  # 内部自动管理连接生命周期
```

## 设计模式优缺点比较

### V1版本：手动资源管理模式

**优点：**
- 概念简单，易于理解
- 对连接生命周期有完全的控制
- 适用于需要长时间保持连接的场景

**缺点：**
- 需要手动调用connect()和close()方法
- 容易因忘记调用close()导致连接泄漏
- 代码冗长，需要try-finally语句保证资源释放
- 依赖关系复杂，组件间耦合度较高

### V2版本：上下文管理器模式

**优点：**
- 自动管理资源生命周期，无需手动关闭连接
- 代码简洁，可读性更好
- 通过with语句确保资源释放，即使发生异常
- 依赖关系简化，使用更加方便
- 符合Python的"优雅"设计理念

**缺点：**
- 对于不熟悉上下文管理器的开发者需要一定学习成本
- 每次查询都会创建新的连接（可通过连接池优化）

## 使用方法

### V1版本使用示例

```python
# 数据库配置
config = {
    "host": "数据库主机地址",
    "port": 3306,
    "user": "用户名",
    "password": "密码",
    "database": "数据库名",
}

# 创建数据库连接
conn = DatabaseConnection(config)
if not conn.connect():
    # 处理连接失败情况
    exit()

try:
    # 创建查询执行器
    executor = QueryExecutor(conn)
    # 创建业务服务
    service = RentQueryService(executor)
    
    # 执行批量查询
    task_ids = ["任务ID1", "任务ID2", ...]
    results = service.batch_query_by_task_ids(task_ids)
    
    # 处理查询结果
    for task_id, data in results.items():
        # 处理每个任务ID的数据
        print(f"任务{task_id}的结果数量: {len(data)}")
finally:
    # 确保连接关闭
    conn.close()
```

### V2版本使用示例（推荐）

```python
# 数据库配置
config = {
    "host": "数据库主机地址",
    "port": 3306,
    "user": "用户名",
    "password": "密码",
    "database": "数据库名",
}

# 直接创建业务服务
service = RentQueryService(config)

# 执行批量查询
task_ids = ["任务ID1", "任务ID2", ...]
results = service.batch_query_by_task_ids(task_ids)

# 处理查询结果
for task_id, data in results.items():
    # 处理每个任务ID的数据
    print(f"任务{task_id}的结果数量: {len(data)}")

# 无需手动关闭连接，上下文管理器会自动处理
```

## 最佳实践

1. **优先使用V2版本**：V2版本采用上下文管理器模式，代码更简洁，资源管理更安全

2. **参数化查询**：两个版本均支持参数化查询，请始终使用参数化查询而不是字符串拼接，防止SQL注入攻击

3. **错误处理**：在实际应用中，建议根据具体业务需求增强错误处理逻辑

4. **日志级别**：根据环境调整日志级别，生产环境建议使用INFO或更高级别

5. **连接池优化**：对于高并发场景，可以考虑在V2版本的基础上添加连接池机制，提高性能

6. **批量操作大小**：当批量查询的task_ids数量较多时，注意分批次处理，避免单条SQL过长

7. **代码复用**：对于类似的查询需求，可以参考这两个模块的设计模式，创建新的业务服务类

## 输入输出示例

#### 输入输出示例
输入：
```python
# 创建服务
service = RentQueryService(db_config)

# 执行查询
results = service.batch_query_by_task_ids(["ZJ2025042111220065", "ZJ2025060512080021"])
```

输出：
```
2023-10-25 15:30:45 - __main__ - INFO - 成功连接到MySQL数据库
2023-10-25 15:30:45 - __main__ - INFO - 数据库连接信息: 主机=172.18.2.199, 端口=3306, 数据库=iot_platform, 用户=pvuser
2023-10-25 15:30:45 - __main__ - INFO - 
正在批量查询2个task_id
2023-10-25 15:30:45 - __main__ - INFO - 查询结果数量: 10
2023-10-25 15:30:45 - __main__ - INFO - 
查询结果示例(前5条):
  记录1: ('WP20250421001', '建议通过', '已授权', '622848******1234', '农业银行', '123456', 'http://image.url/front1.jpg', datetime.datetime(2025, 4, 21, 11, 22, 30), 0)
  记录2: ('WP20250421002', '建议通过', '已授权', '622202******5678', '工商银行', '654321', 'http://image.url/front2.jpg', datetime.datetime(2025, 4, 21, 11, 23, 15), 0)
  记录3: ('WP20250605001', '建议通过', '已授权', '621700******9012', '建设银行', '112233', 'http://image.url/front3.jpg', datetime.datetime(2025, 6, 5, 12, 8, 45), 0)
  记录4: ('WP20250605002', '建议通过', '已授权', '621483******3456', '招商银行', '445566', 'http://image.url/front4.jpg', datetime.datetime(2025, 6, 5, 12, 9, 30), 0)
  记录5: ('WP20250605003', '建议通过', '已授权', '622666******7890', '中信银行', '778899', 'http://image.url/front5.jpg', datetime.datetime(2025, 6, 5, 12, 10, 15), 0)
2023-10-25 15:30:45 - __main__ - INFO - 数据库连接已关闭
```

## 总结

两个DAO模块实现了相同的功能，但采用了不同的设计模式：V1版本采用传统的手动资源管理模式，V2版本采用更加现代化的上下文管理器模式。推荐使用V2版本，因为它提供了更简洁的API和更安全的资源管理机制。根据实际需求，可以进一步扩展和优化这两个模块，例如添加连接池、支持事务、实现更复杂的查询功能等。