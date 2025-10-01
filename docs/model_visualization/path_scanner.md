# PathScanner 模块文档

## 1. 功能介绍

PathScanner 是一个专门用于路径扫描和文件查找的工具类，主要负责在深度学习项目中查找训练目录、模型文件和指标文件。该模块提供了一套简洁而强大的静态方法，可以帮助用户快速定位项目中的关键文件和目录。

### 主要功能：
- 根据命名模式查找训练目录
- 获取最新修改的训练目录
- 在指定目录中查找模型文件
- 在指定目录中查找指标文件
- 支持递归和非递归两种查找模式

## 2. 类与方法详解

### PathScanner 类

```python
class PathScanner:
    """路径扫描模块：负责查找训练目录和各类文件"""
```

### 2.1 find_run_directories

```python
@staticmethod
def find_run_directories(
    pattern: str = "run_\*",  # 文件夹命名模式（支持通配符）
    root_dir: str = ".",     # 查找的根目录
    recursive: bool = False  # 是否递归查找所有子目录
) -> List[str]:
```

**功能**：根据模式查找所有层级的模型文件夹

**参数**：
- `pattern`：文件夹命名模式（支持 glob 通配符，如 "run_\*", "model_\?", "exp\_[0-9]\*"）
- `root_dir`：查找的起始根目录
- `recursive`：是否递归查找所有子目录（True 时会遍历 root_dir 下所有层级）

**返回值**：所有匹配的文件夹绝对路径列表

### 2.2 get_latest_run_directory

```python
@staticmethod
def get_latest_run_directory(
    pattern: str = "run_\*", 
    root_dir: str = ".",
    recursive: bool = False  # 是否递归查找所有子目录
) -> Optional[str]:
```

**功能**：获取最新修改的训练目录

**参数**：
- `pattern`：文件夹命名模式（支持 glob 通配符）
- `root_dir`：查找的起始根目录
- `recursive`：是否递归查找所有子目录

**返回值**：最新修改的文件夹绝对路径，无匹配时返回None

### 2.3 find_model_files

```python
@staticmethod
def find_model_files(
    pattern: str = "\*.pth",  # 模式参数前置，与其他方法保持一致
    root_dir: str = ".",     # 统一参数名为root_dir
    recursive: bool = False  # 支持递归查找
) -> List[str]:
```

**功能**：在指定目录中查找模型文件

**参数**：
- `pattern`：模型文件命名模式（支持 glob 通配符，如 "\*.pth", "model_\*.ckpt"）
- `root_dir`：查找的起始根目录
- `recursive`：是否递归查找所有子目录

**返回值**：所有匹配的模型文件绝对路径列表

### 2.4 find_metric_files

```python
@staticmethod
def find_metric_files(
    pattern: str = "\*.json",  # 模式参数前置，与其他方法保持一致
    root_dir: str = ".",      # 统一参数名为root_dir
    recursive: bool = False   # 支持递归查找
) -> List[str]:
```

**功能**：在指定目录中查找指标文件

**参数**：
- `pattern`：指标文件命名模式（支持 glob 通配符，如 "\*.json", "metrics_\*.log"）
- `root_dir`：查找的起始根目录
- `recursive`：是否递归查找所有子目录

**返回值**：所有匹配的指标文件绝对路径列表

## 3. 使用示例

### 3.1 基本用法

```python
from src.model_visualization.path_scanner import PathScanner

# 1. 查找直接子目录
run_dirs = PathScanner.find_run_directories(root_dir="/path/to/runs")

# 2. 递归查找所有子目录
all_run_dirs = PathScanner.find_run_directories(root_dir="/path/to/runs", recursive=True)

# 3. 获取最新修改的训练目录
latest_run = PathScanner.get_latest_run_directory(root_dir="/path/to/runs")

# 4. 查找模型文件
model_files = PathScanner.find_model_files(root_dir="/path/to/runs", recursive=True)

# 5. 查找指标文件
metric_files = PathScanner.find_metric_files(root_dir="/path/to/runs", recursive=True)
```

### 3.2 自定义查找模式

```python
# 查找特定前缀的目录
special_dirs = PathScanner.find_run_directories(pattern="exp_\*", root_dir=".")

# 查找特定后缀的模型文件
best_model_files = PathScanner.find_model_files(pattern="\*_best.pth", root_dir=".", recursive=True)

# 查找特定格式的指标文件
log_files = PathScanner.find_metric_files(pattern="metrics_\*.log", root_dir=".", recursive=True)
```

## 4. 最佳实践

### 4.1 性能优化
- 在大型项目中，尽量指定具体的 `root_dir` 以缩小搜索范围
- 非必要时，避免使用 `recursive=True`，因为递归搜索会增加系统开销
- 对于频繁调用的场景，可以缓存搜索结果

### 4.2 路径处理
- 使用绝对路径作为 `root_dir`，避免相对路径带来的不确定性
- 注意检查返回的路径列表是否为空，避免后续操作出现异常
- 对于特殊字符或中文路径，请确保编码正确

### 4.3 错误处理
- 在调用前检查根目录是否存在
- 处理可能的权限问题（当访问受限目录时）
- 对于大量文件的搜索结果，考虑分批处理

## 5. 模块集成

PathScanner 模块可以与项目中的其他模块集成使用，例如：

- 与模型分析工具集成，自动查找并分析最近的训练结果
- 与可视化模块集成，为图表生成提供数据文件路径
- 与批处理脚本集成，批量处理多个训练目录的结果

## 6. 输入输出示例

#### 输入输出示例

输入：
```python
# 查找当前项目下最新的运行目录
latest_run = PathScanner.get_latest_run_directory(root_dir="./runs", recursive=True)
print(f"最新运行目录: {latest_run}")
```

输出：
```
最新运行目录: /data/home/project/d2l_note/runs/run_20231015_143022
```

输入：
```python
# 查找所有模型文件
model_files = PathScanner.find_model_files(root_dir="./runs", recursive=True)
print(f"找到 {len(model_files)} 个模型文件")
for file in model_files[:3]:
    print(f"- {file}")
```

输出：
```
找到 12 个模型文件
- /data/home/project/d2l_note/runs/run_20231015_143022/model_best.pth
- /data/home/project/d2l_note/runs/run_20231015_143022/model_epoch_10.pth
- /data/home/project/d2l_note/runs/run_20231014_091536/model_best.pth
```