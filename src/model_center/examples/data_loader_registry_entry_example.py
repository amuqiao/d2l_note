"""
数据加载器注册示例：展示如何使用数据加载器注册中心和装饰器进行数据集的自动注册
"""
import os
import sys
import logging
import torch
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到Python路径，以便能正确导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 设置日志级别为DEBUG，以便查看注册信息
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入数据加载器注册中心入口
from ..data_loader_registry_entry import (
    register_dataset,
    register_dataset_config,
    register_dataset_preprocess_func,
    create_dataset,
    create_data_loader,
    create_train_test_loaders,
    get_dataset_config,
    get_dataset_preprocess_func,
    get_all_registered_datasets,
    is_dataset_registered,
    initialize_data_loaders
)

# ------------------------------
# 示例1：使用装饰器注册简单数据集
# ------------------------------

@register_dataset(name="simple_dataset", namespace="demo")
class SimpleDataset(Dataset):
    """简单数据集类"""
    def __init__(self, size=100, train=True):
        self.train = train
        # 生成简单的随机数据
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.labels[idx]
        }

# ------------------------------
# 示例2：使用装饰器注册带配置和预处理的数据集
# ------------------------------

# 定义预处理函数
def preprocess_dataset(dataset):
    """数据集预处理函数"""
    # 对数据进行标准化处理
    mean = dataset.data.mean(dim=0)
    std = dataset.data.std(dim=0)
    dataset.data = (dataset.data - mean) / (std + 1e-8)  # 避免除零
    logging.info("数据集已预处理：标准化数据")
    return dataset

@register_dataset(
    name="advanced_dataset", 
    namespace="demo",
    config={
        "size": 200,
        "feature_dim": 16,
        "num_classes": 3
    },
    preprocess_func=preprocess_dataset
)
class AdvancedDataset(Dataset):
    """高级数据集类"""
    def __init__(self, size=200, feature_dim=16, num_classes=3, train=True):
        self.train = train
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        # 生成更复杂的随机数据
        self.data = torch.randn(size, feature_dim)
        self.labels = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回更多信息
        return {
            "data": self.data[idx],
            "label": self.labels[idx],
            "is_train": self.train,
            "feature_dim": self.feature_dim
        }

# ------------------------------
# 示例3：使用register_dataset_config注册数据集配置
# ------------------------------

def register_dataset_configs():
    """注册数据集配置"""
    # 注册一个用于图像分类的数据集配置
    register_dataset_config(
        "image_dataset",
        {
            "root_dir": "./data/images",
            "image_size": 224,
            "transforms": [
                "resize",
                "random_crop",
                "horizontal_flip",
                "to_tensor",
                "normalize"
            ],
            "batch_size": 32
        },
        namespace="demo"
    )
    
    # 注册一个用于文本分类的数据集配置
    register_dataset_config(
        "text_dataset",
        {
            "data_file": "./data/text.csv",
            "tokenizer": "bert-base-uncased",
            "max_length": 128,
            "batch_size": 16
        },
        namespace="demo"
    )

# ------------------------------
# 示例4：使用register_dataset_preprocess_func注册预处理函数
# ------------------------------

def register_preprocess_functions():
    """注册预处理函数"""
    # 为简单数据集注册另一个预处理函数
    def simple_preprocess(dataset):
        # 为简单数据集添加噪声
        noise = torch.randn_like(dataset.data) * 0.1
        dataset.data += noise
        logging.info("简单数据集已预处理：添加噪声")
        return dataset
    
    register_dataset_preprocess_func(
        "simple_dataset",
        simple_preprocess,
        namespace="demo"
    )

# ------------------------------
# 演示函数：展示数据加载器注册和使用的完整流程
# ------------------------------

def demonstrate_data_loader_registry():
    """演示数据加载器注册和使用"""
    print("=" * 50)
    print("数据加载器注册中心演示")
    print("=" * 50)
    
    # 1. 注册数据集配置和预处理函数
    print("\n1. 注册数据集配置和预处理函数...")
    register_dataset_configs()
    register_preprocess_functions()
    
    # 2. 初始化数据加载器系统
    print("\n2. 初始化数据加载器系统...")
    initialize_data_loaders()
    
    # 3. 检查数据集是否注册成功
    print("\n3. 检查数据集注册状态:")
    datasets_to_check = [
        ("simple_dataset", "demo"),
        ("advanced_dataset", "demo"),
        ("image_dataset", "demo"),
        ("text_dataset", "demo")
    ]
    
    for dataset_name, namespace in datasets_to_check:
        is_registered = is_dataset_registered(dataset_name, namespace)
        print(f"  - 数据集 '{dataset_name}' (命名空间: '{namespace}') 是否已注册: {is_registered}")
    
    # 4. 获取所有注册的数据集
    print("\n4. 获取所有注册的数据集:")
    all_datasets = get_all_registered_datasets(namespace="demo")
    for ns, datasets in all_datasets.items():
        print(f"  - 命名空间 '{ns}' 中的数据集: {', '.join(datasets)}")
    
    # 5. 获取数据集配置和预处理函数
    print("\n5. 获取数据集配置和预处理函数:")
    
    # 获取高级数据集配置
    adv_config = get_dataset_config("advanced_dataset", "demo")
    print(f"  - 高级数据集配置: {adv_config}")
    
    # 获取简单数据集预处理函数
    simple_preprocess_func = get_dataset_preprocess_func("simple_dataset", "demo")
    print(f"  - 简单数据集预处理函数是否存在: {simple_preprocess_func is not None}")
    
    # 6. 创建数据集实例
    print("\n6. 创建数据集实例:")
    
    # 创建简单数据集实例
    simple_dataset = create_dataset("simple_dataset", "demo", size=50)
    print(f"  - 简单数据集大小: {len(simple_dataset)}")
    sample = simple_dataset[0]
    print(f"  - 简单数据集样本: 数据形状={sample['data'].shape}, 标签={sample['label']}")
    
    # 创建高级数据集实例（会自动应用预处理）
    advanced_dataset = create_dataset("advanced_dataset", "demo", size=100)
    print(f"  - 高级数据集大小: {len(advanced_dataset)}")
    sample = advanced_dataset[0]
    print(f"  - 高级数据集样本: 数据形状={sample['data'].shape}, 标签={sample['label']}, feature_dim={sample['feature_dim']}")
    
    # 7. 创建数据加载器
    print("\n7. 创建数据加载器:")
    
    # 创建简单数据集的数据加载器
    simple_loader = create_data_loader(
        "simple_dataset", 
        "demo",
        batch_size=4,
        shuffle=True,
        num_workers=0,  # 在Windows上设置为0避免多进程问题
        size=20
    )
    print(f"  - 简单数据加载器批次数量: {len(simple_loader)}")
    
    # 获取第一个批次的数据
    for batch in simple_loader:
        print(f"  - 批次数据形状: {batch['data'].shape}")
        print(f"  - 批次标签形状: {batch['label'].shape}")
        break  # 只打印第一个批次
    
    # 8. 创建训练和测试数据加载器
    print("\n8. 创建训练和测试数据加载器:")
    
    train_loader, test_loader = create_train_test_loaders(
        "advanced_dataset",
        "demo",
        train_batch_size=8,
        test_batch_size=16,
        train_shuffle=True,
        test_shuffle=False,
        num_workers=0,
        size=160  # 总共160个样本
    )
    
    print(f"  - 训练数据加载器批次数量: {len(train_loader)}")
    print(f"  - 测试数据加载器批次数量: {len(test_loader)}")
    
    # 尝试从训练加载器中获取数据
    for batch in train_loader:
        print(f"  - 训练批次数据形状: {batch['data'].shape}")
        print(f"  - 训练批次标签形状: {batch['label'].shape}")
        break  # 只打印第一个批次
    
    print("\n数据加载器注册中心演示完成！")

# ------------------------------
# 主函数：运行演示
# ------------------------------

if __name__ == "__main__":
    demonstrate_data_loader_registry()