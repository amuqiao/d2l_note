"""
配置注册示例：展示如何使用配置注册中心和装饰器进行配置的自动注册
"""
import os
import sys
import logging

# 添加项目根目录到Python路径，以便能正确导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 设置日志级别为DEBUG，以便查看注册信息
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入配置注册中心入口
from ..config_registry_entry import (
    register_config,
    register_template,
    register_file,
    get_config,
    get_all_registered_configs,
    is_config_registered,
    initialize_configs,
    merge_configs
)

# ------------------------------
# 示例1：使用装饰器注册配置类
# ------------------------------

@register_config(name="basic_training_config", namespace="demo")
class BasicTrainingConfig:
    """基本训练配置类"""
    def __init__(self):
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 0.001

    def to_dict(self):
        """转换为字典格式"""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
        }

# ------------------------------
# 示例2：使用装饰器注册配置函数并提供模板
# ------------------------------

@register_config(
    name="advanced_training_config", 
    namespace="demo",
    template={
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "scheduler": {
            "type": "cosine",
            "warmup_epochs": 5
        }
    }
)
def create_advanced_config():
    """创建高级训练配置的函数"""
    # 这个函数可以根据需要动态生成配置
    return {
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "scheduler": {
            "type": "cosine",
            "warmup_epochs": 5
        }
    }

# ------------------------------
# 示例3：使用register_template注册配置模板
# ------------------------------

def register_config_templates():
    """注册配置模板"""
    # 注册模型配置模板
    register_template(
        "cnn_model_config",
        {
            "name": "CNN",
            "input_channels": 3,
            "hidden_channels": [64, 128, 256],
            "kernel_size": 3,
            "dropout": 0.5,
            "output_classes": 10
        },
        namespace="demo"
    )
    
    # 注册数据配置模板
    register_template(
        "data_config",
        {
            "dataset_name": "CIFAR10",
            "data_dir": "./data",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "augmentation": {
                "random_crop": 32,
                "horizontal_flip": True,
                "normalize": {
                    "mean": [0.4914, 0.4822, 0.4465],
                    "std": [0.2023, 0.1994, 0.2010]
                }
            }
        },
        namespace="demo"
    )

# ------------------------------
# 示例4：使用register_file注册配置文件
# ------------------------------

def register_config_files():
    """注册配置文件"""
    # 注意：这里只是示例，实际使用时需要确保文件存在
    # 假设有一个配置文件位于项目目录下
    example_config_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs", "example_config.json"
    )
    
    # 检查文件是否存在，如果存在则注册
    if os.path.exists(example_config_file):
        register_file(
            "example_config",
            example_config_file,
            namespace="demo"
        )
    else:
        logging.warning(f"示例配置文件不存在: {example_config_file}")
        # 为了演示，我们可以创建一个临时的配置文件
        temp_config = """{
    "experiment_name": "demo_experiment",
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints",
    "early_stopping": {
        "patience": 10,
        "monitor": "val_loss"
    }
}"""
        
        # 确保configs目录存在
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs"
        )
        os.makedirs(config_dir, exist_ok=True)
        
        # 写入临时配置文件
        with open(example_config_file, "w") as f:
            f.write(temp_config)
        
        logging.info(f"已创建临时配置文件: {example_config_file}")
        
        # 注册临时配置文件
        register_file(
            "example_config",
            example_config_file,
            namespace="demo"
        )

# ------------------------------
# 演示函数：展示配置注册和使用的完整流程
# ------------------------------

def demonstrate_config_registry():
    """演示配置注册和使用"""
    print("=" * 50)
    print("配置注册中心演示")
    print("=" * 50)
    
    # 1. 注册配置模板和文件
    print("\n1. 注册配置模板和文件...")
    register_config_templates()
    register_config_files()
    
    # 2. 初始化配置系统
    print("\n2. 初始化配置系统...")
    initialize_configs()
    
    # 3. 检查配置是否注册成功
    print("\n3. 检查配置注册状态:")
    configs_to_check = [
        ("basic_training_config", "demo"),
        ("advanced_training_config", "demo"),
        ("cnn_model_config", "demo"),
        ("data_config", "demo"),
        ("example_config", "demo")
    ]
    
    for config_name, namespace in configs_to_check:
        is_registered = is_config_registered(config_name, namespace)
        print(f"  - 配置 '{config_name}' (命名空间: '{namespace}') 是否已注册: {is_registered}")
    
    # 4. 获取所有注册的配置
    print("\n4. 获取所有注册的配置:")
    all_configs = get_all_registered_configs(namespace="demo")
    for ns, configs in all_configs.items():
        print(f"  - 命名空间 '{ns}' 中的配置: {', '.join(configs)}")
    
    # 5. 获取并使用配置
    print("\n5. 获取并使用配置:")
    
    # 获取基本训练配置
    basic_config = get_config("basic_training_config", "demo")
    print(f"  - 基本训练配置: {basic_config}")
    
    # 获取高级训练配置并覆盖部分参数
    advanced_config = get_config(
        "advanced_training_config", 
        "demo",
        learning_rate=0.0005,
        optimizer="sgd"
    )
    print(f"  - 修改后的高级训练配置: {advanced_config}")
    
    # 获取CNN模型配置
    cnn_config = get_config("cnn_model_config", "demo")
    print(f"  - CNN模型配置: {cnn_config}")
    
    # 6. 合并多个配置
    print("\n6. 合并多个配置:")
    merged_config = merge_configs(advanced_config, cnn_config)
    print(f"  - 合并后的配置: {merged_config}")
    
    print("\n配置注册中心演示完成！")

# ------------------------------
# 主函数：运行演示
# ------------------------------

if __name__ == "__main__":
    demonstrate_config_registry()