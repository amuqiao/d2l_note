# ========================= 模型默认配置 =========================
# 键：模型名称（与代码中model_type对应）
# 值：该模型的默认参数（输入尺寸、Resize、学习率、批次大小、训练轮次等）
MODEL_DEFAULT_CONFIGS = {
    "LeNet": {
        "input_size": (1, 1, 28, 28),  # (batch, channels, height, width)
        "resize": None,                # Fashion-MNIST原始尺寸28x28，无需Resize
        "lr": 0.8,                     # LeNet适合稍高学习率
        "batch_size": 256,             # 较小输入尺寸可支持更大批次
        "num_epochs": 3,              # 收敛较快，15轮足够
    },
    "AlexNet": {
        "input_size": (1, 1, 224, 224),# AlexNet需要224x224输入
        "resize": 224,                 # 加载数据时Resize到224x224
        "lr": 0.01,                    # 较大模型需较低学习率避免震荡
        "batch_size": 128,             # 224x224输入占用显存较高，批次减小
        "num_epochs": 30,              # 训练较慢，10轮平衡效果与时间
    },
    "VGG": {
        "input_size": (1, 1, 224, 224),# VGG同样需要224x224输入
        "resize": 224,
        "lr": 0.05,                   # 更深模型需更低学习率
        "batch_size": 128,              # VGG参数量大，显存占用更高
        "num_epochs": 10,               # 训练耗时久，8轮兼顾效果
    },
    "NIN": {
        "input_size": (1, 1, 224, 224), # NIN需要224x224输入
        "resize": 224,                  # 加载数据时Resize到224x224
        "lr": 0.1,                      # 参考note.py中的设置
        "batch_size": 128,              # 参考note.py中的设置
        "num_epochs": 10,               # 参考note.py中的设置
    },
    "GoogLeNet": {
        "input_size": (1, 1, 96, 96),   # GoogLeNet需要96x96输入
        "resize": 96,                   # 加载数据时Resize到96x96
        "lr": 0.1,                      # 参考note.py中的设置
        "batch_size": 128,              # 参考note.py中的设置
        "num_epochs": 20,               # 参考note.py中的设置
    }
}