import time
from src.utils.model_registry import ModelRegistry
from src.trainer.trainer import Trainer
from src.utils.data_utils import DataLoader


def train_model_common(model_name, config, enable_visualization=False, save_every_epoch=False):
    """
    统一的模型训练方法，可被run.py和batch_train.py共享
    
    Args:
        model_name: 模型名称
        config: 训练配置参数（包含num_epochs, lr, batch_size, resize等）
        enable_visualization: 是否启用可视化
        save_every_epoch: 是否每轮都保存模型
        
    Returns:
        Dict: 包含训练结果的字典
    """
    print(f"\n{'='*60}")
    print(f"🚀 开始训练模型: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = {
        "model_name": model_name,
        "best_accuracy": 0.0,
        "training_time": 0,
        "error": None,
        "config": config,
        "success": False,
        "run_dir": None
    }
    
    try:
        # 1. 创建模型
        net = ModelRegistry.create_model(model_name)
        print(f"🔧 模型 {model_name} 创建成功")
        
        # 2. 测试网络结构
        test_func = ModelRegistry.get_test_func(model_name)
        test_func(net, input_size=config["input_size"])
        print(f"✅ 模型 {model_name} 网络结构测试通过")
        
        # 3. 加载数据
        print(f"📥 加载数据（batch_size={config['batch_size']}, resize={config['resize']}）")
        train_iter, test_iter = DataLoader.load_data(
            batch_size=config["batch_size"], 
            resize=config["resize"]
        )
        
        # 4. 创建训练器并训练
        trainer = Trainer(net, save_every_epoch=save_every_epoch)
        run_dir, best_acc = trainer.train(
            train_iter=train_iter,
            test_iter=test_iter,
            num_epochs=config["num_epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            enable_visualization=enable_visualization,
            root_dir=config.get("root_dir")
        )
        
        # 5. 记录结果
        training_time = time.time() - start_time
        result.update({
            "best_accuracy": best_acc,
            "training_time": training_time,
            "run_dir": run_dir,
            "success": True
        })
        
        print(f"🎉 {model_name} 训练完成！最佳准确率: {best_acc:.4f}，耗时: {training_time:.2f}秒")
        
    except Exception as e:
        # 异常处理
        training_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        result.update({
            "training_time": training_time,
            "error": error_msg,
            "success": False
        })
        
        print(f"❌ {model_name} 训练失败！错误: {error_msg}")
    
    return result