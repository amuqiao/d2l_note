#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""测试优化后的Trainer类是否正常工作"""

import os
import sys
from src.trainer.trainer import Trainer

# 解决OpenMP运行时库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入模型并注册
from src.models.lenet import LeNet, LeNetBatchNorm


def test_trainer():
    """测试优化后的Trainer类"""
    print("开始测试优化后的Trainer类...")
    
    # 创建训练器实例
    trainer = Trainer()
    
    # 测试配置获取功能
    print("\n测试配置获取功能...")
    config = trainer.get_model_config(
        model_type="LeNet",
        num_epochs=2,  # 使用小轮次进行测试
        lr=0.01,
        batch_size=128
    )
    print(f"获取的配置: {config}")
    
    # 测试训练功能（使用小轮次进行快速测试）
    print("\n测试训练功能（使用小轮次）...")
    try:
        result = trainer.run_training(
            model_type="LeNet",
            config=config,
            enable_visualization=False,  # 禁用可视化以加快测试
            save_every_epoch=False
        )
        
        if result["success"]:
            print(f"训练成功！最佳准确率: {result['best_accuracy']:.4f}")
            print(f"训练目录: {result['run_dir']}")
            
            # 测试预测可视化功能
            print("\n测试预测可视化功能...")
            trainer.run_post_training_prediction(
                run_dir=result["run_dir"],
                n=4,  # 使用少量样本进行测试
                num_samples=5
            )
        else:
            print(f"训练失败: {result['error']}")
            return False
    except Exception as e:
        print(f"训练过程出现异常: {str(e)}")
        return False
    
    print("\n测试完成！")
    return True


if __name__ == "__main__":
    success = test_trainer()
    sys.exit(0 if success else 1)