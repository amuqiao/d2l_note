import torch
from torch import nn
from d2l import torch as d2l
import os
import re
import datetime
import time
from typing import Optional, Dict, Any

from src.utils.visualization import VisualizationTool
from src.utils.file_utils import FileUtils
from src.utils.model_registry import ModelRegistry
from src.utils.data_utils import DataLoader
from src.predictor.predictor import Predictor
from src.utils.log_utils import get_logger

# 初始化日志
logger = get_logger(
    name="trainer",
    log_file="logs/trainer.log",
    global_level="DEBUG",
    console_level="INFO",
    file_level="DEBUG"
)

class Trainer:
    """优化后的模型训练器（集成所有训练相关功能）"""

    def __init__(self, net=None, device=None, save_every_epoch=False):
        """初始化训练器
        
        Args:
            net: 神经网络模型
            device: 训练设备
            save_every_epoch: 是否每轮都保存模型
        """
        # 设置中文字体
        VisualizationTool.setup_font()
        # 初始化设备
        self.device = device if device else d2l.try_gpu()  # 自动检测GPU
        self.net = net
        if net:
            self.net.to(self.device)
        self.run_dir = None  # 训练目录（训练开始时初始化）
        self.best_test_acc = 0.0  # 最佳测试准确率
        self.total_samples = 0  # 累计处理样本数
        self.save_every_epoch = save_every_epoch  # 是否每轮都保存模型
        if net:
            self._init_weights()  # 初始化权重

    def get_model_config(self, model_type: str, **kwargs) -> Dict[str, Any]:
        """
        获取模型的配置参数
        
        Args:
            model_type: 模型类型名称
            **kwargs: 可选的自定义配置参数
        
        Returns:
            模型配置字典
        """
        try:
            # 直接从模型注册中心获取配置
            default_config = ModelRegistry.get_config(model_type)
            
            # 创建配置字典，确保不会有None值传递给关键参数
            config = {
                "num_epochs": kwargs.get("num_epochs") if kwargs.get("num_epochs") is not None else default_config["num_epochs"],
                "lr": kwargs.get("lr") if kwargs.get("lr") is not None else default_config["lr"],
                "batch_size": kwargs.get("batch_size") if kwargs.get("batch_size") is not None else default_config["batch_size"],
                "resize": kwargs.get("resize") if kwargs.get("resize") is not None else default_config["resize"],
                "input_size": kwargs.get("input_size") if kwargs.get("input_size") is not None else default_config["input_size"],
                "root_dir": kwargs.get("root_dir")
            }
            
            # 确保num_epochs不是None
            if config["num_epochs"] is None:
                config["num_epochs"] = 10  # 默认值
                logger.warning(f"⚠️ 警告: {model_type}的num_epochs为None，使用默认值10")
            
            return config
        except ValueError:
            # 处理配置不存在的情况
            logger.warning(f"⚠️ 模型 '{model_type}' 没有默认配置，使用基础配置")
            return {
                "num_epochs": kwargs.get("num_epochs", 10),
                "lr": kwargs.get("lr", 0.01),
                "batch_size": kwargs.get("batch_size", 128),
                "input_size": kwargs.get("input_size", (1, 1, 28, 28)),
                "resize": kwargs.get("resize"),
                "root_dir": kwargs.get("root_dir")
            }

    def run_training(self, model_type: str, config: Dict[str, Any], 
                    enable_visualization: bool = True, 
                    save_every_epoch: bool = False) -> Dict[str, Any]:
        """
        执行模型训练并返回结果
        
        Args:
            model_type: 模型类型名称
            config: 训练配置参数（包含num_epochs, lr, batch_size, resize等）
            enable_visualization: 是否启用可视化
            save_every_epoch: 是否每轮都保存模型
        
        Returns:
            Dict: 包含训练结果的字典
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 开始训练模型: {model_type}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        # 记录具体的开始训练时间点（日期+时间）
        training_start_time = datetime.datetime.now()
        result = {
            "model_name": model_type,
            "best_accuracy": 0.0,
            "training_time": 0,
            "error": None,
            "config": config,
            "success": False,
            "run_dir": None,
            "training_start_time": training_start_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # 1. 创建模型
            net = ModelRegistry.create_model(model_type)
            logger.info(f"🔧 模型 {model_type} 创建成功")
            
            # 2. 测试网络结构
            test_func = ModelRegistry.get_test_func(model_type)
            test_func(net, input_size=config["input_size"])
            logger.info(f"✅ 模型 {model_type} 网络结构测试通过")
            
            # 3. 加载数据
            logger.info(f"📥 加载数据（batch_size={config['batch_size']}, resize={config['resize']}）")
            train_iter, test_iter = DataLoader.load_data(
                batch_size=config["batch_size"], 
                resize=config["resize"]
            )
            
            # 4. 创建训练器并训练
            self.net = net
            self.device = d2l.try_gpu()
            self.net.to(self.device)
            self.save_every_epoch = save_every_epoch
            self._init_weights()  # 初始化权重
            
            run_dir, best_acc = self.train(
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
            # 记录具体的结束训练时间点（日期+时间）
            training_end_time = datetime.datetime.now()
            result.update({
                "best_accuracy": best_acc,
                "training_time": training_time,
                "run_dir": run_dir,
                "success": True,
                "training_end_time": training_end_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            logger.info(f"🎉 {model_type} 训练完成！最佳准确率: {best_acc:.4f}，耗时: {training_time:.2f}秒")
            logger.info(f"⏱️ 训练开始时间: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"⏱️ 训练结束时间: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        except Exception as e:
            # 异常处理
            training_time = time.time() - start_time
            # 异常情况下也记录结束时间
            training_end_time = datetime.datetime.now()
            error_msg = f"{type(e).__name__}: {str(e)}"
            result.update({
                "training_time": training_time,
                "error": error_msg,
                "success": False,
                "training_end_time": training_end_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            logger.error(f"❌ {model_type} 训练失败！错误: {error_msg}")
            logger.info(f"⏱️ 训练开始时间: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"⏱️ 训练结束时间: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return result

    def train(
        self,
        train_iter,
        test_iter,
        num_epochs,
        lr,
        batch_size,
        enable_visualization=True,
        **kwargs,
    ):
        """
        完整训练流程
        Args:
            train_iter: 训练数据迭代器
            test_iter: 测试数据迭代器
            num_epochs: 训练轮次
            lr: 学习率
            batch_size: 批次大小（用于保存配置）
            enable_visualization: 是否启用实时可视化，默认为True
            **kwargs: 其他参数，如root_dir
        Returns:
            run_dir: 训练目录路径
            best_test_acc: 最佳测试准确率
        """
        # 1. 初始化训练目录和配置
        # 从kwargs中获取root_dir参数，如果没有则使用None（默认为当前工作目录）
        root_dir = kwargs.get('root_dir', None)
        self.run_dir = FileUtils.create_run_dir(root_dir=root_dir)
        train_config = {
            "model_name": self.net.__class__.__name__,
            "device": str(self.device),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "train_samples": len(train_iter) * batch_size,
            "test_samples": len(test_iter) * batch_size,
        }
        FileUtils.save_config(train_config, os.path.join(self.run_dir, "config.json"))

        # 2. 初始化训练组件

        # 初始化epoch指标列表（用于保存每轮训练的详细指标）
        epoch_metrics = []
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        animator = None
        if enable_visualization:
            animator = VisualizationTool.create_animator(
                xlabel="迭代周期", xlim=[1, num_epochs]
            )
        timer = d2l.Timer()
        num_batches = len(train_iter)

        # 3. 开始训练
        logger.info(f"\n🚀 开始训练（设备: {self.device}，轮次: {num_epochs}）")
        # 记录训练开始的具体时间点
        training_start_time = datetime.datetime.now()
        for epoch in range(num_epochs):
            self.net.train()  # 切换到训练模式
            metric = d2l.Accumulator(3)  # 累计：损失、正确数、总数

            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()

                # 前向传播+反向传播
                X, y = X.to(self.device), y.to(self.device)
                self.total_samples += X.shape[0]
                y_hat = self.net(X)
                loss = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()

                # 累计指标
                with torch.no_grad():
                    metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])

                timer.stop()
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]

                # 更新可视化（每5个批次或最后一个批次）
                if enable_visualization and (
                    (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1
                ):
                    VisualizationTool.update_realtime(
                        animator,
                        epoch + (i + 1) / num_batches,
                        (train_loss, train_acc, None),
                    )

            # 4. 每轮次评估+保存模型
            test_acc = self.evaluate_accuracy(test_iter)
            if enable_visualization:
                VisualizationTool.update_realtime(
                    animator, epoch + 1, (None, None, test_acc)
                )
            self._save_best_model(epoch, test_acc)

            # 根据参数决定是否保存每轮模型
            if self.save_every_epoch:
                self._save_epoch_model(epoch, test_acc)

            # 保存当前轮次的指标
            epoch_metric = {
                "epoch": epoch + 1,  # 1-based epoch
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "best_test_acc": self.best_test_acc,
                "epoch_time": timer.sum(),  # 累计到当前轮的时间
            }
            epoch_metrics.append(epoch_metric)

            # 每轮次保存epoch指标到临时文件（防止训练中断数据丢失）
            epoch_metrics_path = os.path.join(self.run_dir, "epoch_metrics.json")
            FileUtils.save_metrics(epoch_metrics, epoch_metrics_path)

        # 5. 训练结束：保存最终指标（包含完整的epoch指标历史）
        total_time = timer.sum()
        # 记录训练结束的具体时间点
        training_end_time = datetime.datetime.now()
        final_metrics = {
            "final_train_loss": train_loss,
            "final_train_acc": train_acc,
            "final_test_acc": test_acc,
            "best_test_acc": self.best_test_acc,
            "total_training_time": f"{total_time:.2f}s",
            "samples_per_second": f"{self.total_samples / total_time:.1f}",
            "epoch_metrics": epoch_metrics,  # 包含完整的每轮指标历史
            "training_start_time": training_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_end_time": training_end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        FileUtils.save_metrics(final_metrics, os.path.join(self.run_dir, "metrics.json"))

        # 6. 输出训练总结
        logger.info("\n" + "=" * 80)
        logger.info(f"📝 训练总结（目录: {self.run_dir}）")
        logger.info(
            f"loss: {train_loss:.4f} | 训练acc: {train_acc:.4f} | 测试acc: {test_acc:.4f}")
        logger.info(
            f"最佳acc: {self.best_test_acc:.4f} | 总时间: {total_time:.2f}s | 速度: {self.total_samples/total_time:.1f}样本/秒")
        logger.info("=" * 80)

        return self.run_dir, self.best_test_acc

    def run_post_training_prediction(self, run_dir: str, n: int = 8, num_samples: int = 10) -> None:
        """
        训练后自动进行预测可视化
        
        Args:
            run_dir: 训练目录路径
            n: 可视化样本数
            num_samples: 随机测试样本数
        """
        logger.info(f"\n🎉 训练完成，开始预测可视化（目录: {run_dir}）")
        predictor = Predictor.from_run_dir(run_dir)
        
        # 根据模型类型确定resize参数
        resize = None  # 默认LeNet不需要resize
        if predictor.config and "model_name" in predictor.config:
            if predictor.config["model_name"] == "AlexNet":
                resize = 224  # AlexNet需要224x224输入
            elif predictor.config["model_name"] == "VGG":
                resize = 224  # VGG需要224x224输入
            elif predictor.config["model_name"] == "GoogLeNet":
                resize = 96   # GoogLeNet需要96x96输入
        
        # 重新加载测试数据（使用正确的resize参数）
        _, test_iter_pred = DataLoader.load_data(batch_size=256, resize=resize)
        
        # 执行预测可视化
        predictor.visualize_prediction(test_iter_pred, n=n)
        predictor.test_random_input(num_samples=num_samples)

    def _init_weights(self):
        """Xavier均匀分布初始化权重（适配Linear/Conv2d）"""

        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

        self.net.apply(init_func)

    def evaluate_accuracy(self, data_iter):
        """评估模型在数据集上的准确率"""
        self.net.eval()  # 切换到评估模式
        metric = d2l.Accumulator(2)  # 累计：正确数、总数

        with torch.no_grad():
            for X, y in data_iter:
                X, y = X.to(self.device), y.to(self.device)
                metric.add(d2l.accuracy(self.net(X), y), y.numel())

        return metric[0] / metric[1]

    def _save_best_model(self, epoch, current_test_acc):
        """保存最佳模型（文件名含准确率和epoch）"""
        if current_test_acc <= self.best_test_acc:
            return None  # 未超过最佳，不保存

        # 更新最佳准确率
        self.best_test_acc = current_test_acc

        # 删除之前的最佳模型文件（确保最终只有一个最佳模型）
        best_model_pattern = f"best_model_{self.net.__class__.__name__}_*.pth"
        for filename in os.listdir(self.run_dir):
            if re.match(best_model_pattern.replace("*", ".*"), filename):
                old_model_path = os.path.join(self.run_dir, filename)
                try:
                    os.remove(old_model_path)
                    logger.info(f"🗑️ 删除旧的最佳模型: {filename}")
                except Exception as e:
                    logger.warning(f"⚠️ 删除旧模型时出错: {str(e)}")

        # 保存新的最佳模型
        model_filename = (
            f"best_model_{self.net.__class__.__name__}_"
            f"acc_{self.best_test_acc:.4f}_epoch_{epoch+1}.pth"
        )
        model_path = os.path.join(self.run_dir, model_filename)

        # 保存完整状态（含权重、准确率、epoch）
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "best_test_acc": self.best_test_acc,
                "epoch": epoch + 1,  # 1-based epoch
                "device": str(self.device),
            },
            model_path,
        )

        logger.info(f"📌 保存最佳模型: {model_filename}（准确率: {self.best_test_acc:.4f}）")
        return model_path

    def _save_epoch_model(self, epoch, current_test_acc):
        """保存每轮次模型（无论是否最佳）"""
        model_filename = (
            f"epoch_model_{self.net.__class__.__name__}_"
            f"acc_{current_test_acc:.4f}_epoch_{epoch+1}.pth"
        )
        model_path = os.path.join(self.run_dir, model_filename)

        # 保存完整状态
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "test_acc": current_test_acc,
                "epoch": epoch + 1,  # 1-based epoch
                "device": str(self.device),
            },
            model_path,
        )

        logger.info(f"💾 保存轮次模型: {model_filename}（准确率: {current_test_acc:.4f}）")
        return model_path