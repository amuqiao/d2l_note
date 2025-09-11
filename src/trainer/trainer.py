import torch
from torch import nn
from d2l import torch as d2l
import os
import re
import datetime
from src.utils.visualization import VisualizationTool
from src.utils.file_utils import FileUtils

# ========================= 模型训练类 =========================
class Trainer:
    """模型训练器（集成目录创建、模型保存、指标记录）"""

    def __init__(self, net, device=None, save_every_epoch=False):
        self.net = net
        self.device = device if device else d2l.try_gpu()  # 自动检测GPU
        self.net.to(self.device)
        self.run_dir = None  # 训练目录（训练开始时初始化）
        self.best_test_acc = 0.0  # 最佳测试准确率
        self.total_samples = 0  # 累计处理样本数
        self.save_every_epoch = save_every_epoch  # 是否每轮都保存模型
        self._init_weights()  # 初始化权重

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
                    print(f"🗑️ 删除旧的最佳模型: {filename}")
                except Exception as e:
                    print(f"⚠️ 删除旧模型时出错: {str(e)}")

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

        print(f"📌 保存最佳模型: {model_filename}（准确率: {self.best_test_acc:.4f}）")
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

        print(f"💾 保存轮次模型: {model_filename}（准确率: {current_test_acc:.4f}）")
        return model_path

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
        print(f"\n🚀 开始训练（设备: {self.device}，轮次: {num_epochs}）")
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
        final_metrics = {
            "final_train_loss": train_loss,
            "final_train_acc": train_acc,
            "final_test_acc": test_acc,
            "best_test_acc": self.best_test_acc,
            "total_training_time": f"{total_time:.2f}s",
            "samples_per_second": f"{self.total_samples / total_time:.1f}",
            "epoch_metrics": epoch_metrics,  # 包含完整的每轮指标历史
        }
        FileUtils.save_metrics(final_metrics, os.path.join(self.run_dir, "metrics.json"))

        # 6. 输出训练总结
        print("\n" + "=" * 80)
        print(f"📝 训练总结（目录: {self.run_dir}）")
        print(
            f"loss: {train_loss:.4f} | 训练acc: {train_acc:.4f} | 测试acc: {test_acc:.4f}"
        )
        print(
            f"最佳acc: {self.best_test_acc:.4f} | 总时间: {total_time:.2f}s | 速度: {self.total_samples/total_time:.1f}样本/秒"
        )
        print("=" * 80)

        return self.run_dir, self.best_test_acc