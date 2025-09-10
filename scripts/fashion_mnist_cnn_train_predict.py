import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l
import sys
import matplotlib.pyplot as plt
import os
import glob
import json
import datetime
import re
import argparse

# 解决OpenMP运行时库冲突问题
# 设置环境变量允许多个OpenMP运行时库共存
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ========================= 通用工具类 =========================
class Toolkit:
    """通用工具类：字体设置、目录创建、配置保存等"""

    @staticmethod
    def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5, figsize=None):
        """使用matplotlib原生方法显示图像网格，支持缩放和自定义figsize

        参数:
            imgs: 图像数据 (batch_size, height, width) 或 (batch_size, channels, height, width)
            num_rows: 行数
            num_cols: 列数
            titles: 图像标题列表，长度应等于图像数量
            scale: 缩放因子，默认为1.5
            figsize: 图表大小元组 (width, height)，如果未指定则根据行数和列数自动计算
        """
        # 调整图像维度
        if len(imgs.shape) == 4 and imgs.shape[1] == 1:  # 单通道图像
            imgs = imgs.squeeze(1)  # 移除通道维度

        # 计算图表大小
        if figsize is None:
            figsize = (num_cols * scale, num_rows * scale)

        # 创建图表和子图
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

        # 显示图像
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            # 处理张量图像
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()

            # 显示图像
            if len(img.shape) == 2:  # 灰度图
                # 使用viridis彩色映射来匹配d2l.show_images的显示效果
                ax.imshow(img, cmap="viridis")
            else:  # 彩色图
                ax.imshow(img)

            # 设置标题
            if titles and i < len(titles):
                ax.set_title(
                    titles[i], fontsize=8 * scale
                )  # 标题字体大小与缩放因子相关

            # 隐藏坐标轴
            ax.axis("off")

        # 调整子图间距
        plt.tight_layout(pad=scale * 0.5)  # 增加间距避免标题重叠

    @staticmethod
    def setup_font():
        """配置matplotlib中文显示"""
        if sys.platform.startswith("win"):
            plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
        elif sys.platform.startswith("darwin"):
            plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC"]
        elif sys.platform.startswith("linux"):
            plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

    @staticmethod
    def create_run_dir(prefix="data/run_", root_dir=None):
        """创建时间戳唯一目录（格式：run_年日月_时分秒）
        
        参数:
            prefix: 目录前缀，默认为"run_"
            root_dir: 根目录路径，默认为执行脚本的目录（当前工作目录）
        
        返回:
            创建的目录路径
        """
        # 如果未指定根目录，使用当前工作目录
        if root_dir is None:
            root_dir = os.getcwd()
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(root_dir, f"{prefix}{timestamp}")
        os.makedirs(run_dir, exist_ok=False)  # 目录不存在则创建，避免覆盖
        print(f"✅ 创建训练目录: {run_dir}")
        return run_dir

    @staticmethod
    def save_config(config_dict, save_path):
        """保存配置到JSON文件（便于追溯训练参数）"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        print(f"📝 配置已保存: {os.path.basename(save_path)}")

    @staticmethod
    def save_metrics(metrics_dict, save_path):
        """保存训练指标到JSON文件（便于性能对比）"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
        print(f"📊 指标已保存: {os.path.basename(save_path)}")

    @staticmethod
    def find_best_model_in_dir(run_dir):
        """在训练目录中查找最佳模型文件（匹配best_model开头的.pth）"""
        model_pattern = os.path.join(run_dir, "best_model*.pth")
        model_files = glob.glob(model_pattern)
        if not model_files:
            raise FileNotFoundError(
                f"目录 {run_dir} 中未找到最佳模型（格式：best_model*.pth）"
            )
        return os.path.basename(model_files[-1])  # 默认取最后一个

    @staticmethod
    def list_models_in_dir(run_dir):
        """列出目录中的所有模型文件及其准确率信息"""
        model_pattern = os.path.join(run_dir, "best_model*.pth")
        model_files = glob.glob(model_pattern)
        if not model_files:
            raise FileNotFoundError(
                f"目录 {run_dir} 中未找到模型文件（格式：best_model*.pth）"
            )

        models_info = []
        for model_path in model_files:
            try:
                # 提取文件名中的准确率和轮次信息
                filename = os.path.basename(model_path)
                # 尝试从文件名提取准确率
                acc_match = re.search(r"acc_([0-9.]+)", filename)
                epoch_match = re.search(r"epoch_([0-9]+)", filename)

                acc = float(acc_match.group(1)) if acc_match else 0.0
                epoch = int(epoch_match.group(1)) if epoch_match else 0

                models_info.append(
                    {
                        "path": model_path,
                        "filename": filename,
                        "accuracy": acc,
                        "epoch": epoch,
                    }
                )
            except Exception:
                # 如果解析失败，仍将文件加入列表
                models_info.append(
                    {
                        "path": model_path,
                        "filename": os.path.basename(model_path),
                        "accuracy": 0.0,
                        "epoch": 0,
                    }
                )

        # 按准确率降序排序
        models_info.sort(key=lambda x: x["accuracy"], reverse=True)
        return models_info

    @staticmethod
    def test_network_shape(net, input_size=(1, 1, 28, 28)):
        """测试网络各层输出形状"""
        X = torch.rand(size=input_size, dtype=torch.float32)
        print("\n🔍 网络结构测试:")

        # 如果网络使用Sequential实现
        if isinstance(net, nn.Sequential):
            for i, layer in enumerate(net):
                X = layer(X)
                print(f"{i+1:2d}. {layer.__class__.__name__:12s} → 输出形状: {X.shape}")
        else:
            # 特征提取层
            print("├─ 特征提取部分:")
            if hasattr(net, "features"):
                for i, layer in enumerate(net.features):
                    X = layer(X)
                    print(
                        f"│  {i+1:2d}. {layer.__class__.__name__:12s} → 输出形状: {X.shape}"
                    )

            # 分类器层
            print("└─ 分类器部分:")
            X = net.features(torch.rand(size=input_size, dtype=torch.float32))
            if hasattr(net, "classifier"):
                for i, layer in enumerate(net.classifier):
                    X = layer(X)
                    print(
                        f"   {i+1:2d}. {layer.__class__.__name__:12s} → 输出形状: {X.shape}"
                    )


# ========================= 模型定义类 =========================
class LeNet(nn.Module):
    """LeNet卷积神经网络（原结构保留，适配后续保存/加载逻辑）"""

    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            # 卷积块1：Conv2d → Sigmoid → AvgPool2d
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 卷积块2：Conv2d → Sigmoid → AvgPool2d
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),  # 10分类（Fashion-MNIST）
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    """AlexNet卷积神经网络"""

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 这里使用一个11*11的更大窗口来捕捉对象
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 使用dropout层来减轻过拟合
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    """VGG卷积神经网络（使用简化版本适配Fashion-MNIST）"""
    
    def __init__(self, ratio=4):
        super(VGG, self).__init__()
        # 定义VGG块构造函数
        def vgg_block(num_convs, in_channels, out_channels):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels,
                                        kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)
        
        # VGG配置 - 标准版本
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        
        # 简化版本：使用比例因子缩小通道数
        small_conv_arch = [(num_convs, out_channels // ratio) 
                          for (num_convs, out_channels) in conv_arch]
        
        # 卷积层部分
        conv_blks = []
        in_channels = 1
        for (num_convs, out_channels) in small_conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        
        # 构建网络
        self.features = nn.Sequential(*conv_blks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 全连接层部分
            nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


class NIN(nn.Module):
    """Network in Network (NIN)卷积神经网络"""
    
    def __init__(self):
        super(NIN, self).__init__()
        # 定义NIN块构造函数
        def nin_block(in_channels, out_channels, kernel_size, strides, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
        
        # 构建网络特征提取层
        self.features = nn.Sequential(
            nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数是10
            nin_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器层：简单的展平操作
        self.classifier = nn.Sequential(
            nn.Flatten()  # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


class GoogLeNet(nn.Module):
    """GoogLeNet卷积神经网络"""
    
    def __init__(self):
        super(GoogLeNet, self).__init__()
        
        # 构建网络特征提取层
        # 第一模块：7×7卷积层
        b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第二模块：1×1卷积层后接3×3卷积层
        b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第三模块：两个Inception块
        b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第四模块：五个Inception块
        b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第五模块：两个Inception块+全局平均池化
        b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 组合所有模块
        self.features = nn.Sequential(b1, b2, b3, b4, b5)
        
        # 分类器层：全连接层
        self.classifier = nn.Sequential(
            nn.Linear(1024, 10)  # 10分类（Fashion-MNIST）
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


class Inception(nn.Module):
    """Inception模块实现"""
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# ========================= 数据加载类 =========================
class DataLoader:
    """数据加载类（支持自定义批次大小和图像resize）"""

    @staticmethod
    def load_data(batch_size=256, resize=None):
        """
        加载Fashion-MNIST数据集
        Args:
            batch_size: 批次大小（默认256）
            resize: 图像resize尺寸（默认None，即28x28）
        Returns:
            train_iter: 训练数据迭代器
            test_iter: 测试数据迭代器
        """
        print(f"📥 加载Fashion-MNIST（batch_size={batch_size}, resize={resize}）")
        return d2l.load_data_fashion_mnist(batch_size=batch_size, resize=resize)


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
        self.run_dir = Toolkit.create_run_dir(root_dir=root_dir)
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
        Toolkit.save_config(train_config, os.path.join(self.run_dir, "config.json"))

        # 2. 初始化训练组件

        # 初始化epoch指标列表（用于保存每轮训练的详细指标）
        epoch_metrics = []
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        animator = None
        if enable_visualization:
            animator = AnimatorTool.create_animator(
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
                    AnimatorTool.update_realtime(
                        animator,
                        epoch + (i + 1) / num_batches,
                        (train_loss, train_acc, None),
                    )

            # 4. 每轮次评估+保存模型
            test_acc = self.evaluate_accuracy(test_iter)
            if enable_visualization:
                AnimatorTool.update_realtime(
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
            Toolkit.save_metrics(epoch_metrics, epoch_metrics_path)

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
        Toolkit.save_metrics(final_metrics, os.path.join(self.run_dir, "metrics.json"))

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


# ========================= 模型预测类 =========================
class Predictor:
    """模型预测类（支持从训练目录加载模型，可视化预测结果）"""

    def __init__(self, net, device=None):
        self.net = net
        self.device = device if device else d2l.try_gpu()
        self.net.to(self.device)
        self.net.eval()  # 初始化即切换到评估模式
        self.config = None  # 加载的训练配置
        self.best_acc = None  # 模型最佳准确率

    @classmethod
    def from_run_dir(cls, run_dir, device=None, model_file=None):
        """
        从训练目录创建Predictor（可选择加载特定模型文件）
        Args:
            run_dir: 训练目录路径
            device: 运行设备（默认自动检测GPU）
            model_file: 可选，指定的模型文件名
        Returns:
            Predictor实例
        """
        # 1. 验证目录和文件
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"训练目录不存在: {run_dir}")

        config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件缺失: {config_path}")

        # 2. 确定模型文件路径
        if model_file:
            # 使用指定的模型文件
            model_path = os.path.join(run_dir, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"指定的模型文件不存在: {model_path}")
        else:
            # 自动选择最佳模型文件
            model_file = Toolkit.find_best_model_in_dir(run_dir)
            model_path = os.path.join(run_dir, model_file)
            
        # 3. 复用from_model_path方法加载模型和配置
        # 注意：这里传入了明确的config_path，确保配置能被正确加载
        return cls.from_model_path(model_path, config_path=config_path, device=device)

    @classmethod
    def from_model_path(cls, model_path, config_path=None, device=None):
        """
        从指定的模型文件路径创建Predictor
        Args:
            model_path: 模型文件路径（支持相对路径或绝对路径）
            config_path: 可选，配置文件路径
            device: 运行设备（默认自动检测GPU）
        Returns:
            Predictor实例
        """
        # 1. 验证并规范化模型文件路径
        if not model_path:
            raise ValueError("模型文件路径不能为空")
            
        # 确保路径规范化（处理相对路径和绝对路径）
        model_path = os.path.abspath(os.path.expanduser(model_path))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        if not model_path.endswith('.pth'):
            raise ValueError(f"无效的模型文件格式: {model_path}，应为.pth文件")

        # 2. 加载模型（添加版本兼容性处理）
        try:
            # 检查PyTorch版本是否支持weights_only参数
            try:
                # 尝试使用weights_only参数
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                # 如果报错（不支持该参数），则不使用weights_only
                checkpoint = torch.load(model_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"加载模型文件失败: {str(e)}") from e

        # 验证checkpoint内容
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"模型文件格式错误，缺少'model_state_dict'键: {model_path}")

        # 3. 尝试自动确定配置文件路径
        if not config_path:
            # 假设配置文件在同一目录
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, "config.json")

        # 4. 加载配置（如果有）
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                print(f"✅ 加载配置: {os.path.basename(config_path)}")
                model_name = config.get("model_name", "LeNet")  # 默认LeNet
            except Exception as e:
                print(f"⚠️ 配置文件解析失败，使用默认设置: {str(e)}")
                model_name = "LeNet"
                config["model_name"] = model_name
        else:
            print("⚠️ 未找到配置文件，使用默认模型类型")
            model_name = "LeNet"
            # 即使没有配置文件，也要设置model_name
            config["model_name"] = model_name

        # 5. 创建模型并加载权重
        try:
            if model_name == "LeNet":
                net = LeNet()
            elif model_name == "AlexNet":
                net = AlexNet()
            elif model_name == "VGG":
                net = VGG()
            elif model_name == "NIN":
                net = NIN()
            elif model_name == "GoogLeNet":
                net = GoogLeNet()
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")

            net.load_state_dict(checkpoint["model_state_dict"])
            best_acc = checkpoint.get("best_test_acc", 0.0)
            print(f"✅ 加载模型: {os.path.basename(model_path)}（准确率: {best_acc:.4f}）")
        except Exception as e:
            raise RuntimeError(f"创建模型或加载权重失败: {str(e)}") from e

        # 6. 返回Predictor实例
        predictor = cls(net, device)
        predictor.config = config
        predictor.best_acc = best_acc
        return predictor

    def predict(self, X):
        """基础预测：返回预测类别（1D张量）"""
        with torch.no_grad():
            X = X.to(self.device)
            return torch.argmax(self.net(X), dim=1)

    def visualize_prediction(self, test_iter, n=8):
        """可视化预测结果（正确绿色/错误红色标记）"""
        print(f"\n📊 可视化 {n} 个测试样本预测结果（设备: {self.device}）")

        # 获取测试样本
        X, y = next(iter(test_iter))
        X, y = X[:n], y[:n]
        y_hat = self.predict(X)

        # 转换标签为文本
        true_labels = d2l.get_fashion_mnist_labels(y)
        pred_labels = d2l.get_fashion_mnist_labels(y_hat.cpu())

        # 生成简洁的标题（避免重叠）
        titles = []
        for t, p, y_true, y_pred in zip(true_labels, pred_labels, y, y_hat.cpu()):
            status = "对" if y_true == y_pred else "错"
            titles.append(f"{status}\n真实:{t}\n预测:{p}")

        # 确定图像尺寸 - 根据模型类型自动调整
        image_size = 28  # 默认LeNet的28×28
        if self.config and "model_name" in self.config:
            if self.config["model_name"] == "AlexNet" or self.config["model_name"] == "VGG":
                image_size = 224

        # 重塑图像并显示
        X_reshaped = X.reshape((n, image_size, image_size))

        # 使用增强版的show_images方法显示图像
        Toolkit.show_images(
            X_reshaped,
            num_rows=1,
            num_cols=n,
            titles=titles,
            scale=1.8,  # 增加缩放因子以提供更多空间显示标签
            figsize=(n * 1.5, 3),  # 增加图表高度，确保标题完全显示
        )
        plt.show()

        # 输出预测详情
        correct = torch.sum(y == y_hat.cpu()).item()
        print(f"\n📋 预测详情:")
        print(f"真实标签: {y.tolist()} → {true_labels}")
        print(f"预测标签: {y_hat.tolist()} → {pred_labels}")
        print(f"预测正确率: {correct / n:.2%}（{correct}/{n}）")

    def test_random_input(self, num_samples=10):
        """测试随机输入（验证模型是否正常工作）"""
        # 确定输入尺寸 - 根据模型类型自动调整
        input_size = (1, 28, 28)  # 默认LeNet的输入尺寸
        if self.config and "model_name" in self.config:
            if self.config["model_name"] == "AlexNet" or self.config["model_name"] == "VGG":
                input_size = (1, 224, 224)

        print(
            f"\n🔍 测试 {num_samples} 个随机输入（{input_size[1]}x{input_size[2]}灰度图）"
        )
        random_X = torch.randn(num_samples, *input_size)  # 模拟随机图像
        random_preds = self.predict(random_X)
        random_labels = d2l.get_fashion_mnist_labels(random_preds.cpu())

        print(f"预测类别: {random_preds.tolist()}")
        print(f"预测标签: {random_labels}")

        # 检查预测多样性（避免模型输出单一类别）
        unique_preds = torch.unique(random_preds).numel()
        if unique_preds < 3:
            print(f"⚠️ 警告: 随机预测类别较少（{unique_preds}种），模型可能未充分训练")
        else:
            print(f"✅ 随机预测类别多样（{unique_preds}种），模型状态正常")


# ========================= 动画工具类 =========================
class AnimatorTool:
    """动画工具类：仅保留train模式需要的实时可视化功能"""

    @staticmethod
    def create_animator(xlabel="迭代周期", xlim=None, legend=None):
        """创建基础动画器

        参数:
            xlabel: x轴标签
            xlim: x轴范围 [min, max]
            legend: 图例列表

        返回:
            d2l.Animator实例
        """
        if legend is None:
            legend = ["训练损失", "训练准确率", "测试准确率"]
        if xlim is None:
            xlim = [1, 1]  # 默认值，后续可调整
        return d2l.Animator(xlabel=xlabel, xlim=xlim, legend=legend)

    @staticmethod
    def update_realtime(animator, x_value, metrics):
        """实时模式：在训练过程中更新动画

        参数:
            animator: d2l.Animator实例
            x_value: x轴坐标值
            metrics: 指标元组 (训练损失, 训练准确率, 测试准确率)
        """
        animator.add(x_value, metrics)


# ========================= 主函数入口 =========================
def main(mode="train", run_dir=None, model_file=None, **kwargs):
    """
    主函数：支持训练、预测、结果汇总三种模式
    Args:
        mode: 运行模式（train/predict/summarize）
        run_dir: 训练目录（predict模式需指定，train模式自动生成）
        model_file: 可选，指定的模型文件名
        kwargs: 模式参数（如train的num_epochs、lr等）
    """
    # 初始化字体
    Toolkit.setup_font()

    if mode == "train":
        # 训练模式：直接使用传入的参数（call_args已处理默认值）
        model_type = kwargs.get("model_type")  # 默认使用VGG
        num_epochs = kwargs.get("num_epochs")
        lr = kwargs.get("lr")
        batch_size = kwargs.get("batch_size")
        input_size = kwargs.get("input_size")
        resize = kwargs.get("resize")

        # 1. 创建模型
        if model_type == "LeNet":
            net = LeNet()
        elif model_type == "AlexNet":
            net = AlexNet()
        elif model_type == "VGG":
            net = VGG()
        elif model_type == "NIN":
            net = NIN()
        elif model_type == "GoogLeNet":
            net = GoogLeNet()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 测试网络结构
        Toolkit.test_network_shape(net, input_size=input_size)

        # 2. 加载数据
        print(f"📥 加载Fashion-MNIST（batch_size={batch_size}, resize={resize}）")
        train_iter, test_iter = d2l.load_data_fashion_mnist(
            batch_size=batch_size, resize=resize
        )

        # 3. 训练模型
        save_every_epoch = kwargs.get("save_every_epoch", False)
        trainer = Trainer(net, save_every_epoch=save_every_epoch)
        # 训练模型（启用实时可视化）
        print("🎨 启用实时可视化训练过程...")
        enable_visualization = kwargs.get("enable_visualization", True)
        run_dir, best_acc = trainer.train(
            train_iter=train_iter,
            test_iter=test_iter,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            enable_visualization=enable_visualization,  # 启用实时可视化
        )

        # 4. 训练后自动预测可视化
        print(f"\n🎉 训练完成，开始预测可视化（目录: {run_dir}）")
        predictor = Predictor.from_run_dir(run_dir)

        # 根据模型类型确定resize参数
        resize = None  # 默认LeNet不需要resize
        if predictor.config and "model_name" in predictor.config:
            if predictor.config["model_name"] == "AlexNet":
                resize = 224  # AlexNet需要224x224输入

        # 重新加载测试数据（使用正确的resize参数）
        _, test_iter_pred = DataLoader.load_data(batch_size=256, resize=resize)

        # 执行预测可视化
        predictor.visualize_prediction(test_iter_pred, n=8)
        predictor.test_random_input(num_samples=10)

    elif mode == "predict":
        # 预测模式：支持多种加载方式
        # 方式1：从模型文件直接加载
        # 方式2：从训练目录加载（自动加载最佳模型）

        # 方式3：两者都不指定时，自动选择最新训练目录
        
        if not run_dir and not kwargs.get("model_path"):
            # 自动选择最新训练目录
            print("⚠️ 未指定训练目录或模型路径，自动查找最新目录...")
            run_dirs = sorted(
                [
                    d
                    for d in os.listdir(".")
                    if os.path.isdir(d) and d.startswith("run_")
                ],
                key=lambda x: os.path.getmtime(x),
                reverse=True,
            )
            if not run_dirs:
                raise FileNotFoundError("未找到任何训练目录（需以'run_'开头）")
            run_dir = run_dirs[0]
            print(f"✅ 自动选择最新训练目录: {run_dir}")

        # 1. 创建预测器并确定resize参数
        if kwargs.get("model_path"):
            # 直接从模型文件路径加载
            print(f"🔍 模式：从模型文件直接加载")
            predictor = Predictor.from_model_path(kwargs["model_path"])
        else:
            # 从训练目录加载（可选择模型文件）
            print(f"🔍 模式：从训练目录加载{', 自动选择最佳模型' if not model_file else f', 指定模型文件: {model_file}'}")
            predictor = Predictor.from_run_dir(run_dir, model_file=model_file)

            # 如果指定了目录但未指定模型文件，显示该目录下的所有模型信息
            if not model_file:
                try:
                    models_info = Toolkit.list_models_in_dir(run_dir)
                    print(f"\n📋 {run_dir} 目录中的模型列表（按准确率排序）:")
                    print(f"{'序号':<4} {'文件名':<60} {'准确率':<10} {'轮次':<6}")
                    print("-" * 80)
                    for i, model_info in enumerate(models_info, 1):
                        # 标记当前加载的最佳模型
                        mark = "⭐" if i == 1 else " "
                        print(
                            f"{i:<4} {model_info['filename']:<60} {model_info['accuracy']:.4f}    {model_info['epoch']:<6} {mark}")

                    # 提示用户可以通过model_file参数指定具体模型
                    if len(models_info) > 1:
                        print(f"\n💡 提示：使用 model_file 参数可以加载特定模型，例如:")
                        print(
                            f"   main(mode='predict', run_dir='{run_dir}', model_file='{models_info[1]['filename']}')")
                except Exception as e:
                    print(f"⚠️ 列出模型文件时出错: {str(e)}")

        # 2. 根据模型类型确定resize参数
        resize = None  # 默认LeNet不需要resize
        if predictor.config and "model_name" in predictor.config:
            if predictor.config["model_name"] == "AlexNet":
                resize = 224  # AlexNet需要224x224输入

        # 3. 加载数据（使用正确的resize参数）
        _, test_iter = DataLoader.load_data(
            batch_size=kwargs.get("batch_size", 256), resize=resize
        )

        # 4. 执行预测可视化
        predictor.visualize_prediction(test_iter, n=kwargs.get("n", 8))
        predictor.test_random_input(num_samples=10)

    else:
        raise ValueError(f"不支持的模式: {mode}，请选择'train'/'predict'")


# ========================= 运行入口 =========================
import argparse

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

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="深度学习模型训练与预测工具")
    
    # 基础参数
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='运行模式: train（训练）或 predict（预测）')
    
    # 训练模式参数
    parser.add_argument('--model_type', type=str, default='LeNet', choices=['LeNet', 'AlexNet', 'VGG', 'NIN', 'GoogLeNet'],
                        help='模型类型: LeNet、AlexNet或VGG')

    parser.add_argument('--num_epochs', type=int, default=None,
                        help='训练轮次（AlexNet建议10轮）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（AlexNet建议0.01）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--input_size', type=int, nargs=4, default=None,
                        help='输入尺寸，格式为：批量大小 通道数 高度 宽度（如：1 1 224 224）')
    parser.add_argument('--resize', type=int, default=None,
                        help='图像调整大小（LeNet不需要调整，默认为None）')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='是否每轮都保存模型文件（默认仅保存最佳模型）')
    parser.add_argument('--disable_visualization', action='store_true',
                        help='禁用实时可视化训练过程（默认启用）')
    
    parser.add_argument('--root_dir', type=str, default=None,
                        help='训练目录的根目录路径，默认为执行脚本的目录（当前工作目录）')
    
    # 预测模式参数
    parser.add_argument('--run_dir', type=str, default=None,
                        help='训练目录路径（推荐方式）')
    parser.add_argument('--model_file', type=str, default=None,
                        help='可选，指定要加载的模型文件名，不指定则自动加载该目录下的最佳模型')
    parser.add_argument('--model_path', type=str, default=None,
                        help='完整模型文件路径（最高优先级，设置后会忽略run_dir和model_file）')
    parser.add_argument('--n', type=int, default=8,
                        help='可视化样本数')
    
    # 解析参数
    args = parser.parse_args()
    
    # 准备调用参数
    call_args = {'mode': args.mode}
    
    # 根据模式添加特定参数
    if args.mode == 'train':
        # 获取当前模型类型的默认配置
        default_config = MODEL_DEFAULT_CONFIGS[args.model_type]
        
        # 当参数为None时，使用默认配置的值
        call_args.update({
            'num_epochs': args.num_epochs if args.num_epochs is not None else default_config["num_epochs"],
            'lr': args.lr if args.lr is not None else default_config["lr"],
            'batch_size': args.batch_size if args.batch_size is not None else default_config["batch_size"],
            'input_size': args.input_size if args.input_size is not None else default_config["input_size"],
            'resize': args.resize if args.resize is not None else default_config["resize"],
            'model_type': args.model_type,
            'save_every_epoch': args.save_every_epoch,
            'enable_visualization': not args.disable_visualization,
            'n': args.n,
            'root_dir': args.root_dir
        })
    else:  # predict模式
        call_args.update({
            'run_dir': args.run_dir,
            'model_file': args.model_file,
            'model_path': args.model_path,
            'n': args.n,
            'batch_size': args.batch_size
        })
    
    # 启动主函数
    main(**call_args)
