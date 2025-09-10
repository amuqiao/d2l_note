import torch
from d2l import torch as d2l
import os
import json
import matplotlib.pyplot as plt
from src.models.lenet import LeNet
from src.models.alexnet import AlexNet
from src.models.vgg import VGG
from src.models.nin import NIN
from src.models.googlenet import GoogLeNet
from src.utils.toolkit import Toolkit

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

        