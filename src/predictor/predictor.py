import torch
from d2l import torch as d2l
import os
import sys
import json
import matplotlib.pyplot as plt
import glob
from typing import Optional, Dict, Any, List
from src.models.lenet import LeNet, LeNetBatchNorm
from src.models.alexnet import AlexNet
from src.models.vgg import VGG
from src.models.nin import NIN
from src.models.googlenet import GoogLeNet
from src.models.resnet import ResNet
from src.models.dense_net import DenseNet
from src.models.mlp import MLP
from src.utils.file_utils import FileUtils
from src.utils.visualization import VisualizationTool
from src.utils.data_utils import DataLoader

# ========================= 增强版模型预测类 =========================
class Predictor:
    """增强版模型预测类：整合了预测核心功能和高级工具功能"""

    def __init__(self, net, device=None):
        self.net = net
        self.device = device if device else d2l.try_gpu()
        self.net.to(self.device)
        self.net.eval()  # 初始化即切换到评估模式
        self.config = None  # 加载的训练配置
        self.best_acc = None  # 模型最佳准确率
        self.run_dir = None  # 训练目录路径
        self.model_path = None  # 模型文件路径
        
        # 设置中文字体
        VisualizationTool.setup_font()

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
            model_file = FileUtils.find_best_model_in_dir(run_dir)
            model_path = os.path.join(run_dir, model_file)
            
        # 3. 复用from_model_path方法加载模型和配置
        # 注意：这里传入了明确的config_path，确保配置能被正确加载
        predictor = cls.from_model_path(model_path, config_path=config_path, device=device)
        predictor.run_dir = run_dir
        return predictor

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
            elif model_name == "LeNetBatchNorm":
                net = LeNetBatchNorm()
            elif model_name == "AlexNet":
                net = AlexNet()
            elif model_name == "VGG":
                net = VGG()
            elif model_name == "NIN":
                net = NIN()
            elif model_name == "GoogLeNet":
                net = GoogLeNet()
            elif model_name == "ResNet":
                net = ResNet()
            elif model_name == "DenseNet":
                net = DenseNet()
            elif model_name == "MLP":
                net = MLP()
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
        predictor.model_path = model_path
        return predictor
    
    @classmethod
    def create_from_args(cls, run_dir=None, model_file=None, model_path=None, root_dir=None, device=None):
        """
        从参数创建Predictor实例，支持多种创建方式
        Args:
            run_dir: 训练目录路径
            model_file: 模型文件名
            model_path: 完整模型文件路径
            root_dir: 根目录路径，用于自动查找训练目录
            device: 运行设备
        Returns:
            Predictor实例
        """
        if model_path:
            # 直接从模型文件路径加载
            print(f"🔍 模式：从模型文件直接加载")
            return cls.from_model_path(model_path, device=device)
        else:
            # 如果未指定训练目录，自动选择最新目录
            if not run_dir:
                run_dir = cls.find_latest_run_dir(root_dir)
            
            # 从训练目录加载（可选择模型文件）
            print(f"🔍 模式：从训练目录加载{', 自动选择最佳模型' if not model_file else f', 指定模型文件: {model_file}'}")
            return cls.from_run_dir(run_dir, model_file=model_file, device=device)

    @staticmethod
    def find_latest_run_dir(root_dir: Optional[str] = None) -> str:
        """
        自动查找最新的训练目录
        Args:
            root_dir: 根目录路径
        Returns:
            最新训练目录的路径
        """
        if root_dir is None:
            root_dir = os.getcwd()
        
        # 定义要搜索的目录列表（优先级顺序）
        search_dirs = [
            root_dir,  # 首先在root_dir目录下查找
            os.path.join(root_dir, "data")  # 然后在root_dir/data目录下查找
        ]
        
        # 存储所有找到的run_开头的目录
        all_run_dirs = []
        
        # 遍历搜索目录
        for search_dir in search_dirs:
            try:
                if os.path.exists(search_dir) and os.path.isdir(search_dir):
                    run_dirs_in_search = [
                        os.path.join(search_dir, d)
                        for d in os.listdir(search_dir) 
                        if os.path.isdir(os.path.join(search_dir, d)) and d.startswith("run_")
                    ]
                    all_run_dirs.extend(run_dirs_in_search)
            except Exception as e:
                print(f"⚠️ 搜索目录 {search_dir} 时出错: {str(e)}")
        
        # 按修改时间排序，选择最新的目录
        if all_run_dirs:
            all_run_dirs.sort(key=os.path.getmtime, reverse=True)
            run_dir = all_run_dirs[0]
            print(f"✅ 自动选择最新训练目录: {run_dir}")
            return run_dir
        else:
            # 如果没找到，提供更详细的错误信息
            searched_paths = ", ".join([os.path.join(d, "run_*") for d in search_dirs])
            raise FileNotFoundError(f"未找到任何训练目录（需以'run_'开头）\n已搜索路径: {searched_paths}")

    def get_resize_for_model(self) -> Optional[int]:
        """
        根据模型类型确定合适的resize参数
        Returns:
            适合该模型的resize值或None
        """
        resize = None  # 默认不需要resize
        if self.config and "model_name" in self.config:
            if self.config["model_name"] == "AlexNet" or self.config["model_name"] == "VGG":
                resize = 224  # AlexNet和VGG都需要224x224输入
            elif self.config["model_name"] == "GoogLeNet":
                resize = 96   # GoogLeNet需要96x96输入
        return resize
        
    def list_models_in_directory(self) -> None:
        """
        列出训练目录中的所有模型信息
        """
        if not self.run_dir:
            print("⚠️ 未设置训练目录，无法列出模型")
            return
        
        try:
            models_info = FileUtils.list_models_in_dir(self.run_dir)
            print(f"\n📋 {self.run_dir} 目录中的模型列表（按准确率排序）:")
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
                    f"   predict --run_dir='{self.run_dir}' --model_file='{models_info[1]['filename']}'")
        except Exception as e:
            print(f"⚠️ 列出模型文件时出错: {str(e)}")

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
        VisualizationTool.show_images(
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
            elif self.config["model_name"] == "GoogLeNet":
                input_size = (1, 96, 96)

        print(
            f"\n🔍 测试 {num_samples} 个随机输入（{input_size[1]}x{input_size[2]}灰度图）"
        )
        random_X = torch.randn(num_samples, *input_size)  # 模拟随机图像
        random_preds = self.predict(random_X)
        random_labels = d2l.get_fashion_mnist_labels(random_preds.cpu())

        # 获取模型输出概率分布
        with torch.no_grad():
            random_X = random_X.to(self.device)
            outputs = self.net(random_X)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            max_probs = max_probs.cpu().tolist()
        
        # 按照visualize_prediction的格式输出预测详情
        print(f"\n📋 随机输入预测详情:")
        print(f"预测类别: {random_preds.tolist()} → {random_labels}")
        print(f"预测置信度: {[f'{p:.2f}' for p in max_probs]}")
        
        # 识别高置信度预测（置信度>0.5）
        high_confidence_count = sum(1 for p in max_probs if p > 0.5)
        print(f"高置信度预测({high_confidence_count}/{num_samples}): 置信度>0.5")

        # 检查预测多样性（避免模型输出单一类别）
        unique_preds = torch.unique(random_preds).numel()
        if unique_preds < 3:
            print(f"⚠️ 警告: 随机预测类别较少（{unique_preds}种），模型可能未充分训练")
        else:
            print(f"✅ 随机预测类别多样（{unique_preds}种），模型状态正常")
            
    def run_prediction(self, batch_size: int = 256, num_samples: int = 10) -> Dict[str, Any]:
        """
        执行模型预测并返回结果
        Args:
            batch_size: 批次大小
            num_samples: 随机测试样本数
        Returns:
            预测结果字典
        """
        # 如果有训练目录且未指定模型文件，显示该目录下的所有模型信息
        if self.run_dir and not (self.model_path and os.path.basename(self.model_path) != FileUtils.find_best_model_in_dir(self.run_dir)):
            self.list_models_in_directory()
        
        # 根据模型类型确定resize参数
        resize = self.get_resize_for_model()
        
        # 加载数据（使用正确的resize参数）
        _, test_iter = DataLoader.load_data(
            batch_size=batch_size, resize=resize
        )
        
        # 执行预测可视化
        self.visualize_prediction(test_iter, n=num_samples)
        self.test_random_input(num_samples=num_samples)
        
        # 返回预测结果信息
        result = {
            "success": True,
            "model_name": self.config["model_name"] if self.config else "未知",
            "run_dir": self.run_dir,
            "model_path": self.model_path
        }
        
        return result

        