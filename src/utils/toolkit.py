import torch
import matplotlib.pyplot as plt
import os
import glob
import json
import datetime
import re
import sys
from d2l import torch as d2l
from torch import nn

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


    
    
    