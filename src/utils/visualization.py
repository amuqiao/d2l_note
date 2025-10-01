"""可视化工具类"""
import matplotlib.pyplot as plt
import sys
import torch
from d2l import torch as d2l

class VisualizationTool:
    """可视化工具类：提供字体设置、图像显示等功能"""

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