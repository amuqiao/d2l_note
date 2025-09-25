import torch
# 解决OpenMP运行时库冲突问题
# 设置环境变量允许多个OpenMP运行时库共存
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入新的工具类
from .visualization import VisualizationTool
from .file_utils import FileUtils
from .network_utils import NetworkUtils

# ========================= 通用工具类 =========================
class Toolkit:
    """通用工具类：字体设置、目录创建、配置保存等（保持向后兼容）"""

    @staticmethod
    def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5, figsize=None):
        """使用matplotlib原生方法显示图像网格，支持缩放和自定义figsize"""
        return VisualizationTool.show_images(imgs, num_rows, num_cols, titles, scale, figsize)

    @staticmethod
    def setup_font():
        """配置matplotlib中文显示"""
        return VisualizationTool.setup_font()

    @staticmethod
    def create_run_dir(prefix="runs/run_", root_dir=None):
        """创建时间戳唯一目录（格式：run_年日月_时分秒）"""
        return FileUtils.create_run_dir(prefix, root_dir)

    @staticmethod
    def save_config(config_dict, save_path):
        """保存配置到JSON文件（便于追溯训练参数）"""
        return FileUtils.save_config(config_dict, save_path)

    @staticmethod
    def save_metrics(metrics_dict, save_path):
        """保存训练指标到JSON文件（便于性能对比）"""
        return FileUtils.save_metrics(metrics_dict, save_path)

    @staticmethod
    def find_best_model_in_dir(run_dir):
        """在训练目录中查找最佳模型文件（匹配best_model开头的.pth）"""
        return FileUtils.find_best_model_in_dir(run_dir)

    @staticmethod
    def list_models_in_dir(run_dir):
        """列出目录中的所有模型文件及其准确率信息"""
        return FileUtils.list_models_in_dir(run_dir)

    @staticmethod
    def test_network_shape(net, input_size=(1, 1, 28, 28)):
        """测试网络各层输出形状"""
        return NetworkUtils.test_network_shape(net, input_size)

# ========================= 动画工具类 =========================
class AnimatorTool:
    """动画工具类：实时可视化功能（保持向后兼容）"""

    @staticmethod
    def create_animator(xlabel="迭代周期", xlim=None, legend=None):
        """创建基础动画器"""
        return VisualizationTool.create_animator(xlabel, xlim, legend)

    @staticmethod
    def update_realtime(animator, x_value, metrics):
        """实时模式：在训练过程中更新动画"""
        return VisualizationTool.update_realtime(animator, x_value, metrics)


    
    
    