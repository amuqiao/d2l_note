from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from typing import List, Optional
from src.model_visualization.data_models import ModelInfoData
from src.helper_utils.helper_tools_registry import ToolRegistry
from src.utils.log_utils import get_logger

# 导入基类
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_visualizer.log", global_level="INFO")


class ModelComparisonVisualizer(BaseModelInfoVisualizer):
    """模型准确率比较可视化器：专注于比较多个模型的准确率指标"""
    
    def __init__(self):
        self.model_infos = []
        # 新增：设置排序方式
        self.sort_by = "accuracy"  # 默认按准确率排序
    
    def support(self, model_info: ModelInfoData) -> bool:
        # 这个可视化器需要至少包含准确率信息的模型
        return "final_test_acc" in model_info.metrics or "best_acc" in model_info.metrics
    
    def add_model_info(self, model_info: ModelInfoData):
        """添加要比较的模型信息"""
        self.model_infos.append(model_info)
    
    def set_sort_by(self, sort_by: str):
        """设置排序方式
        
        参数:
            sort_by: 排序依据，可选值: "accuracy" (按准确率) 或 "name" (按名称)
        """
        if sort_by in ["accuracy", "name"]:
            self.sort_by = sort_by
        else:
            logger.warning(f"无效的排序方式: {sort_by}，将使用默认值 'accuracy'")
    
    def visualize(self, model_info: ModelInfoData = None, show: bool = True, 
                  figsize: tuple = (15, 10), plot_type: str = "all") -> Optional[plt.Figure]:
        """
        可视化模型准确率比较
        
        参数:
            model_info: 可选的模型信息对象
            show: 是否显示图表
            figsize: 图表大小
            plot_type: 图表类型，可选值: "all" (全部), "bar" (柱状图), "ranking" (排名图), "scatter" (散点图)
        
        返回:
            matplotlib.pyplot.Figure: 生成的图表对象
        """
        try:
            # 如果传入了model_info，添加到比较列表
            if model_info:
                self.add_model_info(model_info)
            
            # 确保有模型可以比较
            if len(self.model_infos) < 2:
                logger.warning("模型比较需要至少2个模型信息")
                return None
            
            # 改进字体设置，确保中文正常显示
            try:
                # 使用工具注册中心设置字体
                ToolRegistry.call("setup_font")
                logger.info("成功调用ToolRegistry的setup_font方法")
            except Exception as e:
                logger.warning(f"调用ToolRegistry的setup_font失败: {str(e)}")
                
                # 手动设置中文字体，支持多种常见中文字体
                for font in ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS']:
                    try:
                        plt.rcParams["font.family"] = [font, "sans-serif"]
                        # 测试字体是否可用
                        plt.figure().text(0.5, 0.5, "测试中文字体")
                        plt.close()
                        logger.info(f"成功设置中文字体: {font}")
                        break
                    except:
                        continue
                else:
                    logger.warning("未能设置特定中文字体，使用系统默认字体")
                    plt.rcParams["font.family"] = ["sans-serif"]
            
            plt.rcParams["axes.unicode_minus"] = False  # 确保负号能够正确显示
            
            # 记录当前的字体配置，用于调试
            logger.info(f"当前字体配置: {plt.rcParams['font.family']}")
            
            # 准备比较数据
            model_data = []
            
            for info in self.model_infos:
                # 获取模型名称或标识符，改进命名显示
                model_name = self._get_model_name(info)
                
                # 获取测试准确率，优先使用final_test_acc，其次是best_acc
                test_acc = info.metrics.get("final_test_acc", 0)
                if test_acc == 0:
                    test_acc = info.metrics.get("best_acc", 0)
                
                # 获取训练准确率
                train_acc = info.metrics.get("train_acc", 0)
                if train_acc == 0:
                    train_acc = info.metrics.get("final_train_acc", 0)
                
                # 获取验证准确率
                val_acc = info.metrics.get("val_acc", 0)
                if val_acc == 0:
                    val_acc = info.metrics.get("best_val_acc", 0)
                
                # 获取训练时间
                train_time = 0
                if "total_training_time" in info.metrics:
                    time_str = info.metrics["total_training_time"]
                    time_match = re.search(r'([\d.]+)', time_str)
                    if time_match:
                        train_time = float(time_match.group(1))
                
                # 获取参数量
                param_count = info.params.get("total_params", 0)
                
                model_data.append({
                    "name": model_name,
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "train_time": train_time,
                    "params": param_count,
                    "path": info.path
                })
            
            # 根据排序方式排序
            if self.sort_by == "accuracy":
                model_data.sort(key=lambda x: x["test_acc"], reverse=True)
            else:
                model_data.sort(key=lambda x: x["name"])
            
            # 提取排序后的数据
            model_names = [d["name"] for d in model_data]
            test_accs = [d["test_acc"] for d in model_data]
            train_accs = [d["train_acc"] for d in model_data]
            val_accs = [d["val_acc"] for d in model_data]
            train_times = [d["train_time"] for d in model_data]
            params_counts = [max(1, d["params"]) for d in model_data]  # 确保至少为1，避免对数刻度问题
            
            # 创建图表
            if plot_type == "all":
                fig, axes = plt.subplots(2, 2, figsize=figsize)
            elif plot_type == "bar":
                fig, axes = plt.subplots(1, 1, figsize=figsize)
                axes = np.array([[axes]])
            elif plot_type == "ranking":
                fig, axes = plt.subplots(1, 1, figsize=figsize)
                axes = np.array([[axes]])
            elif plot_type == "scatter":
                fig, axes = plt.subplots(1, 1, figsize=figsize)
                axes = np.array([[axes]])
            else:
                fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # 1. 准确率比较（柱状图，支持训练、验证、测试准确率）
            if plot_type in ["all", "bar"]:
                # 设置条形宽度
                bar_width = 0.25
                x = np.arange(len(model_names))
                
                # 绘制三种准确率的条形图
                train_bars = axes[0, 0].bar(x - bar_width, train_accs, bar_width, label='训练准确率')
                val_bars = axes[0, 0].bar(x, val_accs, bar_width, label='验证准确率')
                test_bars = axes[0, 0].bar(x + bar_width, test_accs, bar_width, label='测试准确率')
                
                axes[0, 0].set_ylabel('准确率')
                axes[0, 0].set_title('模型准确率详细比较')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
                axes[0, 0].set_ylim(0, 1.1)
                axes[0, 0].legend()
                
                # 标注测试准确率数值（最重要的指标）
                for i, v in enumerate(test_accs):
                    axes[0, 0].text(i + bar_width, v + 0.02, f'{v:.4f}', ha='center')
                
                if plot_type == "bar":
                    fig.suptitle("模型准确率比较", fontsize=16)
            
            # 2. 测试准确率排名图 - 增强版，专注于比较模型准确率
            if plot_type in ["all", "ranking"]:
                # 使用水平条形图显示排名
                if plot_type == "all":
                    rank_ax = axes[0, 1]
                else:
                    rank_ax = axes[0, 0]
                
                # 反转顺序，使最高分在顶部
                reversed_names = model_names[::-1]
                reversed_accs = test_accs[::-1]
                reversed_train_accs = train_accs[::-1]
                reversed_val_accs = val_accs[::-1]
                
                # 计算准确率差异（泛化能力）
                acc_diffs = [test - train for test, train in zip(test_accs, train_accs)]
                reversed_acc_diffs = acc_diffs[::-1]
                
                # 确定图表高度，为更多信息留出空间
                if plot_type == "ranking":
                    # 为增强版排名图增加高度
                    fig.set_figheight(max(figsize[1], 1.5 * len(model_names)))
                
                # 使用渐变色显示，强调准确率差异
                colors = []
                for acc_diff in reversed_acc_diffs:
                    if acc_diff > 0:
                        # 测试准确率高于训练准确率，绿色系
                        colors.append(plt.cm.Greens(0.3 + min(0.7, acc_diff * 10)))
                    elif acc_diff < 0:
                        # 测试准确率低于训练准确率，红色系
                        colors.append(plt.cm.Reds(0.3 + min(0.7, abs(acc_diff) * 10)))
                    else:
                        # 准确率一致，中性色
                        colors.append(plt.cm.Greys(0.5))
                
                # 绘制主条形图（测试准确率）
                bars = rank_ax.barh(reversed_names, reversed_accs, color=colors, height=0.6, alpha=0.8, label='测试准确率')
                
                # 绘制训练准确率和验证准确率的线（作为参考）
                for i, (train_acc, val_acc) in enumerate(zip(reversed_train_accs, reversed_val_accs)):
                    if train_acc > 0:
                        rank_ax.axvline(x=train_acc, ymin=i/len(reversed_names) + 0.1/len(reversed_names), 
                                        ymax=(i+1)/len(reversed_names) - 0.1/len(reversed_names), 
                                        color='blue', linestyle='--', alpha=0.6, linewidth=1)
                        # 添加训练准确率标签
                        rank_ax.text(train_acc + 0.005, i, f'T: {train_acc:.3f}', va='center', fontsize=7, color='blue')
                    if val_acc > 0:
                        rank_ax.axvline(x=val_acc, ymin=i/len(reversed_names) + 0.3/len(reversed_names), 
                                        ymax=(i+1)/len(reversed_names) - 0.3/len(reversed_names), 
                                        color='purple', linestyle=':', alpha=0.6, linewidth=1)
                        # 添加验证准确率标签
                        rank_ax.text(val_acc + 0.005, i, f'V: {val_acc:.3f}', va='center', fontsize=7, color='purple')
                
                rank_ax.set_xlabel('准确率')
                rank_ax.set_title('模型准确率详细排名比较')
                rank_ax.set_xlim(0, 1.1)
                
                # 添加图例
                rank_ax.legend(['测试准确率', '训练准确率', '验证准确率'], loc='upper right')
                
                # 添加详细指标
                for i, (v, acc_diff, params, time) in enumerate(zip(reversed_accs, reversed_acc_diffs, 
                                                                   params_counts[::-1], train_times[::-1])):
                    # 准确率数值
                    rank_ax.text(v + 0.01, i, f'测试: {v:.4f}', va='center', fontsize=8)
                    # 排名标记
                    rank_ax.text(0.01, i, f'#{len(model_names) - i}', va='center', fontweight='bold')
                    # 准确率差异标记
                    diff_text = f'差异: {acc_diff:+.3f}'
                    diff_color = 'green' if acc_diff > 0 else 'red' if acc_diff < 0 else 'gray'
                    rank_ax.text(0.7, i, diff_text, va='center', fontsize=7, color=diff_color)
                
                if plot_type == "ranking":
                    fig.suptitle("模型准确率详细排名比较", fontsize=16)
            
            # 3. 准确率-时间权衡散点图（增强版）
            if plot_type in ["all", "scatter"]:
                if plot_type == "all":
                    scatter_ax = axes[1, 0]
                else:
                    scatter_ax = axes[0, 0]
                
                # 使用参数量作为点的大小
                sizes = [min(1000, p / 10000) for p in params_counts]  # 缩放参数量以适合点大小
                
                # 使用准确率作为颜色
                scatter = scatter_ax.scatter(train_times, test_accs, s=sizes, c=test_accs, 
                                           cmap='viridis', alpha=0.7, edgecolors='w', linewidths=1)
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=scatter_ax)
                cbar.set_label('测试准确率')
                
                scatter_ax.set_xlabel('训练时间 (秒)')
                scatter_ax.set_ylabel('测试准确率')
                scatter_ax.set_title('准确率-时间-参数量权衡分析')
                scatter_ax.set_ylim(0, 1.1)
                
                # 智能放置标签，避免重叠
                self._annotate_points_without_overlap(scatter_ax, train_times, test_accs, model_names)
                
                if plot_type == "scatter":
                    fig.suptitle("模型准确率-时间-参数量权衡分析", fontsize=16)
            
            # 4. 模型信息摘要表格
            if plot_type == "all":
                # 创建一个表格来展示模型的关键指标
                metrics_table_ax = axes[1, 1]
                metrics_table_ax.axis('off')  # 隐藏坐标轴
                
                # 准备表格数据
                table_data = []
                for i, data in enumerate(model_data):
                    rank = i + 1
                    name = data["name"] if len(data["name"]) <= 12 else data["name"][:9] + "..."
                    test_acc = f"{data['test_acc']:.4f}"
                    params = f"{data['params']:,}"
                    time = f"{data['train_time']:.1f}s" if data['train_time'] > 0 else "N/A"
                    
                    table_data.append([rank, name, test_acc, params, time])
                
                # 创建表格
                table = metrics_table_ax.table(cellText=table_data, 
                                              colLabels=["排名", "模型", "测试准确率", "参数量", "训练时间"],
                                              loc='center', cellLoc='center')
                
                # 设置表格样式
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)  # 调整表格大小
                
                # 突出显示第一行
                for (row, col), cell in table.get_celld().items():
                    if row == 0:  # 表头
                        cell.set_fontsize(11)
                        cell.set_fontweight('bold')
                    if row == 1 and col != -1:  # 第一名
                        cell.set_facecolor('#d4edda')
                        cell.set_fontweight('bold')
                
                metrics_table_ax.set_title('模型关键指标摘要', pad=20)
            
            # 设置整体标题
            if plot_type == "all":
                fig.suptitle("模型准确率比较分析报告", fontsize=16, y=0.98)
            
            # 优化布局管理
            try:
                # 针对不同的图表类型调整布局参数
                if plot_type == "ranking":
                    # 为排名图增加底部空间和左侧空间以适应长标签
                    plt.tight_layout(rect=[0.15, 0.1, 0.95, 0.96])
                else:
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
            except Exception as e:
                logger.warning(f"设置tight_layout失败: {str(e)}")
                # 如果失败，使用subplots_adjust替代，并根据图表类型调整参数
                if plot_type == "ranking":
                    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.92)
                else:
                    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.92, wspace=0.2, hspace=0.3)
            
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"绘制模型比较可视化失败: {str(e)}")
            return None
    
    def _get_model_name(self, model_info: ModelInfoData) -> str:
        """获取有意义的模型名称"""
        # 优先从metrics或params中获取模型名称
        if "model_name" in model_info.metrics:
            return model_info.metrics["model_name"]
        elif "model_type" in model_info.params:
            return model_info.params["model_type"]
        
        # 从路径中提取有意义的名称
        path_basename = os.path.basename(model_info.path)
        
        # 检查是否包含模型名称关键字
        model_types = ["LeNet", "AlexNet", "VGG", "ResNet", "GoogLeNet", "DenseNet", "MLP", "NIN"]
        for model_type in model_types:
            if model_type.lower() in path_basename.lower():
                # 尝试提取epoch或准确率信息
                epoch_match = re.search(r'epoch_(\d+)', path_basename, re.IGNORECASE)
                acc_match = re.search(r'acc_([\d.]+)', path_basename, re.IGNORECASE)
                
                name_parts = [model_type]
                if acc_match:
                    name_parts.append(f"{float(acc_match.group(1))*100:.1f}%")
                if epoch_match:
                    name_parts.append(f"E{epoch_match.group(1)}")
                
                return "-".join(name_parts)
        
        # 如果没有找到明显的模型类型，截取路径名
        if len(path_basename) > 15:
            return path_basename[:12] + '...'
        return path_basename
    
    def _annotate_points_without_overlap(self, ax, x, y, labels):
        """智能标注散点图点，避免标签重叠"""
        # 简单实现：对于密集区域，只标注极值点
        if len(x) <= 10:
            # 少量点，全部标注
            for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
                ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        else:
            # 大量点，只标注极值点
            max_acc_idx = y.index(max(y))
            min_acc_idx = y.index(min(y))
            max_time_idx = x.index(max(x))
            min_time_idx = x.index(min(x))
            
            # 确保索引唯一
            unique_indices = list(set([max_acc_idx, min_acc_idx, max_time_idx, min_time_idx]))
            
            for idx in unique_indices:
                ax.annotate(labels[idx], (x[idx], y[idx]), xytext=(5, 5), textcoords='offset points',
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))