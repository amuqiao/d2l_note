import matplotlib.pyplot as plt
import sys


def setup_matplotlib_font():
    """配置matplotlib中文显示，适配不同操作系统"""
    if sys.platform.startswith("win"):
        # Windows系统默认字体
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    elif sys.platform.startswith("darwin"):
        # macOS系统默认字体
        plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC"]
    elif sys.platform.startswith("linux"):
        # Linux系统默认字体
        plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
