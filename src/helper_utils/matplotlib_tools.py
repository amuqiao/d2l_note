import matplotlib.pyplot as plt
import sys


def setup_matplotlib_font():
    """配置matplotlib中文显示
    
    该函数会根据不同操作系统设置合适的字体，确保中文能够正确显示。
    支持Windows、macOS和Linux系统。
    
    Example:
        >>> from src.helper_utils.font_utils import setup_matplotlib_font
        >>> setup_matplotlib_font()  # 调用后，matplotlib将正确显示中文
    """
    if sys.platform.startswith("win"):
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    elif sys.platform.startswith("darwin"):
        plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC"]
    elif sys.platform.startswith("linux"):
        plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


if __name__ == "__main__":
    # 演示如何使用该工具
    setup_matplotlib_font()
    print("Matplotlib字体设置已完成，当前字体配置:")
    print(f"- 字体族: {plt.rcParams['font.family']}")
    print(f"- 负号显示: {plt.rcParams['axes.unicode_minus']}")