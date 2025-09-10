# 设置中文显示
import sys
import matplotlib.pyplot as plt

# 检测操作系统并设置相应的字体
if sys.platform.startswith('win'):
    # Windows系统
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong"]
elif sys.platform.startswith('darwin'):
    # macOS系统
    plt.rcParams["font.family"] = ["Arial Unicode MS", "Helvetica Neue", "Heiti TC"]
elif sys.platform.startswith('linux'):
    # Linux系统 - 添加字体回退机制
    plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans", "sans-serif"]

# 正确显示负号
plt.rcParams["axes.unicode_minus"] = False


def test_chinese_display():
    """
    测试中文显示功能，创建一个简单图表并显示中文字符
    """
    print("\n=== 测试中文显示功能 ===")
    print(f"当前操作系统: {sys.platform}")
    print(f"设置的字体列表: {plt.rcParams['font.family']}")
    
    # 创建一个简单的图表用于测试中文显示
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    categories = ["苹果", "香蕉", "橙子", "葡萄", "草莓"]
    values = [35, 25, 20, 15, 5]
    
    # 绘制图表
    bars = ax.bar(categories, values, color=['red', 'yellow', 'orange', 'purple', 'pink'])
    
    # 添加中文标题和标签
    ax.set_title("水果销量统计图", fontsize=16)
    ax.set_xlabel("水果种类", fontsize=12)
    ax.set_ylabel("销量占比(%)", fontsize=12)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}%', ha='center', va='bottom')
    
    # 添加图例
    ax.legend(["销量数据"], loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图表（如果是交互环境）
    # plt.show()
    
    # 保存图表到文件，方便查看效果
    plt.savefig("chinese_font_test.png")
    print("\n✅ 中文显示测试完成！图表已保存为 'chinese_font_test.png'")
    print("请查看图表确认中文是否正常显示。")


if __name__ == "__main__":
    # 当脚本直接运行时，执行中文显示测试
    test_chinese_display()


