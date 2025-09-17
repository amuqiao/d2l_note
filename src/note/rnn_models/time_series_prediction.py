"""时间序列预测模块

该模块实现了基于多层感知机(MLP)的时间序列预测功能，主要包括：
1. 时间序列数据生成（正弦函数加噪声）
2. 数据预处理和特征工程
3. 简单MLP模型定义和训练
4. 单步和多步预测实现
5. 预测结果可视化

模块特点：
- 自动检测操作系统并配置matplotlib中文显示
- 解决OpenMP库冲突问题，确保在多环境下正常运行
- 清晰的功能模块化设计，易于扩展和集成
- 丰富的可视化功能，直观展示预测效果

使用示例：
    # 直接运行模块
    python time_series_prediction.py
    
    # 作为模块导入使用
    from src.models.rnn_models.time_series_prediction import main, generate_time_series
    main()  # 运行完整流程
    
主要函数：
- generate_time_series(): 生成时间序列数据
- prepare_data(): 数据预处理和特征工程
- get_net(): 定义MLP模型
- train(): 训练模型
- visualize_predictions(): 可视化预测结果
- main(): 主函数，整合所有流程
"""
import os
import sys
# 解决OpenMP库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


def setup_font():
    """配置matplotlib中文显示"""
    if sys.platform.startswith("win"):
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    elif sys.platform.startswith("darwin"):
        plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC"]
    elif sys.platform.startswith("linux"):
        plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# --------------------------
# 1. 数据生成
# --------------------------
def generate_time_series(T=1000):
    """生成时间序列数据：正弦函数加噪声"""
    time = torch.arange(1, T + 1, dtype=torch.float32)
    # 生成带有噪声的正弦曲线数据
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    # 可视化生成的数据
    d2l.plot(time, [x], '时间', '数值', xlim=[1, 1000], figsize=(6, 3))
    plt.title("原始时间序列数据")
    plt.show()
    return time, x

# --------------------------
# 2. 数据预处理
# --------------------------
def prepare_data(x, tau=4, n_train=600, batch_size=16):
    """准备用于训练的特征和标签数据"""
    T = len(x)
    # 初始化特征矩阵，每行包含tau个时间步的数据
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    
    # 标签为tau步之后的数据
    labels = x[tau:].reshape((-1, 1))
    
    # 构建训练数据迭代器
    train_iter = d2l.load_array(
        (features[:n_train], labels[:n_train]),
        batch_size, 
        is_train=True
    )
    
    return features, labels, train_iter

# --------------------------
# 3. 模型定义
# --------------------------
def init_weights(m):
    """初始化网络权重"""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    """定义一个简单的多层感知机模型"""
    net = nn.Sequential(
        nn.Linear(4, 10),  # 输入层到隐藏层
        nn.ReLU(),         # 激活函数
        nn.Linear(10, 1)   # 隐藏层到输出层
    )
    net.apply(init_weights)  # 应用权重初始化
    return net

# --------------------------
# 4. 模型训练
# --------------------------
def train(net, train_iter, loss, epochs, lr):
    """训练模型"""
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()  # 清零梯度
            l = loss(net(X), y)  # 计算损失
            l.sum().backward()   # 反向传播
            trainer.step()       # 更新参数
        # 打印当前轮次的损失
        print(f'轮次 {epoch + 1}, 损失: {d2l.evaluate_loss(net, train_iter, loss):f}')

# --------------------------
# 5. 预测与可视化
# --------------------------
def visualize_predictions(time, x, onestep_preds, multistep_preds, n_train, tau):
    """可视化单步预测和多步预测结果"""
    # 单步预测可视化
    d2l.plot(
        [time, time[tau:]],
        [x.detach().numpy(), onestep_preds.detach().numpy()], 
        '时间', '数值', 
        legend=['原始数据', '单步预测'], 
        xlim=[1, 1000],
        figsize=(6, 3)
    )
    plt.title("单步预测结果")
    plt.show()
    
    # 多步预测可视化
    d2l.plot(
        [time, time[tau:], time[n_train + tau:]],
        [x.detach().numpy(), onestep_preds.detach().numpy(),
         multistep_preds[n_train + tau:].detach().numpy()], 
        '时间', '数值', 
        legend=['原始数据', '单步预测', '多步预测'], 
        xlim=[1, 1000], 
        figsize=(6, 3)
    )
    plt.title("多步预测结果")
    plt.show()

def visualize_multi_step_preds(time, x, tau=4, max_steps=64):
    """可视化不同步数的预测结果"""
    T = len(x)
    # 准备特征矩阵用于多步预测
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    
    # 填充观测数据
    for i in range(tau):
        features[:, i] = x[i: i + T - tau - max_steps + 1]
    
    # 填充预测数据
    net = get_net()  # 使用相同的网络结构
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)
    
    # 可视化不同步数的预测结果
    steps = (1, 4, 16, 64)
    d2l.plot(
        [time[tau + i - 1: T - max_steps + i] for i in steps],
        [features[:, (tau + i - 1)].detach().numpy() for i in steps], 
        '时间', '数值',
        legend=[f'{i}-步预测' for i in steps], 
        xlim=[5, 1000],
        figsize=(6, 3)
    )
    plt.title("不同步数的预测结果对比")
    plt.show()

# --------------------------
# 主程序
# --------------------------
def main():
    # 设置matplotlib支持中文显示
    setup_font()
    
    # 参数设置
    T = 1000         # 时间序列长度
    tau = 4          # 时间步长
    batch_size = 16  # 批量大小
    n_train = 600    # 训练样本数
    epochs = 5       # 训练轮数
    lr = 0.01        # 学习率
    
    # 生成时间序列数据
    time, x = generate_time_series(T)
    
    # 准备训练数据
    features, labels, train_iter = prepare_data(
        x, tau, n_train, batch_size
    )
    
    # 定义模型和损失函数
    net = get_net()
    loss = nn.MSELoss(reduction='none')  # 平方损失
    
    # 训练模型
    train(net, train_iter, loss, epochs, lr)
    
    # 单步预测
    onestep_preds = net(features)
    
    # 多步预测
    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train + tau] = x[: n_train + tau]
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1))
        )
    
    # 可视化预测结果
    visualize_predictions(time, x, onestep_preds, multistep_preds, n_train, tau)
    
    # 可视化不同步数的预测
    visualize_multi_step_preds(time, x, tau)

if __name__ == "__main__":
    main()
    