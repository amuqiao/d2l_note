# 导入必要的库
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import sys
import os
import re
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

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# 配置d2l库的数据集下载
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)

from src.models.rnn_models.sequence_data_processing import load_data_time_machine

# ----------------------------
# 配置参数
# ----------------------------
# 批量大小和时间步长
batch_size, num_steps = 32, 35
# 隐藏层大小
num_hiddens = 512
# 训练轮数和学习率
num_epochs, lr = 500, 1


# ----------------------------
# 数据准备
# ----------------------------
def load_data():
    """加载时间机器数据集并返回迭代器和词汇表"""
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    return train_iter, vocab

# 加载数据
train_iter, vocab = load_data()
vocab_size = len(vocab)


# ----------------------------
# 模型参数与状态初始化
# ----------------------------
def get_params(vocab_size, num_hiddens, device):
    """初始化RNN模型参数"""
    num_inputs = num_outputs = vocab_size  # 输入和输出大小都等于词汇表大小
    
    # 正态分布初始化函数
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    
    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))  # 输入到隐藏层的权重
    W_hh = normal((num_hiddens, num_hiddens)) # 隐藏层到隐藏层的权重
    b_h = torch.zeros(num_hiddens, device=device)  # 隐藏层偏置
    
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs)) # 隐藏层到输出层的权重
    b_q = torch.zeros(num_outputs, device=device)  # 输出层偏置
    
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    """初始化RNN的隐藏状态"""
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# ----------------------------
# RNN核心逻辑
# ----------------------------
def rnn_forward(inputs, state, params):
    """
    RNN前向传播计算
    
    参数:
        inputs: 输入数据，形状为(时间步数量，批量大小，词表大小)
        state: 初始隐藏状态
        params: 模型参数列表
    
    返回:
        outputs: 所有时间步的输出
        new_state: 最后一个时间步的隐藏状态
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state  # 解包隐藏状态
    outputs = []
    
    # 遍历每个时间步的输入
    for X in inputs:
        # 计算隐藏状态: H = tanh(X·W_xh + H·W_hh + b_h)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 计算输出: Y = H·W_hq + b_q
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    
    # 拼接所有时间步的输出并返回
    return torch.cat(outputs, dim=0), (H,)


# ----------------------------
# RNN模型类
# ----------------------------
class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size = vocab_size  # 词汇表大小
        self.num_hiddens = num_hiddens  # 隐藏层大小
        self.params = get_params(vocab_size, num_hiddens, device)  # 模型参数
        self.init_state = init_state  # 状态初始化函数
        self.forward_fn = forward_fn  # 前向传播函数
    
    def __call__(self, X, state):
        """调用模型进行前向传播"""
        # 将输入转换为独热编码
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        """获取初始状态"""
        return self.init_state(batch_size, self.num_hiddens, device)


# ----------------------------
# 预测功能
# ----------------------------
def predict_chars(prefix, num_preds, net, vocab, device):
    """
    根据前缀生成后续字符
    
    参数:
        prefix: 前缀字符串
        num_preds: 要预测的字符数量
        net: 训练好的模型
        vocab: 词汇表
        device: 计算设备
    
    返回:
        生成的字符串（前缀+预测结果）
    """
    # 初始化状态
    state = net.begin_state(batch_size=1, device=device)
    # 将前缀转换为索引
    outputs = [vocab[prefix[0]]]
    
    # 定义获取输入的函数
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    # 预热期：处理前缀中的每个字符
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    
    # 预测阶段：生成指定数量的字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    
    # 将索引转换回字符并返回
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# ----------------------------
# 训练辅助函数
# ----------------------------
def grad_clipping(net, theta):
    """
    裁剪梯度，防止梯度爆炸
    
    参数:
        net: 神经网络模型
        theta: 梯度裁剪阈值
    """
    # 获取需要计算梯度的参数
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    
    # 计算梯度的L2范数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    
    # 如果范数超过阈值，则裁剪梯度
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# ----------------------------
# 训练流程
# ----------------------------
def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """
    训练网络一个迭代周期
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        updater: 参数更新器
        device: 计算设备
        use_random_iter: 是否使用随机迭代
    
    返回:
        perplexity: 困惑度（越低越好）
        speed: 训练速度（词元/秒）
    """
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 用于累积训练损失和词元数量
    
    for X, Y in train_iter:
        # 初始化状态（首次迭代或随机迭代时）
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            #  detach()用于截断梯度计算图，防止梯度流动到之前的时间步
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        
        # 调整输入输出形状并移动到指定设备
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        
        # 反向传播和参数更新
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)  # 裁剪梯度
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)  # 裁剪梯度
            updater(batch_size=1)  # 自定义更新
        
        # 累积损失和词元数量
        metric.add(l * y.numel(), y.numel())
    
    # 计算困惑度和训练速度
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_model(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """
    训练模型主函数
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        vocab: 词汇表
        lr: 学习率
        num_epochs: 训练轮数
        device: 计算设备
        use_random_iter: 是否使用随机迭代
    """
    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    
    # 设置动画器用于可视化训练过程
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                           legend=['train'], xlim=[10, num_epochs])
    
    # 初始化参数更新器
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    
    # 定义预测函数
    predict = lambda prefix: predict_chars(prefix, 50, net, vocab, device)
    
    # 开始训练
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(
            net, train_iter, loss, updater, device, use_random_iter)
        
        # 每10个epoch打印一次预测结果并更新动画
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    
    # 输出最终结果
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


# ----------------------------
# 主程序执行
# ----------------------------
if __name__ == "__main__":
    # 配置中文字体显示
    setup_font()
    # 获取计算设备（优先使用GPU）
    device = d2l.try_gpu()
    
    # 实例化第一个模型并训练（不使用随机迭代）
    print("训练第一个模型（不使用随机迭代）：")
    net = RNNModelScratch(vocab_size, num_hiddens, device, 
                         get_params, init_rnn_state, rnn_forward)
    train_model(net, train_iter, vocab, lr, num_epochs, device)
    
    # 实例化第二个模型并训练（使用随机迭代）
    print("\n训练第二个模型（使用随机迭代）：")
    net_random = RNNModelScratch(vocab_size, num_hiddens, device,
                                get_params, init_rnn_state, rnn_forward)
    train_model(net_random, train_iter, vocab, lr, num_epochs, device,
               use_random_iter=True)
    
    # 显示所有图像
    plt.show()
