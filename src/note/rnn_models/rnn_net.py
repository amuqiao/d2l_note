# 添加项目根目录到系统路径，以便能够正确导入模块
import sys
import os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（向上两级）
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
# 将项目根目录添加到系统路径
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入必要的库
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import sys
import os
import matplotlib.pyplot as plt
# 导入本地实现的时间机器数据集加载函数和预测训练函数
from src.models.rnn_models.sequence_data_processing import load_data_time_machine
from src.models.rnn_models.rnn_text_prediction import predict_chars, train_model

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def setup_font():
    """配置matplotlib中文显示"""
    if sys.platform.startswith("win"):
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    elif sys.platform.startswith("darwin"):
        plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC"]
    elif sys.platform.startswith("linux"):
        plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 配置中文字体显示
setup_font()

# ------------------------------
# 数据准备与参数配置
# ------------------------------

# 设置批量大小和时间步数
batch_size, num_steps = 32, 35

# 加载时间机器数据集，返回迭代器和词汇表
# 该数据集包含《时间机器》文本，用于训练字符级语言模型
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# ------------------------------
# 循环神经网络(RNN)配置
# ------------------------------

# 隐藏层单元数量
num_hiddens = 256

# 定义RNN层：输入大小为词汇表大小，输出大小为隐藏层单元数
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# 初始化隐藏状态：(层数*方向数, 批量大小, 隐藏单元数)
state = torch.zeros((1, batch_size, num_hiddens))
print(f"初始隐藏状态形状: {state.shape}")  # 输出初始隐藏状态的形状

# 测试RNN层的输出形状
# 创建随机输入：(时间步数, 批量大小, 词汇表大小)
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)

# 输出RNN层的输出形状和更新后的隐藏状态形状
print(f"RNN输出形状: {Y.shape}, 更新后的隐藏状态形状: {state_new.shape}")

# ------------------------------
# RNN模型定义
# ------------------------------

class RNNModel(nn.Module):
    """循环神经网络语言模型"""
    
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(** kwargs)
        self.rnn = rnn_layer          # RNN层
        self.vocab_size = vocab_size  # 词汇表大小
        self.num_hiddens = self.rnn.hidden_size  # 隐藏层单元数
        
        # 根据RNN是否双向，设置方向数并定义全连接层
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        """前向传播函数"""
        # 将输入转换为独热编码：(批量大小, 时间步数) -> (时间步数, 批量大小, 词汇表大小)
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)  # 转换为浮点型张量
        
        # 通过RNN层获取输出和更新后的状态
        Y, state = self.rnn(X, state)
        
        # 通过全连接层计算输出：将RNN输出映射到词汇表大小
        # 先将Y的形状改为(时间步数*批量大小, 隐藏单元数)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        """初始化隐藏状态"""
        if not isinstance(self.rnn, nn.LSTM):
            # GRU和RNN以张量作为隐状态
            return torch.zeros(
                (self.num_directions * self.rnn.num_layers,
                 batch_size, self.num_hiddens),
                device=device
            )
        else:
            # LSTM以元组(隐状态, 细胞状态)作为隐状态
            return (torch.zeros(
                (self.num_directions * self.rnn.num_layers,
                 batch_size, self.num_hiddens), device=device),
                    torch.zeros(
                (self.num_directions * self.rnn.num_layers,
                 batch_size, self.num_hiddens), device=device))

# ------------------------------
# 模型测试与训练
# ------------------------------

# 选择计算设备（GPU优先）
device = d2l.try_gpu()

# 实例化模型并移动到指定设备
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)

# 测试模型预测功能：生成以"time traveller"开头的文本
print("初始预测结果:")
print(predict_chars('time traveller', 10, net, vocab, device))

# 设置训练参数
num_epochs, lr = 500, 1.0

# 保存模型
model_save_path = 'checkpoints/rnn_model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 训练模型
print("\n开始训练模型...")
train_model(net, train_iter, vocab, lr, num_epochs, device)


torch.save(net.state_dict(), model_save_path)
print(f"模型已保存到 {model_save_path}")

# 显示所有图像
plt.show()

# 加载模型
net.load_state_dict(torch.load(model_save_path))
net.eval()

# 训练结束后预测结果:
print("训练结束后预测结果:")
print(predict_chars('time traveller', 10, net, vocab, device))
