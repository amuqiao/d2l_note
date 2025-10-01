# 设置环境变量以解决 OpenMP 运行时库冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


# ------------------------------
# 1. 模型组件：编码器
# ------------------------------
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(** kwargs)
        
        # 嵌入层：将词元索引转换为向量表示
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # GRU层：用于提取序列特征
        # 参数说明：输入特征维度(embed_size)、隐藏状态维度(num_hiddens)
        # 层数(num_layers)、 dropout比率
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输入X形状：(batch_size, num_steps)
        # 经过嵌入层后形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        
        # GRU需要时间步作为第一个维度，因此调整维度顺序
        # 调整后形状：(num_steps, batch_size, embed_size)
        X = X.permute(1, 0, 2)
        
        # 如果未指定初始状态，则默认为0
        # output形状：(num_steps, batch_size, num_hiddens)
        # state形状：(num_layers, batch_size, num_hiddens)
        output, state = self.rnn(X)
        
        return output, state


# ------------------------------
# 2. 模型组件：解码器
# ------------------------------
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(** kwargs)
        
        # 嵌入层：将目标语言词元索引转换为向量表示
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # GRU层：输入包含嵌入向量和编码器最后一层的隐藏状态
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        
        # 全连接层：将隐藏状态转换为词汇表大小的输出
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        """使用编码器的输出初始化解码器状态"""
        return enc_outputs[1]

    def forward(self, X, state):
        # 输入X形状：(batch_size, num_steps)
        # 经过嵌入层后形状：(batch_size, num_steps, embed_size)
        # 调整维度顺序后：(num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        
        # 将编码器最后一层的隐藏状态广播到与X相同的时间步长度
        # context形状：(num_steps, batch_size, num_hiddens)
        context = state[-1].repeat(X.shape[0], 1, 1)
        
        # 拼接嵌入向量和上下文信息
        # X_and_context形状：(num_steps, batch_size, embed_size + num_hiddens)
        X_and_context = torch.cat((X, context), 2)
        
        # 解码器前向传播
        # output形状：(num_steps, batch_size, num_hiddens)
        # state形状：(num_layers, batch_size, num_hiddens)
        output, state = self.rnn(X_and_context, state)
        
        # 通过全连接层转换为词汇表分布，并调整维度顺序
        # 最终output形状：(batch_size, num_steps, vocab_size)
        output = self.dense(output).permute(1, 0, 2)
        
        return output, state


# ------------------------------
# 3. 辅助函数：序列掩码
# ------------------------------
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项（超出有效长度的部分）"""
    maxlen = X.size(1)
    # 创建掩码：对于每个样本，有效长度内为True，之外为False
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # 将无效部分设为指定值
    X[~mask] = value
    return X


# ------------------------------
# 4. 损失函数：带遮蔽的交叉熵
# ------------------------------
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数，忽略填充部分的损失"""
    # pred的形状：(batch_size, num_steps, vocab_size)
    # label的形状：(batch_size, num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        # 创建与label相同形状的权重矩阵
        weights = torch.ones_like(label)
        # 对权重矩阵进行遮蔽，无效位置权重为0
        weights = sequence_mask(weights, valid_len)
        
        # 关闭默认的损失 reduction，以便进行自定义处理
        self.reduction = 'none'
        
        # 检查pred的维度并调整
        if pred.dim() == 2:
            # 如果pred是2D，添加一个维度使其成为3D
            pred = pred.unsqueeze(2)
        
        # 计算未加权的损失
        # 注意：CrossEntropyLoss期望输入形状为(batch_size, vocab_size, num_steps)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        
        # 应用权重并在时间步维度取平均
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


# ------------------------------
# 5. 模型训练函数
# ------------------------------
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    # 初始化权重的Xavier方法
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    # 应用权重初始化
    net.apply(xavier_init_weights)
    # 将模型移动到指定设备
    net.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 损失函数
    loss = MaskedSoftmaxCELoss()
    
    # 切换到训练模式
    net.train()
    # 动画器，用于可视化训练过程
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        # 用于累积损失和词元数量
        metric = d2l.Accumulator(2)
        
        for batch in data_iter:
            optimizer.zero_grad()  # 清零梯度
            
            # 获取批量数据并移动到指定设备
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            
            # 解码器输入：在目标序列前添加开始标记
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            
            # 前向传播
            net_outputs = net(X, dec_input, X_valid_len)
            
            # 检查 net_outputs 的类型和形状
            if isinstance(net_outputs, tuple) or isinstance(net_outputs, list):
                # 如果是元组或列表，取第一个元素作为预测结果
                Y_hat = net_outputs[0]
            else:
                # 如果是单个张量，直接使用
                Y_hat = net_outputs
            
            # 确保 Y_hat 的形状正确 [batch_size, num_steps, vocab_size]
            if Y_hat.dim() == 2:
                # 如果是2D形状，添加一个维度
                Y_hat = Y_hat.unsqueeze(1)
            elif Y_hat.dim() == 3 and Y_hat.shape[0] != Y.shape[0]:
                # 如果是3D但batch_size不匹配，尝试转置
                # 这里可能需要根据实际情况调整
                print(f"Warning: Y_hat shape {Y_hat.shape} may not match expected shape")
                
            # 计算损失
            l = loss(Y_hat, Y, Y_valid_len)
            
            # 反向传播
            l.sum().backward()      # 损失函数的标量进行反向传播
            d2l.grad_clipping(net, 1)  # 梯度裁剪，防止梯度爆炸
            
            # 更新参数
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            
            # 累积损失和词元数量
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        
        # 每10个epoch可视化一次损失
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    
    # 输出最终训练结果
    print(f'损失 {metric[0] / metric[1]:.3f}, '
          f'速度 {metric[1] / timer.stop():.1f} 词元/秒 '
          f'在 {str(device)} 上')


# ------------------------------
# 6. 预测函数
# ------------------------------
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """使用序列到序列模型进行预测（翻译）"""
    # 在预测时将模型设置为评估模式
    net.eval()
    
    # 对源句子进行预处理：转换为词元索引并添加结束标记
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    
    # 有效长度
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    
    # 截断或填充序列至指定长度
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    
    # 添加批量维度
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    
    # 编码器前向传播
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    
    # 初始化解码器状态
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    
    # 解码器初始输入：开始标记
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq, attention_weight_seq = [], []
    
    # 生成目标序列
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        
        # 选择概率最高的词元作为下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        
        # 保存注意力权重（如果需要）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        
        # 如果预测到结束标记，则停止生成
        if pred == tgt_vocab['<eos>']:
            break
        
        output_seq.append(pred)
    
    # 将预测的词元索引转换为字符串
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


# ------------------------------
# 7. 评估指标：BLEU
# ------------------------------
def bleu(pred_seq, label_seq, k):
    """计算BLEU分数，评估翻译质量"""
    # 将预测序列和标签序列拆分为词元
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    
    # 计算长度惩罚因子
    score = math.exp(min(0, 1 - len_label / len_pred))
    
    # 计算n-gram匹配率
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        
        # 统计标签序列中所有n-gram的出现次数
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        
        # 统计预测序列中与标签序列匹配的n-gram数量
        for i in range(len_pred - n + 1):
            current_sub = ' '.join(pred_tokens[i: i + n])
            if label_subs[current_sub] > 0:
                num_matches += 1
                label_subs[current_sub] -= 1
        
        # 累积BLEU分数
        if len_pred - n + 1 > 0:
            score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    
    return score


# ------------------------------
# 8. 模型训练与测试示例
# ------------------------------
if __name__ == "__main__":
    # 超参数设置
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    # 加载数据
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

    # 初始化编码器和解码器
    encoder = Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

    # 组合成完整的编码器-解码器模型
    net = d2l.EncoderDecoder(encoder, decoder)

    # 训练模型
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 测试示例
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, BLEU分数 {bleu(translation, fra, k=2):.3f}')
