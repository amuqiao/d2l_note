# 导入必要的库
import random
import torch
from d2l import torch as d2l
import requests
from io import StringIO
import re

# 配置时间机器数据集
# 这行代码定义了数据集的URL和SHA1哈希值，用于验证下载的文件完整性
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)

# --------------------------
# 修复：添加自定义的read_time_machine函数
# --------------------------

def read_time_machine():
    """
    加载时间机器数据集并进行基本清洗
    
    返回:
        list: 清洗后的文本行列表，每个元素为一行文本
    """
    # 下载并打开文件
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    
    # 清洗文本：只保留字母，其他字符替换为空格，转为小写并去除首尾空白
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# --------------------------
# 1. 数据加载与预处理
# --------------------------

def load_and_preprocess_data():
    """加载文本数据并进行预处理（分词和构建词表）"""
    # 读取《时光机器》文本并进行分词
    raw_text = read_time_machine()
    tokens = d2l.tokenize(raw_text)
    
    # 将所有文本行拼接成一个长序列
    corpus = [token for line in tokens for token in line]
    
    # 构建词表，将 token 映射到整数索引
    vocab = d2l.Vocab(corpus)
    
    return corpus, vocab

# 加载数据并创建词表
corpus, vocab = load_and_preprocess_data()

# 查看词表中频率最高的10个词
print("词表中频率最高的10个词：")
print(vocab.token_freqs[:10])


# --------------------------
# 2. 词频分析与可视化
# --------------------------

def analyze_token_frequency(vocab, corpus):
    """分析不同长度词序列的频率分布并可视化"""
    # 一元词频率
    unigram_freqs = [freq for token, freq in vocab.token_freqs]
    
    # 绘制一元词频率分布（对数坐标）
    d2l.plot(
        unigram_freqs, 
        xlabel='词元索引 (log)', 
        ylabel='频率 (log)',
        xscale='log', 
        yscale='log'
    )
    
    # 生成二元词并计算频率
    bigram_tokens = [' '.join(pair) for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = d2l.Vocab(bigram_tokens)
    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    print("\n二元词中频率最高的10个词对：")
    print(bigram_vocab.token_freqs[:10])
    
    # 生成三元词并计算频率
    trigram_tokens = [' '.join(triple) for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = d2l.Vocab(trigram_tokens)
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    print("\n三元词中频率最高的10个词组合：")
    print(trigram_vocab.token_freqs[:10])
    
    # 对比一元词、二元词和三元词的频率分布
    d2l.plot(
        [unigram_freqs, bigram_freqs, trigram_freqs], 
        xlabel='词元索引 (log)', 
        ylabel='频率 (log)', 
        xscale='log', 
        yscale='log',
        legend=['一元词', '二元词', '三元词']
    )




# --------------------------
# 3. 序列采样方法
# --------------------------

def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    使用随机抽样生成小批量子序列
    
    参数:
        corpus: 完整的语料库序列
        batch_size: 每个批次的样本数
        num_steps: 每个样本的时间步数（序列长度）
    
    返回:
        生成器，每次返回一个批次的 (X, Y)，其中 Y 是 X 偏移一个位置的序列
    """
    # 从随机偏移量开始对序列进行分区，增加随机性
    corpus = corpus[random.randint(0, num_steps - 1):]
    
    # 计算可生成的子序列数量（减去1是因为需要考虑标签）
    num_subseqs = (len(corpus) - 1) // num_steps
    
    # 生成所有子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    
    # 随机打乱起始索引，实现随机采样
    random.shuffle(initial_indices)
    
    # 辅助函数：从指定位置返回长度为num_steps的序列
    def data(pos):
        return corpus[pos: pos + num_steps]
    
    # 计算总批次数量
    num_batches = num_subseqs // batch_size
    
    # 生成每个批次的数据
    for i in range(0, batch_size * num_batches, batch_size):
        # 当前批次的起始索引列表
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        
        # 生成输入序列X和目标序列Y（Y是X偏移1位的结果）
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    使用顺序分区生成小批量子序列，保持序列的连续性
    
    参数:
        corpus: 完整的语料库序列
        batch_size: 每个批次的样本数
        num_steps: 每个样本的时间步数（序列长度）
    
    返回:
        生成器，每次返回一个批次的 (X, Y)，其中 Y 是 X 偏移一个位置的序列
    """
    # 随机选择一个偏移量，增加一定随机性
    offset = random.randint(0, num_steps)
    
    # 计算有效 token 数量，确保能被 batch_size 整除
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    
    # 生成输入序列X和目标序列Y
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    
    # 重塑为 (batch_size, 序列长度) 的形状
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    
    # 计算总批次数量
    num_batches = Xs.shape[1] // num_steps
    
    # 按顺序生成每个批次
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


# 演示两种采样方法的效果
def demo_sampling_methods():
    """演示随机采样和顺序采样的区别"""
    # 创建一个简单的序列用于演示
    my_seq = list(range(35))
    print("\n演示随机采样:")
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y, '\n')
    
    print("演示顺序采样:")
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y, '\n')

# 执行采样方法演示
demo_sampling_methods()


# --------------------------
# 4. 数据加载器类
# --------------------------

class SeqDataLoader:
    """加载序列数据的迭代器类"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """
        初始化序列数据加载器
        
        参数:
            batch_size: 批次大小
            num_steps: 每个样本的时间步数
            use_random_iter: 是否使用随机采样，False则使用顺序采样
            max_tokens: 最大加载的token数量
        """
        # 根据选择使用不同的迭代函数
        self.data_iter_fn = seq_data_iter_random if use_random_iter else seq_data_iter_sequential
        
        # 加载语料库和词表（使用自定义的方法）
        raw_text = read_time_machine()
        tokens = d2l.tokenize(raw_text)
        corpus = [token for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        self.vocab = d2l.Vocab(corpus)
        self.corpus = [self.vocab[token] for token in corpus]
        
        self.batch_size = batch_size
        self.num_steps = num_steps

    def __iter__(self):
        """返回迭代器"""
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """
    返回时光机器数据集的迭代器和词表
    
    参数:
        batch_size: 批次大小
        num_steps: 每个样本的时间步数
        use_random_iter: 是否使用随机采样
        max_tokens: 最大加载的token数量
    
    返回:
        data_iter: 数据迭代器
        vocab: 词表
    """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# 演示代码：仅在直接运行该文件时执行
if __name__ == "__main__":
    # 加载数据并创建词表
    corpus, vocab = load_and_preprocess_data()

    # 查看词表中频率最高的10个词
    print("词表中频率最高的10个词：")
    print(vocab.token_freqs[:10])
    
    # 执行词频分析
    analyze_token_frequency(vocab, corpus)