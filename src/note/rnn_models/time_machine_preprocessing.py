"""时间机器数据集预处理模块

该模块实现了对《时间机器》(The Time Machine)文本数据集的读取、清洗、词元化和词表构建功能，主要用于自然语言处理任务的预处理阶段。

模块主要功能：
1. 数据集的自动下载与读取
2. 文本清洗（去除非字母字符、转换大小写等）
3. 词元化（支持单词级和字符级词元化）
4. 词表构建与管理
5. 语料库的加载与索引转换

核心组件：
- Vocab 类：用于构建和管理词表，支持词元到索引的双向映射
- 各种工具函数：实现数据读取、清洗、词元化和语料库加载等功能

使用场景：
- 自然语言处理模型的训练数据预处理
- 语言建模任务的基础数据准备
- 文本序列分析和预测

使用示例：
    # 作为模块导入使用
    from src.models.rnn_models.time_machine_preprocessing import load_corpus_time_machine, read_time_machine
    
    # 加载语料库和词表
    corpus, vocab = load_corpus_time_machine()
    
    # 读取原始文本行
    lines = read_time_machine()

主要函数和类：
- read_time_machine(): 读取并清洗《时间机器》文本
- tokenize(): 将文本行拆分为词元（单词或字符）
- count_corpus(): 统计词元出现频率
- Vocab 类: 词表管理类，处理词元到索引的映射
- load_corpus_time_machine(): 加载完整的语料库和对应的词表
"""
import collections
import re
from d2l import torch as d2l


# 数据集下载配置
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)


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


def tokenize(lines, token='word'):
    """
    将文本行拆分为词元（单词或字符）
    
    参数:
        lines: 文本行列表
        token: 词元类型，'word'表示按单词拆分，'char'表示按字符拆分
        
    返回:
        list: 词元化后的文本列表，每个元素为一行文本的词元列表
    """
    if token == 'word':
        # 按空格拆分单词
        return [line.split() for line in lines]
    elif token == 'char':
        # 按字符拆分
        return [list(line) for line in lines]
    else:
        raise ValueError(f'错误：未知词元类型：{token}')


def count_corpus(tokens):
    """
    统计词元在语料库中的出现频率
    
    参数:
        tokens: 词元列表，可以是1D列表或2D列表
        
    返回:
        collections.Counter: 词元频率计数器
    """
    # 如果是2D列表（每行一个列表），先展平为1D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """文本词表类，用于将词元映射到整数索引"""
    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        初始化词表
        
        参数:
            tokens: 词元列表
            min_freq: 最小出现频率，低于此频率的词元将被忽略
            reserved_tokens: 预留词元列表
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        # 按出现频率排序词元
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 初始化索引到词元的映射和词元到索引的映射
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        # 遍历词元频率，将满足频率要求的词元加入词表
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break  # 由于已排序，后续词元频率更低，可直接跳出
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        """返回词表大小"""
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        """
        将词元转换为索引
        
        参数:
            tokens: 单个词元或词元列表
            
        返回:
            单个索引或索引列表
        """
        if not isinstance(tokens, (list, tuple)):
            # 单个词元，未知词元返回<unk>的索引
            return self.token_to_idx.get(tokens, self.unk)
        # 词元列表，递归处理每个词元
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        """
        将索引转换为词元
        
        参数:
            indices: 单个索引或索引列表
            
        返回:
            单个词元或词元列表
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        """未知词元的索引（固定为0）"""
        return 0
    
    @property
    def token_freqs(self):
        """返回词元频率列表"""
        return self._token_freqs


def load_corpus_time_machine(max_tokens=-1):
    """
    加载时间机器数据集并返回词元索引列表和对应的词表
    
    参数:
        max_tokens: 最大词元数量，-1表示使用全部词元
        
    返回:
        tuple: (corpus, vocab)，其中corpus是词元索引列表，vocab是词表对象
    """
    # 读取文本行
    lines = read_time_machine()
    # 按字符级别进行词元化
    tokens = tokenize(lines, 'char')
    # 构建词表
    vocab = Vocab(tokens)
    # 将所有文本行展平为一个词元索引列表
    corpus = [vocab[token] for line in tokens for token in line]
    # 如果指定了最大词元数量，截断 corpus
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# 演示代码
if __name__ == "__main__":
    # 读取文本并展示示例
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print(f'第一行文本: {lines[0]}')
    print(f'第十一行文本: {lines[10]}\n')
    
    # 词元化演示
    tokens = tokenize(lines)
    print("前11行的词元化结果:")
    for i in range(11):
        print(tokens[i])
    print()
    
    # 词表构建演示
    vocab = Vocab(tokens)
    print("词表前10个词元及其索引:")
    print(list(vocab.token_to_idx.items())[:10])
    print()
    
    # 词元转索引演示
    for i in [0, 10]:
        print(f'文本: {tokens[i]}')
        print(f'索引: {vocab[tokens[i]]}\n')
    
    # 加载语料库和词表
    corpus, vocab = load_corpus_time_machine()
    print(f"语料库长度: {len(corpus)}, 词表大小: {len(vocab)}")
