# 1. 需求分析
# 本案例旨在使用Informer时序模型对电力负荷数据进行预测。
# Informer是一种针对长序列时间序列预测(LSTF)任务设计的模型，
# 解决了传统Transformer在长序列预测中的计算效率问题。
# 我们将使用公开的电力负荷数据集，实现数据的获取、预处理、模型训练、预测、评估和可视化。

# 2. 导入依赖包
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import zipfile
from datetime import datetime

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 3. 获取数据
class DataGetter:
    """数据获取与缓存类，负责下载数据并缓存到本地"""
    
    def __init__(self, data_url, cache_dir='data_cache'):
        self.data_url = data_url
        self.cache_dir = cache_dir
        self.filename = os.path.basename(data_url)
        self.cache_path = os.path.join(cache_dir, self.filename)
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data(self):
        """获取数据，如果本地有缓存则使用缓存，否则下载"""
        if os.path.exists(self.cache_path):
            print(f"使用本地缓存数据: {self.cache_path}")
            return self._load_local_data()
        else:
            print(f"下载数据: {self.data_url}")
            self._download_data()
            return self._load_local_data()
    
    def _download_data(self):
        """下载数据并保存到缓存目录"""
        response = requests.get(self.data_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(self.cache_path, 'wb') as file, tqdm(
            desc=self.filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    
    def _load_local_data(self):
        """加载本地数据"""
        # 对于压缩文件进行解压
        if self.filename.endswith('.zip'):
            extract_dir = os.path.join(self.cache_dir, self.filename[:-4])
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(self.cache_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # 假设解压后只有一个csv文件
            csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
            if csv_files:
                return pd.read_csv(os.path.join(extract_dir, csv_files[0]))
            else:
                raise ValueError("解压后未找到CSV文件")
        elif self.filename.endswith('.csv'):
            return pd.read_csv(self.cache_path)
        else:
            raise ValueError(f"不支持的数据格式: {self.filename}")

# 使用公开的电力负荷数据集
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
data_getter = DataGetter(data_url)
df = data_getter.get_data()

# 4. 数据预处理
def preprocess_data(df):
    """数据预处理函数"""
    # 处理日期格式和分隔符
    df['date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop('Date', axis=1)
    
    # 选择一个客户的数据作为示例（数据集中有多个客户，我们选第一个）
    df = df[['date', df.columns[0]]].rename(columns={df.columns[0]: 'consumption'})
    
    # 转换为数值类型
    df['consumption'] = df['consumption'].str.replace(',', '.').astype(float)
    
    # 处理缺失值
    df = df.dropna()
    
    # 按小时采样（原始数据是15分钟一次）
    df = df.set_index('date').resample('H').mean().reset_index()
    
    # 筛选出部分数据以加快训练速度
    df = df[(df['date'] >= '2014-01-01') & (df['date'] <= '2014-06-30')]
    
    # 标准化
    scaler = StandardScaler()
    df['consumption_scaled'] = scaler.fit_transform(df['consumption'].values.reshape(-1, 1))
    
    return df, scaler

# 预处理数据
df, scaler = preprocess_data(df)

# 创建数据集类
class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len  # 输入序列长度
        self.pred_len = pred_len  # 预测序列长度
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # 输入序列
        x = self.data[idx:idx+self.seq_len]
        # 目标序列
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# 划分训练集和测试集
train_size = int(0.8 * len(df))
train_data = df['consumption_scaled'].values[:train_size]
test_data = df['consumption_scaled'].values[train_size:]

# 序列长度设置
seq_len = 96  # 4天(96小时)的历史数据
pred_len = 24  # 预测未来1天(24小时)

# 创建数据加载器
train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. 模型训练 - Informer模型实现
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class InformerEncoder(nn.Module):
    """Informer编码器"""
    def __init__(self, input_dim, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(InformerEncoder, self).__init__()
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 输入投影和位置编码
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # Transformer编码
        memory = self.transformer_encoder(src)
        return memory

class InformerDecoder(nn.Module):
    """Informer解码器"""
    def __init__(self, output_dim, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(InformerDecoder, self).__init__()
        self.d_model = d_model
        
        # 目标投影
        self.target_proj = nn.Linear(output_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer解码器层
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # 目标投影和位置编码
        tgt = self.target_proj(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)
        
        # Transformer解码
        output = self.transformer_decoder(tgt, memory)
        
        # 输出投影
        output = self.output_proj(output)
        return output

class Informer(nn.Module):
    """Informer模型"""
    def __init__(self, input_dim=1, output_dim=1, d_model=512, nhead=8, 
                 dim_feedforward=2048, num_layers=3, dropout=0.1):
        super(Informer, self).__init__()
        
        # 编码器
        self.encoder = InformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = InformerDecoder(
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, src, tgt):
        # 编码
        memory = self.encoder(src.unsqueeze(-1))  # 添加特征维度
        
        # 解码
        output = self.decoder(tgt.unsqueeze(-1), memory).squeeze(-1)  # 移除特征维度
        return output

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = Informer(
    input_dim=1,
    output_dim=1,
    d_model=128,
    nhead=4,
    dim_feedforward=512,
    num_layers=2,
    dropout=0.1
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=10, device=device):
    """训练模型"""
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        
        for i, (src, tgt) in progress_bar:
            src, tgt = src.to(device), tgt.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(src, tgt)
            loss = criterion(outputs, tgt)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.6f}')
    
    return model, train_losses

# 训练模型（这里使用较少的epochs以加快运行速度，实际应用中可以增加）
trained_model, train_losses = train_model(model, train_loader, criterion, optimizer, epochs=5)

# 6. 模型预测
def predict(model, test_loader, device=device):
    """使用模型进行预测"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # 预测时，解码器输入使用前一个时间步的预测结果
            # 初始输入使用编码器输入的最后一个值
            pred = torch.zeros_like(tgt)
            pred[:, 0] = src[:, -1]  # 第一个预测值基于最后一个输入值
            
            for i in range(1, pred_len):
                pred[:, i] = model(src, pred[:, :i+1])[:, i]
            
            predictions.extend(pred.cpu().numpy())
            actuals.extend(tgt.cpu().numpy())
    
    return np.array(predictions), np.array(actuals)

# 进行预测
predictions, actuals = predict(trained_model, test_loader)

# 还原标准化
predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
actuals_original = scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)

# 7. 模型评估
def evaluate_model(predictions, actuals):
    """评估模型性能"""
    # 计算MSE
    mse = mean_squared_error(actuals.flatten(), predictions.flatten())
    # 计算RMSE
    rmse = np.sqrt(mse)
    # 计算MAE
    mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
    # 计算MAPE
    mape = np.mean(np.abs((actuals.flatten() - predictions.flatten()) / actuals.flatten())) * 100
    
    print(f"评估指标:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

# 评估模型
metrics = evaluate_model(predictions_original, actuals_original)

# 8. 可视化
def plot_results(df, actuals, predictions, seq_len, pred_len, scaler):
    """可视化结果"""
    # 绘制训练损失
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # 选择一个样本进行可视化
    sample_idx = 10
    actual_sample = actuals[sample_idx]
    pred_sample = predictions[sample_idx]
    
    # 创建时间索引
    start_idx = train_size + seq_len + sample_idx
    time_index = df['date'][start_idx:start_idx+pred_len].values
    
    # 绘制预测结果与实际值对比
    plt.figure(figsize=(15, 6))
    plt.plot(time_index, actual_sample, label='实际值', color='blue')
    plt.plot(time_index, pred_sample, label='预测值', color='red', linestyle='--')
    plt.title('电力负荷预测结果')
    plt.xlabel('时间')
    plt.ylabel('电力负荷')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 绘制更长时间段的对比
    plt.figure(figsize=(15, 6))
    plt.plot(actuals.flatten()[:200], label='实际值', color='blue', alpha=0.7)
    plt.plot(predictions.flatten()[:200], label='预测值', color='red', linestyle='--', alpha=0.7)
    plt.title('电力负荷预测序列对比（前200小时）')
    plt.xlabel('时间步')
    plt.ylabel('电力负荷')
    plt.legend()
    plt.grid(True)
    plt.show()

# 可视化结果
plot_results(df, actuals_original, predictions_original, seq_len, pred_len, scaler)

print("时序预测任务完成！")