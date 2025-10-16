# 顾客数据分析案例：使用K-means聚类算法

## 1. 需求分析
# 本案例旨在通过K-means聚类算法对顾客数据进行分析，
# 将顾客划分为不同群体，帮助企业了解客户特征，
# 制定针对性的营销策略。分析将基于顾客的消费金额、
# 消费频率和平均消费间隔等特征进行。

## 2. 导入依赖包
import os
import warnings

# 设置环境变量以避免OpenMP冲突和KMeans内存泄漏
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复的OpenMP库

# 忽略特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='KMeans is known to have a memory leak')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
sns.set(font='SimHei', font_scale=1.2)

## 3. 获取数据
# 生成模拟的顾客数据，包含4个特征：
# 1. 年度消费金额
# 2. 消费频率（每年消费次数）
# 3. 平均消费间隔（天）
# 4. 平均每次消费金额

# 生成5个聚类中心的模拟数据
X, y_true = make_blobs(n_samples=500, n_features=4, centers=5, 
                       cluster_std=0.60, random_state=0)

# 将数据转换为DataFrame，赋予实际业务含义
customer_features = ['年度消费金额', '消费频率', '平均消费间隔', '平均每次消费金额']
df = pd.DataFrame(X, columns=customer_features)

# 调整数据使其更符合实际业务场景
df['年度消费金额'] = df['年度消费金额'].apply(lambda x: abs(x)*500 + 1000)  # 1000-10000元
df['消费频率'] = df['消费频率'].apply(lambda x: abs(int(x)*3 + 5))  # 5-30次/年
df['平均消费间隔'] = df['平均消费间隔'].apply(lambda x: abs(int(x)*7 + 15))  # 15-100天
df['平均每次消费金额'] = df['平均每次消费金额'].apply(lambda x: abs(x)*200 + 100)  # 100-2000元

# 查看数据基本信息
print("数据集基本信息：")
print(df.info())
print("\n数据集前5行：")
print(df.head())
print("\n数据集统计描述：")
print(df.describe())

## 4. 数据预处理
# 数据标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 将标准化后的数据转换为DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=customer_features)

# 查看标准化后的数据
print("\n标准化后的数据集前5行：")
print(df_scaled.head())

# 绘制特征之间的相关性热力图
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()

## 5. 模型训练
# 使用肘部法确定最佳K值
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('聚类数量 (K)')
plt.ylabel('惯性 (Inertia)')
plt.title('肘部法确定最佳K值')
plt.grid(True)
plt.show()

# 基于肘部图，我们选择K=5进行最终聚类
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(df_scaled)

## 6. 模型预测
# 获取聚类结果
df['聚类标签'] = kmeans.labels_
df_scaled['聚类标签'] = kmeans.labels_

# 查看各聚类的数量
print("\n各聚类的顾客数量：")
print(df['聚类标签'].value_counts().sort_index())

## 7. 模型评估
# 计算轮廓系数（越接近1越好）
silhouette_avg = silhouette_score(df_scaled.drop('聚类标签', axis=1), df['聚类标签'])
print(f"\n轮廓系数: {silhouette_avg:.4f}")

# 计算Calinski-Harabasz指数（值越大越好）
calinski_score = calinski_harabasz_score(df_scaled.drop('聚类标签', axis=1), df['聚类标签'])
print(f"Calinski-Harabasz指数: {calinski_score:.4f}")

# 计算Davies-Bouldin指数（越接近0越好）
davies_score = davies_bouldin_score(df_scaled.drop('聚类标签', axis=1), df['聚类标签'])
print(f"Davies-Bouldin指数: {davies_score:.4f}")

## 8. 展示聚类效果

# 8.1 各聚类的特征均值分析
cluster_analysis = df.groupby('聚类标签').mean()
print("\n各聚类的特征均值：")
print(cluster_analysis)

# 绘制各聚类的特征雷达图
plt.figure(figsize=(12, 10))
categories = customer_features
N = len(categories)

# 计算角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# 绘制每个聚类的雷达图
for i in range(best_k):
    values = cluster_analysis.iloc[i].values.flatten().tolist()
    values += values[:1]
    plt.polar(angles, values, 'o-', linewidth=2, label=f'聚类 {i}')
    plt.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], categories)
plt.yticks([])
plt.title('各聚类的特征雷达图')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.tight_layout()
plt.show()

# 8.2 使用PCA降维可视化聚类结果
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled.drop('聚类标签', axis=1))
pca_df = pd.DataFrame(data=principal_components, columns=['主成分1', '主成分2'])
pca_df['聚类标签'] = df['聚类标签']

# 解释方差比例
print(f"\nPCA主成分解释方差比例: {pca.explained_variance_ratio_}")

# 绘制PCA降维后的聚类散点图
plt.figure(figsize=(12, 8))
sns.scatterplot(x='主成分1', y='主成分2', hue='聚类标签', data=pca_df, 
                palette='viridis', s=100, alpha=0.7, style='聚类标签', markers=['o', 's', 'D', '^', 'P'])
plt.title('PCA降维后的顾客聚类结果')
plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
plt.grid(True, alpha=0.3)
plt.legend(title='聚类标签')
plt.tight_layout()
plt.show()

# 8.3 各特征的聚类分布箱线图
plt.figure(figsize=(16, 12))
for i, feature in enumerate(customer_features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='聚类标签', y=feature, data=df, hue='聚类标签', palette='viridis', legend=False)
    plt.title(f'{feature}的聚类分布')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. 聚类结果分析与结论
print("\n===== 聚类结果分析 =====")
for i in range(best_k):
    cluster = cluster_analysis.iloc[i]
    print(f"\n聚类 {i} 特征分析:")
    print(f"  年度消费金额: {cluster['年度消费金额']:.2f} 元")
    print(f"  消费频率: {cluster['消费频率']:.2f} 次/年")
    print(f"  平均消费间隔: {cluster['平均消费间隔']:.2f} 天")
    print(f"  平均每次消费金额: {cluster['平均每次消费金额']:.2f} 元")
    
    # 根据特征给出客户群体描述
    if cluster['年度消费金额'] > df['年度消费金额'].mean() * 1.2 and cluster['消费频率'] > df['消费频率'].mean() * 1.2:
        print("  客户类型: 高价值忠诚客户")
    elif cluster['年度消费金额'] > df['年度消费金额'].mean() * 1.2 and cluster['消费频率'] < df['消费频率'].mean() * 0.8:
        print("  客户类型: 高价值低频客户")
    elif cluster['年度消费金额'] < df['年度消费金额'].mean() * 0.8 and cluster['消费频率'] > df['消费频率'].mean() * 1.2:
        print("  客户类型: 低价值高频客户")
    elif cluster['平均消费间隔'] > df['平均消费间隔'].mean() * 1.5:
        print("  客户类型: 沉睡客户")
    else:
        print("  客户类型: 普通客户")
