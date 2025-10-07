import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import random

# 导入项目中的matplotlib字体设置工具
try:
    from src.helper_utils.matplotlib_tools import setup_matplotlib_font
    # 设置matplotlib字体，确保中文能正确显示
    setup_matplotlib_font()
except ImportError:
    # 如果导入失败，手动设置字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 数据准备
# 生成模拟数据，创建4个聚类
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 使用sklearn的KMeans算法
# 创建并训练KMeans模型
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

# 获取聚类中心
centroids = kmeans.cluster_centers_

# 3. 自定义KMeans算法实现（用于理解原理）
class SimpleKMeans:
    def __init__(self, n_clusters=4, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None  # 用于评估模型
        
    def fit(self, X):
        # 设置随机种子
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            
        # 随机选择初始聚类中心
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()
        
        # 迭代优化
        for i in range(self.max_iter):
            # 计算每个样本到各聚类中心的距离，并分配到最近的聚类中心
            distances = np.zeros((n_samples, self.n_clusters))
            for j in range(self.n_clusters):
                distances[:, j] = np.sqrt(np.sum((X - self.centroids[j]) ** 2, axis=1))
            
            # 分配样本到最近的聚类中心
            labels = np.argmin(distances, axis=1)
            
            # 计算新的聚类中心
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                if np.sum(labels == j) > 0:
                    new_centroids[j] = np.mean(X[labels == j], axis=0)
                else:
                    # 如果某个聚类没有样本，随机选择一个样本作为中心
                    new_centroids[j] = X[np.random.choice(n_samples)]
            
            # 计算聚类中心的变化量
            centroid_shift = np.sum(np.sqrt(np.sum((new_centroids - self.centroids) ** 2, axis=1)))
            
            # 更新聚类中心
            self.centroids = new_centroids.copy()
            
            # 计算惯性（Inertia）
            self.inertia_ = np.sum([np.sum((X[labels == j] - self.centroids[j]) ** 2) for j in range(self.n_clusters)])
            
            # 如果变化量小于阈值，停止迭代
            if centroid_shift < self.tol:
                break
        
        return self
        
    def predict(self, X):
        # 预测新样本的聚类
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        for j in range(self.n_clusters):
            distances[:, j] = np.sqrt(np.sum((X - self.centroids[j]) ** 2, axis=1))
        
        return np.argmin(distances, axis=1)

# 使用自定义KMeans实现
simple_kmeans = SimpleKMeans(n_clusters=4, random_state=42)
simple_kmeans.fit(X_scaled)
simple_y_pred = simple_kmeans.predict(X_scaled)

# 4. 模型评估
# 计算轮廓系数（Silhouette Coefficient）
silhouette_avg = silhouette_score(X_scaled, y_pred)
# 计算Davies-Bouldin指数
db_score = davies_bouldin_score(X_scaled, y_pred)

print("模型评估结果：")
print(f"轮廓系数: {silhouette_avg:.4f}")
print(f"Davies-Bouldin指数: {db_score:.4f}")
print(f"Sklearn KMeans 惯性值: {kmeans.inertia_:.4f}")
print(f"自定义KMeans 惯性值: {simple_kmeans.inertia_:.4f}")

# 5. 可视化聚类结果
plt.figure(figsize=(15, 5))

# 原始数据可视化
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], s=50, c=y_true, cmap='viridis')
plt.title('原始数据分布')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.grid(True, linestyle='--', alpha=0.7)

# sklearn KMeans 结果可视化
plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], s=50, c=y_pred, cmap='viridis')
# 将标准化后的聚类中心转换回原始数据空间进行可视化
original_centroids = scaler.inverse_transform(centroids)
plt.scatter(original_centroids[:, 0], original_centroids[:, 1], s=200, c='red', marker='X', label='聚类中心')
plt.title(f'sklearn KMeans 聚类结果 (轮廓系数: {silhouette_avg:.3f})')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 自定义KMeans 结果可视化
plt.subplot(133)
plt.scatter(X[:, 0], X[:, 1], s=50, c=simple_y_pred, cmap='viridis')
# 将标准化后的聚类中心转换回原始数据空间进行可视化
custom_original_centroids = scaler.inverse_transform(simple_kmeans.centroids)
plt.scatter(custom_original_centroids[:, 0], custom_original_centroids[:, 1], s=200, c='red', marker='X', label='聚类中心')
plt.title('自定义KMeans 聚类结果')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
# 获取脚本所在路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 保存图片到脚本所在路径
plt.savefig(os.path.join(script_dir, 'kmeans_clustering_result.png'))
plt.show()

# 6. 肘部法则选择最佳K值
def plot_elbow_method(X, max_k=10):
    inertia = []
    silhouette_scores = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        
        if k > 1:  # 轮廓系数需要至少2个聚类
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    plt.figure(figsize=(12, 5))
    
    # 肘部法则图
    plt.subplot(121)
    plt.plot(range(1, max_k + 1), inertia, 'bo-')
    plt.title('肘部法则选择最佳K值')
    plt.xlabel('聚类数K')
    plt.ylabel('惯性值(Inertia)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 轮廓系数图
    plt.subplot(122)
    plt.plot(range(2, max_k + 1), silhouette_scores, 'ro-')
    plt.title('轮廓系数选择最佳K值')
    plt.xlabel('聚类数K')
    plt.ylabel('轮廓系数')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # 获取脚本所在路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 保存图片到脚本所在路径
    plt.savefig(os.path.join(script_dir, 'kmeans_elbow_method.png'))
    plt.show()

# 运行肘部法则分析
print("\n运行肘部法则分析以选择最佳聚类数...")
plot_elbow_method(X_scaled, max_k=8)