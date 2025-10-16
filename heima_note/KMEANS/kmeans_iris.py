import os
import warnings
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', message='The figure layout has changed to tight')

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 数据加载与查看
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
# 将英文特征名称转换为中文
chinese_feature_names = {
    'sepal length (cm)': '花萼长度 (cm)',
    'sepal width (cm)': '花萼宽度 (cm)',
    'petal length (cm)': '花瓣长度 (cm)',
    'petal width (cm)': '花瓣宽度 (cm)'
}
# 创建中文列名的DataFrame
df = pd.DataFrame(X, columns=[chinese_feature_names[name] for name in feature_names])
target_names = iris.target_names
df['target'] = [target_names[i] for i in y]
print("数据集前5行:\n", df.head())

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 确定最佳K值（含图片保存）
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
# 手肘图
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('聚类数量 (k)')
plt.ylabel('惯性值')
plt.title('手肘法确定最佳k值')
# 轮廓系数图
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('聚类数量 (k)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数法确定最佳k值')
plt.tight_layout()
# 保存图片到脚本目录
plt.savefig(os.path.join(script_dir, "kmeans_elbow_silhouette.png"), dpi=300, bbox_inches='tight')
plt.show()  # 显示图片

# 4. 模型训练与预测
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
kmeans.fit(X_scaled)
y_pred = kmeans.predict(X_scaled)
df['cluster'] = y_pred

# 5. 模型评估
print(f"\n轮廓系数: {silhouette_score(X_scaled, y_pred):.4f}")
print(f"调整兰德指数: {adjusted_rand_score(y, y_pred):.4f}")
print("\n交叉表:\n", pd.crosstab(df['target'], df['cluster'], rownames=['真实标签'], colnames=['聚类结果']))

# 6. PCA可视化（含图片保存）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = [target_names[i] for i in y]
df_pca['cluster'] = y_pred

plt.figure(figsize=(12, 5))
# 聚类结果图
plt.subplot(1, 2, 1)
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_pca, palette='viridis', s=100, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='质心')
plt.title('K-means聚类结果 (k=3)')
plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.legend()
# 真实标签图
plt.subplot(1, 2, 2)
sns.scatterplot(x='PC1', y='PC2', hue='target', data=df_pca, palette='Set1', s=100, alpha=0.7)
plt.title('真实标签分布')
plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.legend()
plt.tight_layout()
# 保存图片到脚本目录
plt.savefig(os.path.join(script_dir, "kmeans_pca_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()  # 显示图片

# 7. 特征散点图（含图片保存和显示修复）
# 创建pairplot，使用中文特征名称
chinese_feature_list = [chinese_feature_names[name] for name in feature_names]
g = sns.pairplot(df, hue='cluster', palette='viridis', vars=chinese_feature_list, diag_kind='kde')
g.fig.suptitle('按聚类结果着色的特征散点图矩阵', y=1.02)  # 调整标题位置

# 保存图片
plt.savefig(os.path.join(script_dir, "kmeans_pairplot.png"), dpi=300, bbox_inches='tight')

# 修复显示问题：明确获取图形并显示
plt.figure(g.fig.number)  # 激活pairplot的图形
plt.show()  # 显示图片

# 关闭图形以释放资源
plt.close('all')
