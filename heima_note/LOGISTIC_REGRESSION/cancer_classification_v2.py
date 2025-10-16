import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 加载数据
cancer = load_breast_cancer()

# 转换为DataFrame以便查看
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target
df['target_name'] = df['target'].map({0: '恶性', 1: '良性'})

# 数据基本信息
print("数据集形状：", df.shape)
print("\n数据集前5行：")
print(df.head())

# 缺失值处理
print("\n缺失值情况：")
print(df.isnull().sum())

# 确定特征值和目标值
X = cancer.data  # 特征值
y = cancer.target  # 目标值，0表示恶性，1表示良性

# 查看类别分布
print("\n类别分布：", np.bincount(y))
print("类别含义：", dict(zip([0, 1], ['恶性', '良性'])))

# 1. 数据可视化 - 目标变量分布
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='target_name', hue='target_name', data=df, palette=['#ff9999','#66b3ff'], legend=False)
plt.title('乳腺癌类型分布', fontsize=15)
plt.xlabel('肿瘤类型', fontsize=12)
plt.ylabel('数量', fontsize=12)

# 添加数据标签
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# 2. 数据可视化 - 特征相关性分析
# 选取10个重要特征进行相关性分析
# 确保包含target列进行相关性计算
top_features = df[cancer.feature_names.tolist() + ['target']].corr()['target'].abs().sort_values(ascending=False).index[1:11]
corr_matrix = df[top_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('特征相关性热图', fontsize=15)
plt.tight_layout()
plt.show()

# 3. 数据可视化 - 重要特征的分布
# 选取4个最重要的特征，绘制其在不同类别中的分布
plt.figure(figsize=(16, 12))
top4_features = top_features[:4]

for i, feature in enumerate(top4_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=feature, hue='target_name', kde=True, 
                 element='step', palette=['#ff9999','#66b3ff'])
    plt.title(f'{feature}在不同肿瘤类型中的分布', fontsize=13)
    plt.xlabel(feature, fontsize=11)
    plt.ylabel('频数', fontsize=11)

plt.tight_layout()
plt.show()

# 4. 数据可视化 - 箱线图比较
plt.figure(figsize=(16, 12))

for i, feature in enumerate(top4_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='target_name', y=feature, data=df, hue='target_name', palette=['#ff9999','#66b3ff'], legend=False)
    plt.title(f'{feature}在不同肿瘤类型中的分布', fontsize=13)
    plt.xlabel('肿瘤类型', fontsize=11)
    plt.ylabel(feature, fontsize=11)

plt.tight_layout()
plt.show()

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 特征工程：标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练：逻辑回归
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 模型预测
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]  # 预测为良性的概率

# 模型评估
print("\n===== 模型评估结果 =====")
# 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 精确率
precision = precision_score(y_test, y_pred)
print(f"精确率: {precision:.4f}")

# 召回率
recall = recall_score(y_test, y_pred)
print(f"召回率: {recall:.4f}")

# F1分数
f1 = f1_score(y_test, y_pred)
print(f"F1分数: {f1:.4f}")

# AUC值
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC值: {auc_score:.4f}")

# 5. 模型评估可视化 - 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['恶性', '良性'],
            yticklabels=['恶性', '良性'])
plt.title('混淆矩阵', fontsize=15)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('实际标签', fontsize=12)
plt.tight_layout()
plt.show()

# 6. 模型评估可视化 - ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('受试者工作特征曲线 (ROC)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['恶性', '良性']))

# 查看特征重要性
feature_importance = pd.DataFrame({
    '特征': cancer.feature_names,
    '系数': log_reg.coef_[0]
})
feature_importance['绝对值'] = np.abs(feature_importance['系数'])
feature_importance = feature_importance.sort_values('绝对值', ascending=False)

print("\n最重要的5个特征:")
print(feature_importance.head(5))

# 7. 特征重要性可视化
plt.figure(figsize=(12, 8))
top10_features = feature_importance.head(10)
sns.barplot(x='系数', y='特征', data=top10_features, palette='coolwarm', hue='特征', legend=False)
plt.title('特征重要性（逻辑回归系数）', fontsize=15)
plt.xlabel('系数值', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
