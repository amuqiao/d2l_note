# 导入依赖包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 加载数据
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame

# 数据预处理
# 查看数据基本信息
print("数据集基本信息：")
print(df.info())
print("\n数据集前5行：")
print(df.head())

# 选择相关特征和目标变量
# 选择有意义的特征
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

# 处理缺失值
print("\n缺失值统计：")
print(df[features].isnull().sum())

# 划分特征和目标变量
X = df[features]
y = df[target].astype('int')  # 将目标变量转换为整数

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 特征工程
# 区分数值特征和分类特征
numeric_features = ['age', 'sibsp', 'parch', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']

# 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 用中位数填充数值型缺失值
    ('scaler', StandardScaler())  # 标准化
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 用最频繁值填充分类缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
])

# 组合所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 构建决策树模型管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))  # 限制树深度防止过拟合
])

# 模型训练
print("\n开始训练模型...")
model.fit(X_train, y_train)

# 模型预测
print("\n进行预测...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 生存概率

# 模型评估
print("\n模型评估结果：")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n分类报告：")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵：")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('预测结果')
plt.ylabel('实际结果')
plt.title('混淆矩阵')
plt.show()

# ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (面积 = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()

# 特征重要性分析
# 获取特征名称
numeric_features_processed = numeric_features
categorical_features_processed = list(model.named_steps['preprocessor']
                                      .named_transformers_['cat']
                                      .named_steps['onehot']
                                      .get_feature_names_out(categorical_features))

all_features = numeric_features_processed + categorical_features_processed

# 获取特征重要性
importances = model.named_steps['classifier'].feature_importances_

# 排序并可视化
feature_importance = pd.DataFrame({'特征': all_features, '重要性': importances})
feature_importance = feature_importance.sort_values('重要性', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance)
plt.title('特征重要性')
plt.tight_layout()
plt.show()
