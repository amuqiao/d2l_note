import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 加载手写数字数据集
digits = load_digits()
X = digits.data  # 特征数据：64维向量
y = digits.target  # 标签：0-9

# 查看数据集信息
print(f"数据集特征形状: {X.shape}")
print(f"数字类别: {np.unique(y)}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集时保留原始索引（关键修复）
indices = np.arange(len(X))  # 保存原始数据的索引
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, indices, test_size=0.3, random_state=42, stratify=y
)

# 初始化KNN分类器
knn = KNeighborsClassifier()

# 定义超参数搜索范围
param_grid = {
    'n_neighbors': np.arange(1, 16),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# 网格搜索
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 预测与评估
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
print(f"\n测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# 打印混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 混淆矩阵可视化
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), 
            yticklabels=np.unique(y))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('手写数字识别混淆矩阵')
plt.show()

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化部分测试样本（修复索引问题）
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # 使用原始索引获取图像（关键修复）
    original_idx = idx_test[i]  # 从保存的测试集索引中获取原始数据索引
    plt.imshow(digits.images[original_idx],  # 直接用原始索引访问图像
               cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"预测: {y_pred[i]}\n真实: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 使用joblib保存训练好的模型和标准化器到脚本目录
print("\n保存模型和标准化器...")
model_path = os.path.join(script_dir, 'knn_digit_recognizer.pkl')
scaler_path = os.path.join(script_dir, 'digit_scaler.pkl')
joblib.dump(best_knn, model_path)
joblib.dump(scaler, scaler_path)
print(f"模型已保存到: {model_path}")
print(f"标准化器已保存到: {scaler_path}")


# 加载模型和标准化器进行预测的示例
def predict_with_saved_model(sample_images):
    """使用保存的模型和标准化器进行预测的示例函数"""
    # 获取脚本所在目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建模型和标准化器的完整路径
    model_path = os.path.join(current_script_dir, 'knn_digit_recognizer.pkl')
    scaler_path = os.path.join(current_script_dir, 'digit_scaler.pkl')
    
    # 加载模型和标准化器
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    
    # 预处理输入样本
    # 将图像展平为一维数组
    flattened_samples = [img.flatten() for img in sample_images]
    # 使用加载的标准化器进行特征缩放
    scaled_samples = loaded_scaler.transform(flattened_samples)
    
    # 进行预测
    predictions = loaded_model.predict(scaled_samples)
    probabilities = loaded_model.predict_proba(scaled_samples) if hasattr(loaded_model, 'predict_proba') else None
    
    return predictions, probabilities


# 演示如何使用保存的模型进行预测
print("\n演示加载模型进行预测...")
# 从测试集中选取几个样本作为示例
num_demo_samples = 5
# 获取原始图像数据
# 注意：在实际应用中，这里应该是新的未见过的数字图像
# 在这个演示中，我们只是从测试集中选择一些样本

demo_indices = idx_test[:num_demo_samples]  # 从测试集中选择几个样本的原始索引
demo_images = [digits.images[idx] for idx in demo_indices]  # 获取原始图像
demo_true_labels = [digits.target[idx] for idx in demo_indices]  # 获取真实标签

# 使用保存的模型进行预测
predictions, probabilities = predict_with_saved_model(demo_images)

# 显示预测结果
print("预测结果：")
for i in range(num_demo_samples):
    print(f"样本 {i+1}: 预测为 {predictions[i]}，真实标签为 {demo_true_labels[i]}")
    if probabilities is not None:
        confidence = np.max(probabilities[i])
        print(f"  置信度: {confidence:.4f}")

# 可视化预测结果
plt.figure(figsize=(12, 5))
for i in range(num_demo_samples):
    plt.subplot(1, num_demo_samples, i+1)
    plt.imshow(demo_images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"预测: {predictions[i]}\n真实: {demo_true_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
    