# 导入依赖包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import graphviz
from sklearn.tree import export_graphviz

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 获取数据
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame

# 数据探索
print("数据集基本信息：")
print(df.info())
print("\n数据集前5行：")
print(df.head())
print("\n生存情况统计：")
print(df['survived'].value_counts())

# 数据预处理
# 选择相关特征
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

# 处理缺失值
df = df[features + [target]].dropna()

# 划分特征和目标变量
X = df[features]
y = df[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义特征预处理
numeric_features = ['age', 'sibsp', 'parch', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# 模型训练：创建并训练决策树模型
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
])

model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 模型评估
print("\n模型评估结果：")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n混淆矩阵：")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['未幸存', '幸存'], 
            yticklabels=['未幸存', '幸存'])
plt.xlabel('预测结果')
plt.ylabel('实际结果')
plt.title('混淆矩阵')
plt.show()

print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 决策树可视化
# 提取预处理后的特征名称
try:
    numeric_transformer = preprocessor.named_transformers_['num']
    categorical_transformer = preprocessor.named_transformers_['cat']

    categorical_feature_names = list(categorical_transformer.get_feature_names_out(categorical_features))
    feature_names = numeric_features + categorical_feature_names

    # 获取脚本所在目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 使用plot_tree可视化
    plt.figure(figsize=(20, 10))
    plot_tree(
        model.named_steps['classifier'],
        feature_names=feature_names,
        class_names=['未幸存', '幸存'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('泰坦尼克号生存预测决策树')
    plt.tight_layout()
    # 保存到脚本目录
    png_output_path = os.path.join(script_dir, 'titanic_decision_tree_plot.png')
    plt.savefig(png_output_path, dpi=300)
    plt.show()
    print(f"\n决策树已保存为脚本目录下的{png_output_path}文件")

    # 尝试使用graphviz生成更清晰的可视化（带异常处理）
    try:
        output_file = os.path.join(script_dir, "titanic_decision_tree")
        
        dot_data = export_graphviz(
            model.named_steps['classifier'],
            out_file=None,
            feature_names=feature_names,
            class_names=['未幸存', '幸存'],
            filled=True,
            rounded=True,
            special_characters=True,
            fontname='SimHei'  # 支持中文
        )

        graph = graphviz.Source(dot_data)
        graph.render(output_file)
        print(f"\n决策树已保存为脚本目录下的{output_file}.pdf文件")
    except Exception as e:
        print(f"\n注意：无法使用Graphviz生成PDF可视化。错误信息：{str(e)}")
        print("请安装Graphviz并将其添加到系统PATH中，或使用已生成的PNG图像。")
except Exception as e:
    print(f"\n可视化过程中发生错误：{str(e)}")
