import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import string
import random
import os
import ssl

# 解决SSL证书问题（国内环境常见问题）
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 设置nltk数据目录（使用用户目录下的nltk_data）
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)  # 添加自定义数据目录

# 国内镜像源列表
mirror_urls = [
    "https://mirrors.tuna.tsinghua.edu.cn/nltk_data/",
    "https://mirrors.aliyun.com/nltk_data/",
    "https://nltk.github.io/nltk_data/"  # 官方源作为最后尝试
]

# 从国内镜像下载所需资源
def download_with_mirror(package):
    try:
        # 尝试加载包，如果已存在则不下载
        nltk.data.find(f"corpora/{package}")
        print(f"{package} 已存在，无需下载")
        return True
    except LookupError:
        print(f"尝试下载 {package}...")
        # 尝试多个镜像源
        from nltk.downloader import Downloader
        for mirror in mirror_urls:
            try:
                print(f"尝试使用镜像: {mirror}")
                downloader = Downloader()
                downloader._update_index(mirror)
                if downloader.download(
                    package,
                    download_dir=nltk_data_dir,
                    quiet=False
                ):
                    print(f"从 {mirror} 成功下载 {package}")
                    return True
            except Exception as e:
                print(f"使用 {mirror} 下载失败: {str(e)}")
                continue
        
        # 如果所有镜像都失败，尝试直接下载方式
        try:
            print("尝试使用nltk默认下载方式")
            nltk.download(package, download_dir=nltk_data_dir)
            return True
        except Exception as e:
            print(f"所有下载方式都失败: {str(e)}")
            print(f"请手动下载 {package} 并放置到 {nltk_data_dir} 目录下")
            return False

# 下载必要的数据集和资源
download_with_mirror('stopwords')
download_with_mirror('movie_reviews')

# 验证资源是否下载成功
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/movie_reviews')
except LookupError as e:
    print(f"错误：必要的NLTK资源未找到 - {e}")
    print("请手动下载所需资源后再运行程序")
    exit(1)

# 获取数据 - 使用nltk的电影评论数据集
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# 提取文本和标签
texts = [' '.join(words) for words, _ in documents]
labels = [1 if category == 'pos' else 0 for _, category in documents]

# 打印数据集前5行示例
print("\n数据集前5行示例:")
for i in range(min(5, len(texts))):
    print(f"\n样本 {i+1}:")
    print(f"情感标签: {'正面评论' if labels[i] == 1 else '负面评论'}")
    # 只显示前100个字符，避免输出过长
    print(f"文本内容: {texts[i][:100]}...")

# 数据预处理
stop_words = set(stopwords.words('english') + list(string.punctuation))

def preprocess_text(text):
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words and not word.isdigit()]
    return ' '.join(words)

processed_texts = [preprocess_text(text) for text in texts]

# 特征提取
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(processed_texts).toarray()
y = np.array(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练与预测
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['负面评论', '正面评论']))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 预测函数
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(text_vector)
    return "正面评论" if prediction[0] == 1 else "负面评论"

# 测试示例
test_sentences = [
    "This product is amazing, I really love it!",
    "Terrible experience, would not recommend to anyone.",
    "It works well and has great features.",
    "Poor quality, broke after one use."
]

print("\n预测示例:")
for sentence in test_sentences:
    print(f"句子: {sentence}")
    print(f"预测结果: {predict_sentiment(sentence)}\n")
