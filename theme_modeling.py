import json
import re
import numpy as np
import jieba
import jieba.analyse
import sys
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tencentcloud.nlp.v20190408 import nlp_client, models
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

# 设置标准输入输出的编码为utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# 腾讯云API密钥 (注意: 实际应用中应存储在环境变量或配置文件中)
SECRET_ID = "AKIDnOnzyBIF2c1HLIDyVotnV0KTcdR2QdeU"
SECRET_KEY = "g8RefV9dzA9i7NWSCkfR0MUP7c0CAvTo"

# 常用中文停用词列表
CHINESE_STOP_WORDS = [
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也',
    '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '中', '或',
    '与', '以', '及', '等', '为', '对', '由', '从'
]


# 初始化腾讯云NLP客户端
def get_nlp_client():
    try:
        cred = credential.Credential(SECRET_ID, SECRET_KEY)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "nlp.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        return nlp_client.NlpClient(cred, "ap-guangzhou", clientProfile)
    except Exception as e:
        print(f"初始化腾讯云客户端失败: {e}")
        return None


def split_text(text, max_length=512):
    """将文本按句子划分为小段"""
    sentences = re.split(r'(?<=。|！|？|；|\.|!|\?|;)', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        # 计算UTF-8编码的字节长度
        if len((current_chunk + sentence).encode('utf-8')) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # 如果没有分段成功，强制分段
    if not chunks and text:
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    return chunks


def preprocess_chinese_text(texts):
    """中文文本预处理：分词、去停用词"""
    segmented_texts = []
    for text in texts:
        # 使用jieba进行分词
        words = jieba.cut(text)
        # 过滤停用词
        filtered_words = [word for word in words if word not in CHINESE_STOP_WORDS and len(word.strip()) > 1]
        segmented_texts.append(' '.join(filtered_words))

    return segmented_texts


def extract_keywords_with_jieba(text, topK=20):
    """使用jieba提取关键词"""
    # 使用TF-IDF算法提取关键词
    keywords_tfidf = jieba.analyse.extract_tags(text, topK=topK, withWeight=True)

    # 使用TextRank算法提取关键词
    keywords_textrank = jieba.analyse.textrank(text, topK=topK, withWeight=True)

    # 合并两种算法的结果
    keywords = {}
    for word, weight in keywords_tfidf:
        keywords[word] = weight

    for word, weight in keywords_textrank:
        if word in keywords:
            # 取两种算法权重的平均值
            keywords[word] = (keywords[word] + weight) / 2
        else:
            keywords[word] = weight

    # 按权重排序
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_keywords[:topK]]


def chinese_lda_topic_modeling(texts, n_topics=5):
    """使用LDA进行中文主题建模"""
    # 预处理文本
    segmented_texts = preprocess_chinese_text(texts)

    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(
        max_features=2000,  # 限制特征数量
        min_df=1,  # 至少出现在2个文档中
        max_df=1.0  # 最多出现在90%的文档中
    )

    # 转换文本为TF-IDF矩阵
    try:
        tfidf_matrix = vectorizer.fit_transform(segmented_texts)

        # 检查矩阵是否为空
        if tfidf_matrix.shape[1] == 0:
            print("警告: TF-IDF矩阵为空，可能是因为文本太短或者过滤后没有足够的词语")
            return None, None

        # 应用LDA模型
        lda = LatentDirichletAllocation(
            n_components=min(n_topics, tfidf_matrix.shape[0], tfidf_matrix.shape[1]),
            random_state=42,
            max_iter=50
        )
        lda.fit(tfidf_matrix)

        return lda, vectorizer
    except ValueError as e:
        print(f"LDA建模错误: {e}")
        # 如果向量化失败，则使用jieba直接提取关键词
        all_text = ' '.join(texts)
        keywords = extract_keywords_with_jieba(all_text)
        print(f"使用jieba提取的关键词: {keywords}")
        return None, None


def get_topic_keywords(lda, vectorizer, n_top_words=10):
    """获取每个主题的关键词"""
    if lda is None or vectorizer is None:
        return {}

    try:
        feature_names = vectorizer.get_feature_names_out()
        topic_keywords = {}

        for topic_idx, topic in enumerate(lda.components_):
            # 获取每个主题的top关键词
            top_keywords_idx = topic.argsort()[:-n_top_words - 1:-1]
            top_keywords = [feature_names[i] for i in top_keywords_idx]
            topic_keywords[topic_idx] = top_keywords

        return topic_keywords
    except Exception as e:
        print(f"获取主题关键词失败: {e}")
        return {}


def calculate_relevance_score(text, target_keywords=None):
    """计算文本与目标主题的相关度分数"""
    if target_keywords is None:
        # 默认"民营经济发展"相关关键词
        target_keywords = ["民营", "经济", "发展", "企业", "创新", "支持", "政策", "扶持", "改革", "市场"]

    # 将文本分成小段
    segments = split_text(text)

    if not segments:
        print("警告: 文本分段后为空")
        return 0

    # 1. 使用jieba直接提取关键词
    text_keywords = extract_keywords_with_jieba(text)
    print(f"文本关键词: {text_keywords}")

    # 2. 尝试使用LDA主题模型
    lda_model, vectorizer = chinese_lda_topic_modeling(segments)
    topic_keywords = get_topic_keywords(lda_model, vectorizer)

    # 3. 计算关键词匹配分数
    # 3.1 直接关键词匹配
    direct_matches = set(target_keywords).intersection(text_keywords)
    direct_score = len(direct_matches) / len(target_keywords) * 10  # 转换为0-10的分数

    # 3.2 LDA主题关键词匹配
    lda_score = 0
    if topic_keywords:
        # 计算每个主题与目标关键词的匹配度
        topic_scores = []
        for topic_idx, keywords in topic_keywords.items():
            matches = set(target_keywords).intersection(keywords)
            topic_scores.append(len(matches) / len(target_keywords) * 10)

        # 取最高匹配度的主题分数
        if topic_scores:
            lda_score = max(topic_scores)

    # 3.3 计算最终分数 (直接匹配和LDA主题匹配的加权平均)
    final_score = 0.7 * direct_score + 0.3 * lda_score

    # 调试信息
    print(f"直接匹配分数: {direct_score}")
    print(f"LDA主题匹配分数: {lda_score}")
    print(f"关键词匹配项: {direct_matches}")

    return round(final_score)


def analyze_chinese_topic_relevance(text, target_topic="民营经济发展"):
    """分析中文文本与特定主题的相关度"""
    if not text or len(text.strip()) == 0:
        print("错误: 输入文本为空")
        return {"relevance_score": 0}

    # 根据目标主题设置相关关键词
    topic_keywords_map = {
        "民营经济发展": ["民营", "经济", "发展", "企业", "创新", "支持", "政策", "扶持", "改革", "市场"],
        "科技创新": ["科技", "创新", "研发", "技术", "人才", "突破", "数字", "智能", "现代化", "核心"],
        "乡村振兴": ["乡村", "振兴", "农业", "农村", "农民", "现代化", "产业", "生态", "文化", "组织"]
    }

    # 获取目标主题关键词
    target_keywords = topic_keywords_map.get(target_topic, topic_keywords_map["民营经济发展"])

    # 计算相关度分数
    relevance_score = calculate_relevance_score(text, target_keywords)

    return {"relevance_score": relevance_score}


if __name__ == "__main__":
    try:
        # 获取用户输入
        print("请输入要分析的中文文本:")
        text = input()

        # 分析文本与"民营经济发展"主题的相关度
        result = analyze_chinese_topic_relevance(text)

        # 输出结果
        print(f"\n最终相关度分数: {result['relevance_score']}")

        # 如果分数为0，给出提示
        if result['relevance_score'] == 0:
            print("\n注意: 相关度分数为0，可能是因为:")
            print("1. 文本内容与目标主题无关")
            print("2. 文本太短，无法提取有效关键词")
            print("3. 中文分词或关键词提取出现问题")

    except Exception as e:
        print(f"\n程序运行出错: {e}")

    finally:
        print("\n进程已结束，退出代码为 0")