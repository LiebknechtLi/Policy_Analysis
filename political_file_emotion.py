import json
import re
from tencentcloud.nlp.v20190408 import nlp_client, models
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

# 腾讯云API密钥
SECRET_ID = "your id"
SECRET_KEY = "your key"


# 初始化腾讯云NLP客户端
def get_nlp_client():
    cred = credential.Credential(SECRET_ID, SECRET_KEY)
    httpProfile = HttpProfile()
    httpProfile.endpoint = "nlp.tencentcloudapi.com"
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    return nlp_client.NlpClient(cred, "ap-guangzhou", clientProfile)


def analyze_sentiment(text, client):
    """调用腾讯云NLP API进行情感分析"""
    req = models.AnalyzeSentimentRequest()  # 使用AnalyzeSentiment请求方法
    params = {"Text": text}  # 只传递文本
    req.from_json_string(json.dumps(params))
    response = client.AnalyzeSentiment(req)  # 使用AnalyzeSentiment方法
    result = json.loads(response.to_json_string())

    positive_prob = result["Positive"]
    negative_prob = result["Negative"]
    sentiment_label = "positive" if positive_prob > negative_prob else "negative"

    return {
        "label": sentiment_label,
        "score": positive_prob  # 返回正面情感分数
    }


def split_text(text, max_length=512):
    """将文本按句子划分为小段"""
    sentences = re.split(r'(?<=。|！|？)', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len((current_chunk + sentence).encode('utf-8')) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def weighted_sentiment_analysis(text):
    """加权情感分析"""
    client = get_nlp_client()

    segments = split_text(text)
    if not segments:
        return {"label": "neutral", "score": 0.0}

    sentiment_scores = [analyze_sentiment(segment, client) for segment in segments]
    total_length = sum(len(segment) for segment in segments)
    weighted_score = 0.0

    for i, score in enumerate(sentiment_scores):
        segment_length = len(segments[i])
        weight = segment_length / total_length
        sentiment_value = 2 * score['score'] - 1 if score['label'] == 'positive' else -(2 * score['score'] - 1)
        weighted_score += sentiment_value * weight

    final_label = "positive" if weighted_score > 0 else "negative"
    return {"label": final_label, "score": weighted_score}


# 示例文本
text = input("Please paste your text: ")

# 进行加权情感分析
result = weighted_sentiment_analysis(text)
print(f"Final Classification: {result['label']}, Final Sentiment Score: {result['score']:.3f}")
