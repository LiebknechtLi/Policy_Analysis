from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# 加载中文情感分析模型
model_name = "uer/roberta-base-finetuned-jd-binary-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def analyze_sentiment(text):
    """对单个文本段落进行情感分析"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    # 假设标签0为负面，标签1为正面
    positive_prob = probabilities[0, 1].item()

    # 转换为-1到1的分数
    score = 2 * positive_prob - 1

    return {
        "label": "positive" if positive_prob > 0.5 else "negative",
        "score": positive_prob
    }


def split_text(text, max_length=512):
    """将文本按句子划分为小段"""
    sentences = re.split(r'(?<=。|！|？)', text)  # 按句号等标点符号划分句子
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence:  # 跳过空句子
            continue
        # 考虑tokenizer的token长度而非字符长度
        if len(tokenizer.encode(current_chunk + sentence)) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence
    if current_chunk:
        chunks.append(current_chunk.strip())  # 添加剩余部分

    return chunks


def weighted_sentiment_analysis(text):
    """对输入文本进行情感分析，采用加权平均计算最终情感值"""
    # 1. 将文本划分为多个语段
    segments = split_text(text)

    if not segments:
        return 0.0  # 空文本返回中性分数

    # 2. 对每个语段进行情感分析
    sentiment_scores = []
    for segment in segments:
        result = analyze_sentiment(segment)
        sentiment_scores.append(result)

    # 3. 计算加权情感分数
    total_length = sum(len(segment) for segment in segments)
    weighted_score = 0.0
    for i, score in enumerate(sentiment_scores):
        segment_length = len(segments[i])
        weight = segment_length / total_length

        # 从0-1范围转换到-1到1范围
        sentiment_value = 2 * score['score'] - 1 if score['label'] == 'positive' else -(2 * score['score'] - 1)
        weighted_score += sentiment_value * weight

    return weighted_score


# 示例文本
text = "今年中国经济形势复杂，政策加码，发展充满挑战。政府加强对创新的支持，民营企业迎来新的机会。"

# 进行加权情感分析
final_score = weighted_sentiment_analysis(text)
print(f"最终情感分数: {final_score:.3f}")