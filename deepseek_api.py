import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import json

# 初始化语义相似度模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 定义新的 prompt 模板
PROMPT_TEMPLATE = """你是一个知识渊博且严谨的助手。以下是一个问题和相关文本：
【问题】：
{question}
【相关文本】：
{text}
请根据相关文本回答问题，答案越简单越好，不需要任何多余解释，只要对应问题的最简单答案。如果问题的答案无法从文本中推导，请回答“ ”。你的回答应简洁明了。
"""

# 加载 SQuAD2.0 数据集（此处假定为 JSON 文件）
with open("train-v2.0.json", "r", encoding="utf-8") as file:
    squad_data = json.load(file)

# 只取前 500 个问题进行测试
questions_to_evaluate = []
for entry in squad_data["data"]:
    for paragraph in entry["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            questions_to_evaluate.append({
                "question": qa["question"],
                "context": context,
                "reference_answer": qa["answers"][0]["text"] if not qa["is_impossible"] else "",
                "is_impossible": qa["is_impossible"]
            })
            if len(questions_to_evaluate) >= 5:
                break
        if len(questions_to_evaluate) >= 5:
            break
    if len(questions_to_evaluate) >= 5:
        break

# 初始化 DeepSeek API 客户端
def generate_answer_with_api(question, context):
    client = OpenAI(
        api_key="sk-b272418942d44d62b1d1017239a60b45",  # 替换为你的 DeepSeek API 密钥
        base_url="https://api.deepseek.com"
    )

    # 填充 prompt
    prompt = PROMPT_TEMPLATE.format(question=question, text=context)

    # 调用 API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你需要根据文章选段的内容进行答题，答案只要包含要点即可，越简单越好"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=150,
        stream=False
    )

    # 解析 API 响应
    if response.choices:
        return response.choices[0].message.content.strip()
    else:
        print(f"API 请求失败，状态码：{response.status_code}")
        return ""

# 计算熵（用于衡量不确定性）
def calculate_entropy(probabilities):
    """计算概率分布的熵"""
    return -np.sum(probabilities * np.log(probabilities + 1e-10))  # 避免 log(0) 的情况

# 计算语义相似度
def compute_sts(sentence1, sentence2):
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

# 计算生成答案的不确定性（基于多样性）
def calculate_uncertainty(question, context, num_samples=3):
    """通过生成多个答案并计算多样性来衡量不确定性"""
    candidate_answers = []
    for _ in range(num_samples):
        answer = generate_answer_with_api(question, context)
        candidate_answers.append(answer)

    # 计算答案之间的平均语义相似度
    similarities = []
    for i in range(len(candidate_answers)):
        for j in range(i + 1, len(candidate_answers)):
            sim = compute_sts(candidate_answers[i], candidate_answers[j])
            similarities.append(sim)

    average_similarity = np.mean(similarities) if similarities else 0.0
    uncertainty = 1 - average_similarity  # 多样性越高，不确定性越大
    return uncertainty, candidate_answers

# 存储评估结果
results = []
total_scores = 0  # 总分数
total_uncertainty = 0  # 总不确定性

# 遍历问题并评估
for item in questions_to_evaluate:
    question = item["question"]
    context = item["context"]
    reference_answer = item["reference_answer"]
    is_impossible = item["is_impossible"]

    # 使用 API 生成答案
    generated_answer = generate_answer_with_api(question, context)

    # 计算生成答案的不确定性
    uncertainty, candidate_answers = calculate_uncertainty(question, context)

    # 计算语义相似度（用于评估答案质量）
    if generated_answer:
        if is_impossible:
            # 对于无答案问题，检查模型是否正确拒绝回答
            result = 1.0 if generated_answer == "无法根据文本回答该问题" else 0.0
        else:
            # 对于有答案问题，计算生成答案与参考答案的语义相似度
            result = compute_sts(reference_answer, generated_answer)

        # 保存评估结果
        results.append({
            "question": question,
            "context": context,
            "generated_answer": generated_answer,
            "reference_answer": reference_answer,
            "is_impossible": is_impossible,
            "score": result,
            "uncertainty": uncertainty,
            "candidate_answers": candidate_answers
        })

        # 累加分数和不确定性
        total_scores += result
        total_uncertainty += uncertainty

# 计算平均分和平均不确定性
average_score = total_scores / len(questions_to_evaluate)
average_uncertainty = total_uncertainty / len(questions_to_evaluate)

# 打印结果
print(f"平均语义相似度评分: {average_score:.5f}")
print(f"平均不确定性: {average_uncertainty:.5f}")
for res in results[:5]:  # 打印前 5 个结果示例
    print(f"问题: {res['question']}")
    print(f"上下文: {res['context'][:200]}...")  # 截取上下文前 200 字符
    print(f"生成答案: {res['generated_answer']}")
    print(f"参考答案: {res['reference_answer']}")
    print(f"得分: {res['score']:.2f}")
    print(f"不确定性: {res['uncertainty']:.2f}")
    print(f"候选答案: {res['candidate_answers']}")
    print("=" * 50)