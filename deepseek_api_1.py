import random
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# 初始化语义相似度模型
nli_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# 定义生成答案的函数
def generate_answer_with_api(prompt, temperature=1.0):
    """通过 API 生成答案"""
    client = OpenAI(
        api_key="sk-b272418942d44d62b1d1017239a60b45",  # 替换为你的 API 密钥
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个知识渊博的助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=150
    )
    if response.choices:
        return response.choices[0].message.content.strip()
    else:
        raise Exception(f"API 请求失败，状态码：{response.status_code}")


# 计算语义相似度
def compute_similarity(sentence1, sentence2):
    """计算两段文本的语义相似度"""
    embedding1 = nli_model.encode(sentence1, convert_to_tensor=True)
    embedding2 = nli_model.encode(sentence2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()


# 观测一致性评分
def observed_consistency(question, context, reference_answer, k=5):
    """通过多次采样计算观测一致性评分"""
    answers = []
    scores = []

    # 生成多个候选答案
    for _ in range(k):
        generated_answer = generate_answer_with_api(
            f"Question: {question}\nContext: {context}\nAnswer the question based on the context as briefly as possible.",
            temperature=random.uniform(0.7, 1.3)
        )
        answers.append(generated_answer)
        # 计算每个答案与参考答案的相似度
        score = compute_similarity(generated_answer, reference_answer)
        scores.append(score)

    # 返回每个答案和对应的观测一致性得分
    return answers, scores


# 自反思一致性评分
def self_reflection_certainty(question, generated_answer):
    """
    基于自反思的置信度评分，包含两轮判断：
    - 第一轮：对答案的初步分析。
    - 第二轮：重新确认答案是否正确。
    """
    # 第一轮判断
    prompt_1 = f"""Question: {question}, Proposed Answer: {generated_answer}.
Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template:
explanation: [insert analysis], answer: [choose one letter from among choices A through C]"""

    response_1 = generate_answer_with_api(prompt_1, temperature=0.7)
    if "A" in response_1:  # 正确
        score_1 = 1.0
    elif "B" in response_1:  # 错误
        score_1 = 0.0
    elif "C" in response_1:  # 不确定
        score_1 = 0.5
    else:
        score_1 = 0.0  # 默认值

    # 第二轮判断
    prompt_2 = f"""Question: {question}, Proposed Answer: {generated_answer}.
Are you really sure the proposed answer is correct? Choose again: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template:
explanation: [insert analysis], answer: [choose one letter from among choices A through C]"""

    response_2 = generate_answer_with_api(prompt_2, temperature=0.7)
    if "A" in response_2:  # 正确
        score_2 = 1.0
    elif "B" in response_2:  # 错误
        score_2 = 0.0
    elif "C" in response_2:  # 不确定
        score_2 = 0.5
    else:
        score_2 = 0.0  # 默认值

    # 综合两轮判断的平均分作为最终置信度
    return (score_1 + score_2) / 2


# 综合评分
def compute_confidence(question, context, reference_answer, k=5):
    """结合观测一致性和自反思计算综合置信度，并选取最佳答案"""
    # 计算观测一致性得分
    answers, observation_scores = observed_consistency(question, context, reference_answer, k)

    # 计算每个答案的自反思置信度得分
    reflection_scores = [self_reflection_certainty(question, answer) for answer in answers]

    # 计算综合评分
    beta = 0.7  # 权重
    combined_scores = [
        beta * observation_score + (1 - beta) * reflection_score
        for observation_score, reflection_score in zip(observation_scores, reflection_scores)
    ]

    # 选取得分最高的答案
    best_index = combined_scores.index(max(combined_scores))
    best_answer = answers[best_index]
    best_score = combined_scores[best_index]

    return best_answer, best_score


# 示例数据
questions_to_evaluate = [
    {
        "question": "When did Beyonce start becoming popular?",
        "context": "Beyoncé Giselle Knowles-Carter is an American singer. She started becoming popular in the late 1990s as the lead singer of the R&B girl-group Destiny's Child.",
        "reference_answer": "in the late 1990s",
        "is_impossible": False
    },
    {
        "question": "What areas did Beyonce compete in when she was growing up?",
        "context": "Beyoncé was involved in singing and dancing competitions while growing up.",
        "reference_answer": "singing and dancing",
        "is_impossible": False
    }
]

# 执行评估
results = []
for item in questions_to_evaluate:
    question = item["question"]
    context = item["context"]
    reference_answer = item["reference_answer"]

    # 综合置信度计算并选取最佳答案
    best_answer, best_score = compute_confidence(question, context, reference_answer, k=5)

    # 保存结果
    results.append({
        "question": question,
        "context": context,
        "reference_answer": reference_answer,
        "best_answer": best_answer,
        "best_score": best_score
    })

# 打印结果
for res in results:
    print(f"问题: {res['question']}")
    print(f"上下文: {res['context'][:200]}...")  # 截取上下文前 200 字符
    print(f"参考答案: {res['reference_answer']}")
    print(f"最佳生成答案: {res['best_answer']}")
    print(f"综合置信度得分: {res['best_score']:.4f}")
    print("=" * 50)
