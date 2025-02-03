import json
import re
from typing import Union
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# --------------------- 答案规范化模块 ---------------------
def normalize_answer(text: Union[str, list]) -> str:
    """
    答案规范化处理：提取关键实体、数字，统一小写格式
    """
    if isinstance(text, list):
        text = " ".join(text)

    # 提取所有数字和字母组成的实体（过滤标点）
    numbers = re.findall(r"\d+", text)
    entities = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    # 合并并去重
    normalized = list(set(entities + numbers))
    return " ".join(sorted(normalized))  # 排序确保顺序一致


# --------------------- 动态评分模块 ---------------------
def dynamic_comprehensive_evaluation(generated: str, reference: str) -> dict:
    """
    综合评估答案质量（精确匹配 + 模糊匹配 + 语义相似度）
    动态调整评分权重
    """
    # 规范化处理
    norm_gen = normalize_answer(generated)
    norm_ref = normalize_answer(reference)

    # 1. 精确匹配
    exact_match = 1.0 if norm_ref == norm_gen else 0.0

    # 2. 模糊匹配（F1 Score）
    gen_tokens = set(norm_gen.split())
    ref_tokens = set(norm_ref.split())

    tp = len(gen_tokens & ref_tokens)
    precision = tp / len(gen_tokens) if gen_tokens else 0
    recall = tp / len(ref_tokens) if ref_tokens else 0
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)  # 防止除零

    # 3. 语义熵计算
    semantic_score = compute_semantic_entropy(generated, reference)

    # 动态调整权重
    if exact_match == 1.0:
        weight_exact = 1.0
        weight_f1 = 0
        weight_semantic = 0
    elif len(norm_ref.split()) <= 3:  # 短答案偏重精确匹配
        weight_exact = 0
        weight_f1 = 0.6
        weight_semantic = 0.4
    else:  # 长答案偏重语义匹配
        weight_exact = 0
        weight_f1 = 0.4
        weight_semantic = 0.6

    # 综合评分
    final_score = weight_exact * exact_match + weight_f1 * f1_score + weight_semantic * semantic_score

    return {
        "exact_match": exact_match,
        "f1_score": f1_score,
        "semantic_score": semantic_score,
        "final_score": final_score,
        "is_correct": final_score >= 0.5  # 阈值可调整
    }


# --------------------- 语义熵计算 ---------------------
def compute_semantic_entropy(s1, s2):
    """
    计算两个文本的语义熵差异，基于语言模型的输出。
    使用GPT类模型来计算文本的生成概率，从而推算出语义熵。
    """
    model_name = "gpt2"  # 可以选择不同的预训练语言模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 编码文本
    inputs_s1 = tokenizer(s1, return_tensors="pt")
    inputs_s2 = tokenizer(s2, return_tensors="pt")

    # 计算文本的log概率（熵）
    with torch.no_grad():
        logits_s1 = model(**inputs_s1).logits
        logits_s2 = model(**inputs_s2).logits

    # 计算每个token的log概率
    probs_s1 = torch.nn.functional.softmax(logits_s1, dim=-1)
    probs_s2 = torch.nn.functional.softmax(logits_s2, dim=-1)

    # 计算熵：H(x) = - Σ p(x) * log(p(x))
    entropy_s1 = -torch.sum(probs_s1 * torch.log(probs_s1 + 1e-9)).item()  # 加小常数避免log(0)
    entropy_s2 = -torch.sum(probs_s2 * torch.log(probs_s2 + 1e-9)).item()

    # 返回熵的差异作为相似度评分，差异小则相似度高
    return 1 / (1 + np.abs(entropy_s1 - entropy_s2))  # 返回越接近1表示语义越相似


# --------------------- 原有逻辑增强 ---------------------
# 初始化语义相似度模型
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")


# Prompt模板
PROMPT_TEMPLATE = """你是一个知识渊博且严谨的助手。以下是一个问题：
【问题】：{question}
请回答这个问题。答案越简单越好，不需要任何多余解释。用英语回答，答案越简洁越好，不需要重复问句中的信息。"""

# 加载数据集
triviaqa_path = "triviaqa.jsonl"  # 数据集路径
questions_to_evaluate = []
with open(triviaqa_path, "r", encoding="utf-8") as file:
    for idx, line in enumerate(file):
        if idx >= 100:  # 只加载前100条
            break
        entry = json.loads(line)
        questions_to_evaluate.append({
            "question": entry["question"],
            "reference_answer": entry["answer"],
            "is_impossible": entry.get("answer", "") == ""
        })


# OpenAI生成函数
def generate_answer_with_api(question):
    client = OpenAI(api_key="sk-b272418942d44d62b1d1017239a60b45", base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system", "content": "直接回答问题，答案只需包含核心信息。"
            }, {
                "role": "user", "content": PROMPT_TEMPLATE.format(question=question)
            }],
            temperature=1.0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {str(e)}")
        return ""


# --------------------- 评估流程增强 ---------------------
results = []
for item in questions_to_evaluate:
    gen_answer = generate_answer_with_api(item["question"])
    if not gen_answer:
        continue

    # 执行综合评估
    eval_result = dynamic_comprehensive_evaluation(
        generated=gen_answer,
        reference=item["reference_answer"]
    )

    # 记录结果
    results.append({
        "question": item["question"],
        "generated": gen_answer,
        "reference": item["reference_answer"],
        **eval_result,
        "is_impossible": item["is_impossible"]
    })


# --------------------- 结果分析增强 ---------------------
def print_colored(text, color_code):
    """终端彩色输出"""
    print(f"\033[{color_code}m{text}\033[0m")


# 统计综合评分的平均分
average_final_score = sum(res["final_score"] for res in results) / len(results) if results else 0

print(f"\n综合评分平均分: {average_final_score:.4f}")

print("=" * 60)

# 详细结果展示
for res in results[:100]:  # 打印前10个示例
    color_code = "32" if res["is_correct"] else "31"
    print_colored(f"问题：{res['question']}", color_code)
    print(f"生成答案：{res['generated']}")
    print(f"参考答案：{res['reference']}")
    print(f"精确匹配：{res['exact_match']:.4f} | F1分数：{res['f1_score']:.4f} | 语义分：{res['semantic_score']:.4f}")
    print(f"综合评分：{res['final_score']:.4f} | 判定结果：{'正确' if res['is_correct'] else '错误'}")
    print("-" * 60)
