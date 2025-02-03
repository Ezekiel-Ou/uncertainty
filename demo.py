import json
import re
from typing import Union
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from zhipuai import ZhipuAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
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
    综合评估答案质量（精确匹配 + 模糊匹配 + 语义密度）
    动态调整评分权重
    """
    # 规范化处理
    norm_gen = normalize_answer(generated)
    norm_ref = normalize_answer(reference)

    # 1. 精确匹配
    exact_match = 1.0 if norm_ref == norm_gen else 0.0

     #2. 模糊匹配（F1 Score）
    gen_tokens = set(norm_gen.split())
    ref_tokens = set(norm_ref.split())

    tp = len(gen_tokens & ref_tokens)
    precision = tp / len(gen_tokens) if gen_tokens else 0
    recall = tp / len(ref_tokens) if ref_tokens else 0
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)  # 防止除零

    # 3. 语义密度
    semantic_density = compute_semantic_density(norm_gen, norm_ref)

    #动态调整权重
    if exact_match == 1.0:
        weight_exact = 1.0
        weight_f1 = 0
        weight_density = 0
    elif len(norm_ref.split()) <= 3:  # 短答案偏重精确匹配
        weight_exact = 0
        weight_f1 = 0.2
        weight_density = 0.8
    else:  # 长答案偏重语义匹配
        weight_exact = 0
        weight_f1 = 0.05
        weight_density = 0.95

    '''if exact_match == 1.0:
        weight_exact = 1.0
        weight_density = 0
    else:
        weight_exact = 0
        weight_density=1.0'''

    # 综合评分
    final_score = (weight_exact * exact_match +
                   weight_f1 * f1_score +
                   weight_density * semantic_density)

    return {
        "exact_match": exact_match,
        "f1_score": f1_score,
        "semantic_density": semantic_density,
        "final_score": final_score,
        "is_correct": final_score >= 0.6  # 阈值可调整
    }

# --------------------- 原有逻辑增强 ---------------------
# 初始化NLI模型
nli_model_name = "microsoft/deberta-large-mnli"
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)

# Prompt模板
PROMPT_TEMPLATE = """你是一个知识渊博且严谨的助手。以下是一个问题：
【问题】：{question}
请回答这个问题。答案越简单越好，不需要任何多余解释。用英语回答，答案越简洁越好，不需要重复问句中的信息."""

# 加载数据集
triviaqa_path = "triviaqa.jsonl"  # 替换为你的数据路径
questions_to_evaluate = []
with open(triviaqa_path, "r", encoding="utf-8") as file:
    for idx, line in enumerate(file):
        if idx >= 20:  # 只加载前20条
            break
        entry = json.loads(line)
        questions_to_evaluate.append({
            "question": entry["question"],
            "reference_answer": entry["answer"],
            "is_impossible": entry.get("answer", "") == ""
        })

# OpenAI生成函数
def generate_answer_with_api(question):
    client = ZhipuAI(api_key="4b58ba8916374a9ba1bf693f8224bf15.KhOOWDSO8ja8HK98")
    try:
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "system", "content": "直接回答问题，答案只需包含核心信息。"},
                {"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}
            ],
            temperature=1.0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {str(e)}")
        return ""

# 语义密度计算
def compute_semantic_density(generated, reference):
    inputs = nli_tokenizer(generated, reference, return_tensors="pt", truncation=True)
    outputs = nli_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    p_entailment = probs[:, 2].item()
    p_neutral = probs[:, 1].item()
    p_contradiction = probs[:, 0].item()
    semantic_density = p_entailment + 0.5 * p_neutral
    return semantic_density

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
for res in results[:200]:  # 打印前20个示例
    color_code = "32" if res["is_correct"] else "31"
    print_colored(f"问题：{res['question']}", color_code)
    print(f"生成答案：{res['generated']}")
    print(f"参考答案：{res['reference']}")
    print(f"精确匹配：{res['exact_match']:.4f} | F1分数：{res['f1_score']:.4f} | 语义密度：{res['semantic_density']:.4f}")
    #print(f"精确匹配：{res['exact_match']:.4f} | 语义密度：{res['semantic_density']:.4f}")
    print(f"综合评分：{res['final_score']:.4f} | 判定结果：{'正确' if res['is_correct'] else '错误'}")
    print("-" * 60)


def generate_separate_charts(results, save_dir="visualization_results"):
    """
    生成并保存四张独立图表到指定目录
    返回保存成功的文件路径列表
    """
    # 创建存储目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 准备数据
    final_scores = [r["final_score"] for r in results]
    exact_matches = [r["exact_match"] for r in results]
    f1_scores = [r["f1_score"] for r in results]
    semantic_densities = [r["semantic_density"] for r in results]
    correctness = [r["is_correct"] for r in results]

    saved_files = []

    try:
        # 1. 综合评分分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(final_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.8)
        plt.title('Final Score Distribution', fontsize=14)
        plt.xlabel('Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        hist_path = os.path.join(save_dir, "score_distribution.png")
        plt.savefig(hist_path, dpi=120, bbox_inches='tight')
        plt.close()
        saved_files.append(hist_path)

        # 2. 正确率饼图
        plt.figure(figsize=(8, 8))
        correct_count = sum(correctness)
        sizes = [correct_count, len(results) - correct_count]
        explode = (0.05, 0)  # 突出显示正确部分
        plt.pie(sizes, labels=['Correct', 'Incorrect'], autopct='%1.1f%%',
                colors=['#66b3ff', '#ff9999'], startangle=90, explode=explode,
                textprops={'fontsize': 12}, shadow=True)
        plt.title('Accuracy Ratio', fontsize=14)
        pie_path = os.path.join(save_dir, "accuracy_pie.png")
        plt.savefig(pie_path, dpi=120, bbox_inches='tight')
        plt.close()
        saved_files.append(pie_path)

        # 3. 指标雷达图
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        categories = ['Exact Match', 'F1 Score', 'Semantic\nDensity']
        avg_values = [
            np.mean(exact_matches),
            np.mean(f1_scores),
            np.mean(semantic_densities)
        ]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        avg_values += avg_values[:1]
        angles += angles[:1]

        ax.plot(angles, avg_values, color='#4CAF50', linewidth=3, linestyle='solid')
        ax.fill(angles, avg_values, color='#4CAF50', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_rlabel_position(30)
        plt.title('Metrics Radar Chart', fontsize=14, y=1.08)
        radar_path = os.path.join(save_dir, "metrics_radar.png")
        plt.savefig(radar_path, dpi=120, bbox_inches='tight')
        plt.close()
        saved_files.append(radar_path)

        # 4. 语义密度散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(semantic_densities, final_scores,
                              c=final_scores, cmap='viridis',
                              alpha=0.7, edgecolors='w', linewidth=0.5)
        plt.colorbar(scatter, label='Final Score', shrink=0.8)
        plt.xlabel('Semantic Density', fontsize=12)
        plt.ylabel('Final Score', fontsize=12)
        plt.title('Semantic Density vs Final Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        scatter_path = os.path.join(save_dir, "semantic_scatter.png")
        plt.savefig(scatter_path, dpi=120, bbox_inches='tight')
        plt.close()
        saved_files.append(scatter_path)

    except Exception as e:
        print_colored(f"图表生成失败: {str(e)}", "31")
        return []

    return saved_files


# --------------------- 在结果分析后调用 ---------------------
# 替换原有的 generate_visualizations 调用
output_dir = "evaluation_charts"  # 可自定义目录名称
saved_files = generate_separate_charts(results, save_dir=output_dir)

if saved_files:
    print_colored(f"\n📊 四张图表已保存至 {output_dir} 目录：", "32")
    for path in saved_files:
        print_colored(f"✅ {os.path.basename(path)}", "36")
else:
    print_colored("\n⚠️ 图表保存失败，请检查错误信息", "31")