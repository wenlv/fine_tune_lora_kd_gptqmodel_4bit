"""
filter_chinese_dataset_offline.py
---------------------------------
💡 本程序在完全离线环境下运行，无需联网。
用于中文语料的质量检测与过滤，支持：
  1. 正则规则过滤（去除无效文本）
  2. 有害内容检测（TextDetox + Alibaba 模型）
  3. 流畅度检测（使用 Qwen1.5-1.8B 计算困惑度 PPL）
"""

import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


# ======================================================
# 🧩 1️⃣ 正则规则过滤函数
# ======================================================
def rule_filter(text, min_len=5, max_len=500):
    """用于检测文本是否合格（长度、隐私、URL 等）"""
    if not text or not text.strip():  # 空行或空白
        return False
    if len(text) < min_len or len(text) > max_len:  # 长度太短或太长
        return False
    if re.search(r"(http[s]?://|www\.)", text):  # 包含网址
        return False
    if re.search(r"\S+@\S+", text):  # 包含邮箱
        return False
    if re.search(r"\d{7,}", text):  # 出现长数字串（可能是身份证/电话）
        return False
    return True  # 符合要求则保留


# ======================================================
# 🧩 2️⃣ 有害内容检测类（完全离线）
# ======================================================
class ToxicityDetector:
    """离线模式的有害内容检测器，加载本地模型进行文本安全性判断"""

    def __init__(self, local_path, label_for_bad=1):
        # local_path: 模型文件夹路径，例如 "../models/pai-bert-base-zh-llm-risk-detection"
        # label_for_bad: 模型输出中代表“有害”的类别编号（通常是1）
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"❌ 模型目录不存在: {local_path}")

        print(f"📦 从本地加载有害检测模型: {local_path}")
        # 加载分词器和模型（完全离线）
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(local_path, local_files_only=True)
        self.model.eval()  # 设置为推理模式
        self.label_for_bad = label_for_bad

    def is_safe(self, text, threshold=0.5):
        """判断文本是否安全：返回 True 表示安全"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():  # 关闭梯度计算，提高推理速度
            logits = self.model(**inputs).logits  # 得到模型输出分数
            probs = torch.softmax(logits, dim=1)[0]  # 计算概率
        bad_prob = float(probs[self.label_for_bad])  # 获取“有害”类别的概率
        return bad_prob < threshold  # 若低于阈值则认为安全


# ======================================================
# 🧩 3️⃣ 困惑度检测类 (Perplexity)
# ======================================================
class PerplexityScorer:
    """使用本地 Qwen 模型计算文本困惑度 (PPL)"""

    def __init__(self, model_path="../models/Qwen1.5-1.8B-Chat"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Qwen 模型目录不存在: {model_path}")

        print(f"🔹 从本地加载 Qwen1.5-1.8B 模型: {model_path}")
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        # 加载模型（自动检测 GPU / CPU）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
            torch_dtype="auto"
        )
        self.model.eval()  # 关闭训练模式

    def score(self, text):
        """计算文本的困惑度 (PPL)，数值越低代表越自然"""
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc.input_ids.to(self.model.device)
        target_ids = input_ids.clone()
        with torch.no_grad():  # 不计算梯度
            outputs = self.model(input_ids, labels=target_ids)
            loss = outputs.loss
        ppl = torch.exp(loss)  # PPL = e^(loss)
        return ppl.item()  # 返回 float 值


# ======================================================
# 🧩 4️⃣ 主清洗流程函数
# ======================================================
def clean_chinese_dataset_offline(
    input_file,                 # 输入文件（JSONL）
    output_file,                # 输出文件（JSONL）
    qwen_path,
    detox_path,
    pai_path,
    ppl_threshold=80,           # 困惑度阈值，超过则判定为“不自然”
    toxic_threshold=0.5,        # 有害内容阈值
    min_len=5,                  # 最小字符数
    max_len=500                 # 最大字符数
):
    print("=========================1 执行中文数据的清洗与质量检测===================")
    """执行中文数据的清洗与质量检测"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("=========================2 读取输入数据（逐行 JSON）===================")
    # 读取输入数据（逐行 JSON）
    with open(input_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"📂 原始样本数: {len(samples)}")

    print("=========================3 初始化三个模型（全部离线加载）===================")
    # 初始化三个模型（全部离线加载）
    detector1 = ToxicityDetector(local_path=detox_path)  # 多语言毒性检测模型
    detector2 = ToxicityDetector(local_path=pai_path)    # 阿里中文风险检测模型
    ppl_scorer = PerplexityScorer(model_path=qwen_path)  # Qwen 流畅度检测模型

    cleaned = []  # 存储通过过滤的样本
    stats = {"rule": 0, "tox1": 0, "tox2": 0, "ppl": 0}  # 统计各类过滤数

    print("=========================4 核心循环：逐条检测 ===================")
    # ===== 核心循环：逐条检测 =====
    for s in tqdm(samples):
        # 支持两种输入格式：dialogue 或 text
        if "messages" in s:
            # 如果是 ChatML 对话结构
            text = " ".join([m.get("content", "") for m in s["messages"]])
        else:
            # 如果是普通单文本结构
            text = s.get("text", "")

        # ---------- 规则过滤 ----------
        if not rule_filter(text, min_len, max_len):
            stats["rule"] += 1
            continue

        # ---------- 有害检测 (TextDetox) ----------
        if not detector1.is_safe(text, toxic_threshold):
            stats["tox1"] += 1
            continue

        # ---------- 有害检测 (Alibaba PAI) ----------
        if not detector2.is_safe(text, toxic_threshold):
            stats["tox2"] += 1
            continue

        # ---------- 困惑度检测 (PPL) ----------
        ppl = ppl_scorer.score(text)
        if ppl > ppl_threshold:
            stats["ppl"] += 1
            continue

        # 如果全部通过，保留样本
        cleaned.append(s)

    print("=========================5 保存清洗后的数据===================")
    # ===== 保存清洗后的数据 =====
    with open(output_file, "w", encoding="utf-8") as f:
        for s in cleaned:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("=========================6 输出清洗报告===================")
    # ===== 输出清洗报告 =====
    print("\n📊 数据清洗报告：")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  ✅ 保留样本: {len(cleaned)} / {len(samples)}")
    print(f"  📁 输出文件: {output_file}")

    print("=========================end===================")

# ======================================================
# 🧩 5️⃣ 主程序入口
# ======================================================
if __name__ == "__main__":
    # 调用主函数
    clean_chinese_dataset_offline(
        input_file="../data/source_data/raw_chinese_data.jsonl",                 # 输入文件路径
        output_file="../data/data_cleaning/cleaning_chinese_data_offline.jsonl",    # 输出文件路径
        qwen_path="../models/Qwen1.5-1.8B-Chat",
        detox_path = "../models/bert-multilingual-toxicity-classifier",  #多语言毒性检测模型
        pai_path = "../models/pai-bert-base-zh-llm-risk-detection", #阿里中文风险检测模型
        ppl_threshold=80,                                    # 困惑度阈值
        toxic_threshold=0.5                                  # 有害内容阈值
    )
