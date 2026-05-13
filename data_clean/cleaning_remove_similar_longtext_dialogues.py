# -*- coding: utf-8 -*-
"""
功能说明：
-----------------------------------------------
该脚本对 ChatML 格式语料执行两步清洗：
1️⃣ 哈希去重：用 MD5 检测完全重复样本；
2️⃣ 语义相似度过滤：用 Sentence-BERT + FAISS 去除语义高度相似样本；
-----------------------------------------------
输出结果可直接用于模型训练（如 SFT 或 LoRA）。
"""
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import hashlib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ======================================================
# 1️⃣ 哈希去重
# ======================================================
def hash_dedup(samples):
    """
    使用 MD5 对语料去重
    参数：
        samples: List[Dict]，每个样本为 {"id":..., "messages":[...]}
    返回：
        unique_samples: 去重后的样本列表
    """
    seen = set()                    # 存放已出现过的哈希值
    unique_samples = []             # 存放唯一样本

    for s in samples:
        # 将多轮对话拼成一个长字符串（用空格连接所有 message 内容）
        text = " ".join([m["content"] for m in s["messages"]])
        # 计算 MD5 哈希
        h = hashlib.md5(text.strip().encode("utf-8")).hexdigest()

        # 若未出现过该哈希，则保留
        if h not in seen:
            seen.add(h)
            unique_samples.append(s)
    return unique_samples


# ======================================================
# 2️⃣ 相似度过滤（语义重复检测）
# ======================================================
def similarity_filter(samples,threshold=0.9,
                      model_name="../models/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    使用 Sentence-BERT + FAISS 去除语义重复样本。
    参数：
        samples: 样本列表
        threshold: 相似度阈值 (0~1)，高于此值即认为内容重复
        model_name: 句向量模型（支持多语言）
    返回：
        unique_samples: 过滤后的样本列表
    """
    print("🔹 正在加载 Sentence-BERT 模型（用于相似度计算）...")
    model = SentenceTransformer(model_name)   # 加载语义编码模型

    # 提取每条样本的文本（此处取所有 messages 内容拼接）
    texts = [" ".join([m["content"] for m in s["messages"]]) for s in samples]

    print(f"🔹 正在编码 {len(texts)} 条样本...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True   # ✅ 向量归一化，使内积等价于余弦相似度
    )

    # 创建 FAISS 向量索引
    dim = embeddings.shape[1]       # 向量维度
    index = faiss.IndexFlatIP(dim)  # 内积相似度索引
    index.add(embeddings)           # 加入所有样本向量

    unique_samples = []             # 存放唯一样本
    seen = set()                    # 已被判定为重复的样本索引

    print("🔹 正在进行相似度过滤...")
    for i, emb in enumerate(tqdm(embeddings, desc="相似度检测")):
        if i in seen:
            continue  # 若已标记为重复则跳过
        unique_samples.append(samples[i])

        # 在索引中搜索相似样本
        sim, idx = index.search(np.array([emb]), len(samples))
        for j, score in zip(idx[0], sim[0]):
            if score >= threshold:
                seen.add(j)  # 标记为重复

    return unique_samples


# ======================================================
# 3️⃣ 主程序：整合流程
# ======================================================
def process_jsonl(input_file, output_file, threshold=0.9):
    """
    主函数：执行哈希去重 + 相似度过滤
    """
    print("=========================1读取输入文件===================")
    # ---------- 读取输入文件 ----------
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"📂 原始样本数量: {len(samples)}")
    print("=========================2相似度过滤===================")
    # ---------- 相似度过滤 ----------
    samples = hash_dedup(samples)
    print(f"✅ 哈希去重后: {len(samples)}")
    print("=========================3相似度过滤===================")
    # ---------- 相似度过滤 ----------
    samples = similarity_filter(samples, threshold=threshold)
    print(f"✅ 相似度过滤后: {len(samples)}")

    print("=========================4 重新编号===================")
    # ---------- 重新编号 ----------
    for i, s in enumerate(samples):
        s["id"] = str(i)

    print("=========================5 写入输出文件===================")
    # ---------- 写入输出文件 ----------
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"🎯 清洗完成，已保存到：{output_file}")


# ======================================================
# 4️⃣ 命令行入口
# ======================================================
if __name__ == "__main__":
    process_jsonl(
        input_file="../data/source_data/similar_longtext_dialogues_chatml.jsonl",     # 输入：ChatML 语料
        output_file="../data/data_cleaning/cleaning_remove_similar_longtext_dialogues.jsonl",            # 输出：清洗后语料
        threshold=0.85                                    # 相似度阈值（越高越严格）
    )
