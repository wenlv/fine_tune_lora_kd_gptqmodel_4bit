# download_filter_models.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from modelscope import snapshot_download

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import json

print("开始下载数据集清洗所需的模型...")
print("="*60)

base_model="../models"
# 创建模型目录
os.makedirs(base_model, exist_ok=True)

# 下载 Qwen1.5-1.8B-Chat 模型（如果还没有下载）
print("\n 检查 Qwen1.5-1.8B-Chat 模型...")
qwen_path = f"{base_model}/Qwen1.5-1.8B-Chat"

if not os.path.exists(qwen_path):
    model_dir = snapshot_download(
        model_id='qwen/Qwen1.5-1.8B-Chat',  # 模型的唯一标识ID，例如 'qwen/Qwen1.5-1.8B-Chat'
        local_dir=qwen_path,  # 可选，指定模型的下载根目录
        revision='master'  # 可选，指定版本，默认为 'master'
    )
    print(f"模型已下载到: {model_dir}")
else:
    print("   ✅ Qwen 模型已存在")

# 下载 Qwen1.5-1.8B-Chat 模型（如果还没有下载）
print("\n 检查 Qwen2-0.5B-Instruct 模型...")
qwen_path = f"{base_model}/Qwen2-0.5B-Instruct"
if not os.path.exists(qwen_path):

    model_dir = snapshot_download(
        model_id='Qwen/Qwen2-0.5B-Instruct',  # 模型的唯一标识ID，例如 'qwen/Qwen1.5-1.8B-Chat'
        local_dir=qwen_path,  # 可选，指定模型的下载根目录
        revision='master'  # 可选，指定版本，默认为 'master'
    )
    print(f"模型已下载到: {model_dir}")
else:
    print("   ✅ Qwen 模型已存在")

# 下载 Qwen1.5-1.8B-Chat 模型（如果还没有下载）
print("\n 检查 Qwen2.5-7B-Instruct 模型...")
qwen_path = f"{base_model}/Qwen2.5-7B-Instruct"
if not os.path.exists(qwen_path):

    model_dir = snapshot_download(
        model_id='Qwen/Qwen2.5-7B-Instruct',  # 模型的唯一标识ID，例如 'qwen/Qwen1.5-1.8B-Chat'
        local_dir=qwen_path,  # 可选，指定模型的下载根目录
        revision='master'  # 可选，指定版本，默认为 'master'
    )
    print(f"模型已下载到: {model_dir}")
else:
    print("   ✅ Qwen 模型已存在")


# 下载 Qwen1.5-1.8B-Chat-GPTQ-4bit 模型（如果还没有下载）
print("\n  检查 Qwen1.5-1.8B-Chat-GPTQ-4bit 模型...")
gptq_path = f"{base_model}/Qwen1.5-1.8B-Chat-GPTQ-4bit"
if not os.path.exists(gptq_path):

    model_dir = snapshot_download(
        model_id='qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4',  # 模型的唯一标识ID，例如 'qwen/Qwen1.5-1.8B-Chat'
        local_dir=gptq_path,  # 旧版本用 cache_dir 即可
        revision='master',  # 可选，指定版本，默认为 'master'
        ignore_file_pattern = None  # 下载所有文件
    )
    print(f"Qwen1.5-1.8B-Chat-GPTQ-4bit模型已下载到: {model_dir}")
else:
    print("   ✅ Qwen1.5-1.8B-Chat-GPTQ-4bit模型已存在")

# 下载 Qwen2.5-VL-3B-Instruct 模型（如果还没有下载）
print("\n  检查 Qwen2.5-VL-3B-Instruct 模型...")
gptq_path = f"{base_model}/Qwen2.5-VL-3B-Instruct"
if not os.path.exists(gptq_path):

    model_dir = snapshot_download(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        local_dir=gptq_path,
        ignore_patterns=["imgs/*", "*.DS_Store"]
    )
    print(f"Qwen2.5-VL-3B-Instruct模型已下载到: {model_dir}")
else:
    print("   ✅ Qwen2.5-VL-3B-Instruct模型已存在")

quantize_config_path = os.path.join(gptq_path,'quantize_config.json')
if not os.path.exists(quantize_config_path):
    print("     手动创建Qwen1.5-1.8B-Chat-GPTQ-Int4 模型的quantize_config.json文件...")

    config = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "true_sequential": True,
        "model_name_or_path": 'qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4'
    }

    config_path = os.path.join(gptq_path, 'quantize_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ 手动创建Qwen1.5-1.8B-Chat-GPTQ-Int4 模型的 {config_path}")
else:
    print("   ✅ Qwen1.5-1.8B-Chat-GPTQ-Int4 模型的quantize_config.json文件已存在")

# 下载 paraphrase-multilingual-MiniLM-L12-v2 模型（如果还没有下载）
print("\n paraphrase-multilingual-MiniLM-L12-v2 模型...")
L12_path = f"{base_model}/paraphrase-multilingual-MiniLM-L12-v2"
if not os.path.exists(L12_path):
    model_dir = snapshot_download(
        model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',   # 模型的唯一标识ID，例如 'qwen/Qwen1.5-1.8B-Chat'
        local_dir=L12_path,  # 旧版本用 cache_dir 即可
    )
    print(f"模型已下载到: {model_dir}")
else:
    print("   ✅ paraphrase-multilingual-MiniLM-L12-v2 模型已存在")

# 下载 bge-m3模型（如果还没有下载）
print("\n bge-m3 模型...")
L12_path = f"{base_model}/bge-m3"
if not os.path.exists(L12_path):
    model_dir = snapshot_download(
        repo_id="BAAI/bge-m3",  # HuggingFace 模型名称
        local_dir=L12_path,  # 本地保存路径
        ignore_patterns=["imgs/*", "*.DS_Store"]  # 忽略无用文件
    )
    print(f"bge-m3 模型已下载到: {model_dir}")
else:
    print("   ✅ bge-m3 模型已存在")

# 下载多语言毒性检测模型（BERT多语言版本）
print("\n  下载多语言毒性检测模型...")
detox_path =f"{base_model}/bert-multilingual-toxicity-classifier"
if not os.path.exists(detox_path):
    print("   下载 unitary/toxic-bert 模型...")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model.save_pretrained(detox_path)
    tokenizer.save_pretrained(detox_path)
    print("   ✅ 毒性检测模型下载完成")
else:
    print("   ✅ 毒性检测模型已存在")

# 下载阿里中文风险检测模型
print("\n 下载阿里中文风险检测模型...")
pai_path =f"{base_model}/pai-bert-base-zh-llm-risk-detection"
if not os.path.exists(pai_path):
    print("   下载阿里中文风险检测模型...")
    try:
        # 尝试从 ModelScope 下载（阿里模型推荐）

        snapshot_download(
            'damo/nlp_bert_base_zh_llm_risk_detection',
            cache_dir=pai_path
        )
        print("   ✅ 阿里风险检测模型下载完成")
    except:
        # 如果 ModelScope 失败，尝试从 HuggingFace 下载
        print("   尝试从 HuggingFace 下载...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-chinese",  # 使用基础模型作为替代
            num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        model.save_pretrained(pai_path)
        tokenizer.save_pretrained(pai_path)
        print("   ✅ 使用 BERT-base-chinese 作为替代")
else:
    print("   ✅ 阿里风险检测模型已存在")




print("\n" + "="*60)
print("所有模型下载完成！")
print("\n" + "="*60)

