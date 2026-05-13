from dotenv import load_dotenv
import torch
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# 清空显存
torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = os.getenv('MODEL_NAME')
TRAIN_MODEL_NAME=f"../models/{MODEL_NAME}"
FAKE_QUANT_OUTPUT = f"../output/{TRAIN_MODEL_NAME}_fake_quant"

# 分词器
tokenizer = AutoTokenizer.from_pretrained(
    TRAIN_MODEL_NAME,
    trust_remote_code=True
)

print("加载蒸馏后模型进行Fake伪量化预热...")

model = AutoModelForCausalLM.from_pretrained(
    TRAIN_MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

model.eval()

# 标准校准前向，不炸内存、不用废弃API
with torch.no_grad():
    text = "大模型QAT量化感知训练 GPTQ 4bit 伪量化校准"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    _ = model(**inputs)

print("保存伪量化校准模型...")
model.save_pretrained(FAKE_QUANT_OUTPUT, safe_serialization=True)
tokenizer.save_pretrained(FAKE_QUANT_OUTPUT)

print("✅ Fake伪量化QAT预热完成")
gc.collect()
torch.cuda.empty_cache()