from dotenv import load_dotenv
import os
import torch
import gc
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

load_dotenv()

# 清空显存
torch.cuda.empty_cache()
gc.collect()

TRAIN_MODEL_NAME = os.getenv('MODEL_NAME')
KD_FAKE_QUANT_LORA_MODEL = f"../output/{TRAIN_MODEL_NAME}_joint_kd_lora_merged"
KD_FAKE_QUANT_LORA_GPTQ_4BIT_OUTPUT = f"../output/{TRAIN_MODEL_NAME}_joint_kd_lora_merged_gptq_4bit"

# 分词器
tokenizer = AutoTokenizer.from_pretrained(KD_FAKE_QUANT_LORA_MODEL, trust_remote_code=True)

# GPTQ工业最优配置
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    model_seqlen=2048
)

print("加载QAT模型进入GPTQ 4bit量化...")

# calib_texts = [
#     "QAT量化感知训练后进行GPTQ 4bit量化，精度损失最小。",
#     "知识蒸馏+QAT适配后的模型，量化后推理效果接近原模型。"
# ]
# calib_dataset = [tokenizer(x) for x in calib_texts]
calib_texts = [
    "QAT量化感知训练后进行GPTQ 4bit量化，精度损失最小。",
    "知识蒸馏+QAT适配后的模型，量化后推理效果接近原模型。",
    #扩充校准数据避免警告
    "大模型轻量化部署是工业界的重要方向。",
    "GPTQ 4bit量化可以大幅降低显存占用，加速推理。",
] * 16  #重复16次，得到 32×16 = 512 条数据

# 直接传字符串，不要手动 tokenize
calib_dataset = calib_texts  # ✅ 正确：直接传字符串列表

model = GPTQModel.from_pretrained(
    KD_FAKE_QUANT_LORA_MODEL,
    quantize_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

print("执行GPTQ 4bit量化...")
model.quantize(calib_dataset)

# model.save_quantized(KD_FAKE_QUANT_LORA_GPTQ_4BIT_OUTPUT, use_safetensors=True)
model.save_quantized(KD_FAKE_QUANT_LORA_GPTQ_4BIT_OUTPUT)
tokenizer.save_pretrained(KD_FAKE_QUANT_LORA_GPTQ_4BIT_OUTPUT)

print("✅ 全链路完成：KD→Fake→QAT→GPTQ4bit")