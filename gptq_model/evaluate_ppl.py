from dotenv import load_dotenv
import torch
import gc
import math
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel

load_dotenv()

# 清空显存
torch.cuda.empty_cache()
gc.collect()

TRAIN_MODEL_NAME = os.getenv("MODEL_NAME")

# 你要评估哪个模型，就填哪个路径
MODEL_PATH = f"../output/{TRAIN_MODEL_NAME}_fake_qat_kd_lora_joint_gptq_4bit"
# MODEL_PATH = f"../output/{TRAIN_MODEL_NAME}_kd_fake_quant_lora"
# MODEL_PATH = f"../output/{TRAIN_MODEL_NAME}_kd_fake_quant_lora_gptq_4bit"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 自动判断是否是 GPTQ 模型
try:
    model = GPTQModel.from_quantized(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True
    )
    print("✅ 加载 GPTQ 4bit 模型")
except:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ 加载 FP16/BF16 模型")

model.eval()

# 测试文本（评估PPL用）
test_text = """大模型量化感知训练（QAT）可以有效降低低比特量化带来的精度损失，
在保持模型性能的同时大幅减少显存占用，提高推理速度。
结合 LoRA 微调与 GPTQ 量化，已成为当前大模型落地部署的主流方案。"""

inputs = tokenizer(
    test_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to("cuda")

# 计算 PPL
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    ppl = math.exp(loss)

print(f"\n📊 模型路径: {MODEL_PATH}")
print(f"📉 Loss: {loss:.4f}")
print(f"📈 PPL (困惑度): {ppl:.4f}")
print("----------------------------------")
print("✅ PPL 越小 = 模型精度越高")