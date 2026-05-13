from dotenv import load_dotenv
import os
import torch
import gc
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel

load_dotenv()

# 清空显存
torch.cuda.empty_cache()
gc.collect()

TRAIN_MODEL_NAME = os.getenv("MODEL_NAME")
TEACHER_MODEL_NAME = os.getenv("TEACHER_MODEL_NAME")

paths = [
    # ("① 蒸馏后原始模型", f"../output/{TRAIN_MODEL_NAME}_kd"),
    #     # ("② 伪量化预热模型", f"../output/{TRAIN_MODEL_NAME}_kd_fake_quant"),
    #     # ("③ LoRA-QAT训练后模型", f"../output/{TRAIN_MODEL_NAME}_kd_fake_quant_lora"),
    #     # ("④ 最终GPTQ 4bit模型", f"../output/{TRAIN_MODEL_NAME}_fake_qat_kd_lora_joint_gptq_4bit"),
    ("① 学生原始模型", f"../models/{TRAIN_MODEL_NAME}"),
    ("② 教师原始模型", f"../models/{TEACHER_MODEL_NAME}"),
    ("③ 蒸馏+LORA后原始模型", f"../output/{TRAIN_MODEL_NAME}_joint_kd_lora_merged"),
    ("④ 最终GPTQ 4bit模型", f"../output/{TRAIN_MODEL_NAME}_joint_kd_lora_merged_gptq_4bit"),
]

test_text = """大模型量化感知训练（QAT）可以有效降低低比特量化带来的精度损失，
在保持模型性能的同时大幅减少显存占用，提高推理速度。
结合知识蒸馏与LoRA微调、GPTQ量化，是工业级部署标准方案。"""

print("===== 全链路 PPL 困惑度对比（越小越好）=====\n")

for name, path in paths:
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = GPTQModel.from_quantized(path, device_map="auto", trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

    model.eval()
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        ppl = math.exp(loss)

    print(f"{name}")
    print(f"   Loss: {loss:.4f}")
    print(f"   PPL : {ppl:.4f}\n")