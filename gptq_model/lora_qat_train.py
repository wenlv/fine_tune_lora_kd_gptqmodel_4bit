from dotenv import load_dotenv
import torch
import os
import gc
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

load_dotenv()

# 清空显存
torch.cuda.empty_cache()
gc.collect()

TRAIN_MODEL_NAME = os.getenv('MODEL_NAME')
KD_MODEL = f"../output/{TRAIN_MODEL_NAME}_kd"
KD_LORA_OUTPUT = f"../output/{TRAIN_MODEL_NAME}_kd_lora"

# 分词器
tokenizer = AutoTokenizer.from_pretrained(KD_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 工业最优LoRA配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("加载伪量化模型开始LoRA-QAT训练...")

model = AutoModelForCausalLM.from_pretrained(
    KD_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_texts = [
    "量化感知训练让模型提前适配4bit量化误差，大幅降低GPTQ精度损失。",
    "LoRA-QAT在不改动主干权重前提下，微调适配量化分布，是轻量化最优方案。"
]
train_dataset = [tokenizer(t, truncation=True, max_length=512) for t in train_texts]
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=KD_LORA_OUTPUT,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

print("🚀 开始LoRA训练...")
trainer.train()
trainer.save_model(KD_LORA_OUTPUT)
print("✅ LoRA训练完成")
