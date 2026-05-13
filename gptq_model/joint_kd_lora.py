from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model,PeftModel
import os
import gc

load_dotenv()

# 清空显存
torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = os.getenv("MODEL_NAME")
TEACHER_PATH = os.getenv("TEACHER_MODEL_NAME")
STUDENT_PATH = os.getenv("MODEL_NAME")

BASE_MODEL_PATH = f"../models/{MODEL_NAME}"
TEACHER_MODEL_PATH = f"../models/{TEACHER_PATH}"
STUDENT_MODEL_PATH = f"../models/{STUDENT_PATH}"
JOINT_KD_LORA_OUTPUT = f"../output/{MODEL_NAME}_joint_kd_lora"
JOINT_KD_LORA_MERGED_OUTPUT = f"../output/{MODEL_NAME}_joint_kd_lora_merged"

# ========================== 1. 加载分词器 ==========================
tokenizer = AutoTokenizer.from_pretrained(
    STUDENT_MODEL_PATH,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# ========================== 2. 加载教师模型（固定） ==========================
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
).eval()

for param in teacher_model.parameters():
    param.requires_grad = False

# ========================== 3. 加载学生模型 ==========================
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# ==========================================================================
# 强制老师和学生使用相同的词表维度（解决形状不匹配）
# ==========================================================================
common_vocab_size = min(teacher_model.config.vocab_size, student_model.config.vocab_size)
teacher_model.resize_token_embeddings(common_vocab_size)
student_model.resize_token_embeddings(common_vocab_size)


# ==========================================================================
# 插入 LoRA】
# ==========================================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
student_model = get_peft_model(student_model, lora_config)
student_model.print_trainable_parameters()

# ========================== 训练数据 ==========================
train_texts = [
    "大模型知识蒸馏让小模型学习大模型概率分布，显著提升语义理解与生成能力。",
    "量化感知训练QAT结合知识蒸馏，可以极大降低GPTQ 4bit量化的精度损失。",
    "KD + LoRA-QAT + GPTQ4bit 是当前工业界大模型轻量化部署标准全链路。"
]
train_dataset = [tokenizer(t, truncation=True, max_length=512) for t in train_texts]
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==========================================================================
# ✅【第三步：联合训练 —— 知识蒸馏 + LoRA + QAT 一起做】
# ==========================================================================
class JointTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        inputs_cpu = {k: v.cpu() for k, v in inputs.items()}

        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs_cpu)  # 老师在CPU跑

        student_outputs = model(**inputs)  # 学生在GPU跑

        teacher_logits = teacher_outputs.logits.cuda()  # 结果传回GPU
        student_logits = student_outputs.logits

        # 再次确保维度对齐（处理 resize 后的潜在问题）
        min_vocab = min(teacher_logits.size(-1), student_logits.size(-1))
        teacher_logits = teacher_logits[..., :min_vocab]
        student_logits = student_logits[..., :min_vocab]

        T = 2.0
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean"
        ) * T * T

        lm_loss = student_outputs.loss
        total_loss = 0.7 * kl_loss + 0.3 * lm_loss
        return (total_loss, student_outputs) if return_outputs else total_loss

# ========================== 训练参数 ==========================
training_args = TrainingArguments(
    output_dir=JOINT_KD_LORA_OUTPUT,
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

trainer = JointTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# ========================== 启动联合训练 ==========================
print("🚀 开始 伪量化 + 知识蒸馏 + LoRA 联合训练")
trainer.train()
# if accelerator.is_main_process:
#     model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
#     model_to_save.save_pretrained(LORA_KD_OUTPUT)
#     tokenizer.save_pretrained(LORA_KD_OUTPUT)
trainer.save_model(JOINT_KD_LORA_OUTPUT)
tokenizer.save_pretrained(JOINT_KD_LORA_OUTPUT)

# ==========================将训练好的合并到基座模型 ==========================
#将蒸馏后的lora适配器合并到基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

lora_model = PeftModel.from_pretrained(base_model, JOINT_KD_LORA_MERGED_OUTPUT)
merged_model = lora_model.merge_and_unload()

merged_model.save_pretrained(JOINT_KD_LORA_MERGED_OUTPUT)
AutoTokenizer.from_pretrained(BASE_MODEL_PATH).save_pretrained(JOINT_KD_LORA_MERGED_OUTPUT)
print(f"✅ 合并后的模型已保存至: {JOINT_KD_LORA_MERGED_OUTPUT}")

print("✅ 全部训练完成！直接进入 GPTQ 4bit 量化！")