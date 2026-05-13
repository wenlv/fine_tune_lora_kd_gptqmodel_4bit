from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import os
import gc

load_dotenv()

# 清空显存
torch.cuda.empty_cache()
gc.collect()

TRAIN_MODEL_NAME = os.getenv("MODEL_NAME")
TRAIN_TEACHER_MODEL = os.getenv("TEACHER_MODEL_NAME")


TEACHER_MODEL_PATH = f"../models/{TRAIN_TEACHER_MODEL}"
STUDENT_MODEL_PATH = f"../models/{TRAIN_MODEL_NAME}"
KD_OUTPUT_PATH = f"../output/{TRAIN_MODEL_NAME}_kd"

device = "cuda"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    STUDENT_MODEL_PATH,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# 教师模型 冻结不训练
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_PATH,
    dtype=torch.bfloat16,
    # device_map="auto",
    device_map="cpu",  #
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# 学生模型
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# ==========================================================================
# 强制老师和学生使用相同的词表维度（解决形状不匹配）
# ==========================================================================
common_vocab_size = min(teacher_model.config.vocab_size, student_model.config.vocab_size)
teacher_model.resize_token_embeddings(common_vocab_size)
student_model.resize_token_embeddings(common_vocab_size)

# 标准蒸馏通用语料（可后续替换自己数据集）
train_texts = [
    "大模型知识蒸馏让小模型学习大模型概率分布，显著提升语义理解与生成能力。",
    "量化感知训练QAT结合知识蒸馏，可以极大降低GPTQ 4bit量化的精度损失。",
    "KD + LoRA-QAT + GPTQ4bit 是当前工业界大模型轻量化部署标准全链路。",
    "知识蒸馏通过KL散度对齐教师与学生输出分布，在保留能力同时压缩模型体积。"
]
train_dataset = [tokenizer(t, truncation=True, max_length=512) for t in train_texts]
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 自定义KD蒸馏训练器
class KDTrainer(Trainer):
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

# 最优训练参数
training_args = TrainingArguments(
    output_dir=KD_OUTPUT_PATH,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = KDTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

print("🚀 开始最优参数知识蒸馏 KD ...")
trainer.train()
trainer.save_model(KD_OUTPUT_PATH)
tokenizer.save_pretrained(KD_OUTPUT_PATH)

print(f"✅ 知识蒸馏完成：{KD_OUTPUT_PATH}")
# 清空显存
torch.cuda.empty_cache()
gc.collect()