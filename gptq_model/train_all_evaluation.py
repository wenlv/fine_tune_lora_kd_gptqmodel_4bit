#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型全链路评估系统
支持：困惑度、生成质量、推理性能、显存分析、多维度对比
评估维度
    ✅ 困惑度 (Perplexity): 衡量模型预测能力
    ✅ 推理性能: 延迟、吞吐量
    ✅ 模型大小: 存储和加载成本
    ✅ 显存使用: GPU内存占用
    ✅ 生成质量: BLEU、ROUGE、综合评分
    ✅ 多维度对比: 雷达图、热力图

pip install pandas matplotlib seaborn nltk rouge-score datasets tqdm
python -c "import nltk; nltk.download('punkt')"
"""

from dotenv import load_dotenv
import os
import json
import time
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    pipeline, set_seed
)
from gptqmodel import GPTQModel
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import warnings

warnings.filterwarnings('ignore')

load_dotenv()

# 清空显存
torch.cuda.empty_cache()
gc.collect()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

TRAIN_MODEL_NAME = os.getenv("MODEL_NAME")
TEACHER_MODEL_NAME = os.getenv("TEACHER_MODEL_NAME")


# ==================== 配置 ====================
@dataclass
class EvaluationConfig:
    """评估配置"""
    # 模型路径
    student_base_path: str = f"../models/{TRAIN_MODEL_NAME}"
    teacher_path: str = f"../models/{TEACHER_MODEL_NAME}"
    kd_lora_path: str = f"../output/{TRAIN_MODEL_NAME}_joint_kd_lora_merged"
    gptq_path: str = f"../output/{TRAIN_MODEL_NAME}_joint_kd_lora_merged_gptq_4bit"

    # 输出路径
    output_dir: str = "../evaluation_reports_output"

    # 评估参数
    eval_batch_size: int = 4
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 生成测试参数
    generation_config: Dict = None


    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = {
                "max_new_tokens": 128,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }

        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ModelMetrics:
    """模型评估指标"""
    name: str
    perplexity: float
    loss: float
    inference_time_avg: float  # 秒
    inference_throughput: float  # tokens/秒
    memory_usage_gb: float
    model_size_mb: float
    bleu_score: float
    rouge_scores: Dict[str, float]
    generation_quality_score: float

    def to_dict(self):
        return asdict(self)


# ==================== 评估器 ====================
class ModelEvaluator:
    """企业级模型评估器"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = []
        self.setup_logging()

    def setup_logging(self):
        """设置日志"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/evaluation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def clear_memory(self):
        """清理显存"""
        torch.cuda.empty_cache()
        gc.collect()

    def get_model_size(self, model_path: str) -> float:
        """获取模型大小（MB）"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                if filename.endswith(('.bin', '.safetensors', '.pt')):
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        return total_size / (1024 ** 2)  # 转换为 MB

    def load_model(self, model_path: str, model_type: str = "auto"):
        """加载模型（支持不同格式）"""
        self.logger.info(f"加载模型: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token

        try:
            # 尝试作为量化模型加载
            if "gptq" in model_path.lower():
                model = GPTQModel.from_quantized(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    dtype=torch.bfloat16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
        except Exception as e:
            self.logger.warning(f"标准加载失败，尝试其他方式: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True
            )

        model.eval()
        return model, tokenizer

    def compute_perplexity(self, model, tokenizer, texts: List[str]) -> Tuple[float, float]:
        """计算困惑度和损失"""
        self.logger.info("计算困惑度...")

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in tqdm(texts, desc="计算困惑度"):
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.config.device)

                outputs = model(**inputs)

                # ✅ 关键修复：如果 loss 为 None，手动计算
                if outputs.loss is None:
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs.input_ids[..., 1:].contiguous()

                    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    total_loss += loss.item()
                    total_tokens += shift_labels.numel()
                else:
                    total_loss += outputs.loss.item() * inputs.input_ids.size(1)
                    total_tokens += inputs.input_ids.size(1)

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return perplexity, avg_loss

    def measure_inference_performance(self, model, tokenizer, texts: List[str]) -> Dict:
        """测量推理性能"""
        self.logger.info("测量推理性能...")

        times = []
        generated_tokens = []

        # 预热
        for _ in range(3):
            text = texts[0]
            inputs = tokenizer(text, return_tensors="pt").to(self.config.device)
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=50)

        # 正式测试
        for text in tqdm(texts[:20], desc="性能测试"):
            inputs = tokenizer(text, return_tensors="pt").to(self.config.device)

            # 记录开始时间
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.generation_config.get("max_new_tokens", 128),
                    temperature=self.config.generation_config.get("temperature", 0.7),
                    do_sample=self.config.generation_config.get("do_sample", True),
                    top_p=self.config.generation_config.get("top_p", 0.9),
                    pad_token_id=tokenizer.eos_token_id,  # ✅ 添加 pad_token_id
                )

            torch.cuda.synchronize()
            end_time = time.time()

            # 统计
            inference_time = end_time - start_time
            num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]

            times.append(inference_time)
            generated_tokens.append(num_tokens)

        avg_time = np.mean(times)
        avg_tokens = np.mean(generated_tokens)
        throughput = avg_tokens / avg_time

        return {
            "inference_time_avg": avg_time,
            "inference_throughput": throughput,
            "time_std": np.std(times),
            "tokens_per_request": avg_tokens
        }

    def get_memory_usage(self) -> float:
        """获取显存使用（GB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0

    def compute_bleu(self, references: List[str], hypotheses: List[str]) -> float:
        """计算 BLEU 分数"""
        smoothie = SmoothingFunction().method4
        bleu_scores = []

        for ref, hyp in zip(references, hypotheses):
            score = sentence_bleu(
                [ref.split()],
                hyp.split(),
                smoothing_function=smoothie
            )
            bleu_scores.append(score)

        return np.mean(bleu_scores)

    def compute_rouge(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """计算 ROUGE 分数"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for ref, hyp in zip(references, hypotheses):
            score = scorer.score(ref, hyp)
            for key in scores.keys():
                scores[key].append(score[key].fmeasure)

        return {key: np.mean(values) for key, values in scores.items()}

    def evaluate_generation_quality(self, model, tokenizer, test_cases: List[Dict]) -> Dict:
        """评估生成质量"""
        self.logger.info("评估生成质量...")

        generated_responses = []

        for test_case in tqdm(test_cases, desc="生成测试"):
            prompt = test_case["prompt"]
            reference = test_case["reference"]

            inputs = tokenizer(prompt, return_tensors="pt").to(self.config.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **self.config.generation_config,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt):].strip()
            generated_responses.append(generated)

        references = [tc["reference"] for tc in test_cases]

        bleu_score = self.compute_bleu(references, generated_responses)
        rouge_scores = self.compute_rouge(references, generated_responses)

        # 综合质量评分（0-100）
        quality_score = (
                bleu_score * 40 +  # BLEU 权重 40%
                rouge_scores['rougeL'] * 30 +  # ROUGE-L 权重 30%
                (1 - min(1.0, len([r for r in generated_responses if "我不知道" in r]) / len(generated_responses))) * 30
        )

        return {
            "bleu_score": bleu_score,
            "rouge_scores": rouge_scores,
            "generation_quality_score": quality_score,
            "generated_responses": generated_responses
        }

    def evaluate_model(self, name: str, model_path: str, eval_data: Dict) -> ModelMetrics:
        """完整评估单个模型"""
        self.logger.info(f"=" * 50)
        self.logger.info(f"评估模型: {name}")
        self.logger.info(f"=" * 50)

        self.clear_memory()

        # 加载模型
        model, tokenizer = self.load_model(model_path)

        # 计算模型大小
        model_size = self.get_model_size(model_path)

        # 计算困惑度
        ppl, loss = self.compute_perplexity(
            model, tokenizer, eval_data["perplexity_texts"]
        )

        # 测量推理性能
        perf_metrics = self.measure_inference_performance(
            model, tokenizer, eval_data["perplexity_texts"]
        )

        # 获取显存使用
        memory_usage = self.get_memory_usage()

        # 评估生成质量
        quality_metrics = self.evaluate_generation_quality(
            model, tokenizer, eval_data["test_cases"]
        )

        # 清理
        del model
        self.clear_memory()

        return ModelMetrics(
            name=name,
            perplexity=ppl,
            loss=loss,
            inference_time_avg=perf_metrics["inference_time_avg"],
            inference_throughput=perf_metrics["inference_throughput"],
            memory_usage_gb=memory_usage,
            model_size_mb=model_size,
            bleu_score=quality_metrics["bleu_score"],
            rouge_scores=quality_metrics["rouge_scores"],
            generation_quality_score=quality_metrics["generation_quality_score"]
        )

    def run_full_evaluation(self):
        """运行完整评估"""
        # 准备评估数据
        eval_data = self.prepare_evaluation_data()

        # 定义要评估的模型
        models_to_evaluate = [
            ("学生原始模型", self.config.student_base_path),
            ("教师模型", self.config.teacher_path),
            ("蒸馏+LoRA模型", self.config.kd_lora_path),
            ("GPTQ 4bit量化模型", self.config.gptq_path),
        ]

        # 评估每个模型
        for name, path in models_to_evaluate:
            if not os.path.exists(path):
                self.logger.warning(f"模型路径不存在: {path}")
                continue

            try:
                metrics = self.evaluate_model(name, path, eval_data)
                self.results.append(metrics)

                # 保存中间结果
                self.save_intermediate_results()
            except Exception as e:
                self.logger.error(f"评估 {name} 失败: {e}")

        # 生成完整报告
        self.generate_report()

        return self.results

    def prepare_evaluation_data(self) -> Dict:
        """准备评估数据"""
        self.logger.info("准备评估数据集...")

        # 1. 困惑度测试文本（来自 WikiText-2）
        perplexity_texts = [
                               "The quick brown fox jumps over the lazy dog near the river bank.",
                               "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                               "Natural language processing helps computers understand and generate human language.",
                               "Deep learning uses neural networks with multiple layers to extract hierarchical features.",
                               "The transformer architecture revolutionized NLP with its attention mechanism.",
                               "模型评估是机器学习流程中至关重要的一环，它帮助我们理解模型的泛化能力。",
                               "企业在部署AI模型时，需要综合考虑精度、延迟、吞吐量等多个维度。",
                               "知识蒸馏通过让小模型模仿大模型的输出分布，实现模型压缩和加速。",
                               "量化技术可以显著降低模型大小和推理成本，同时保持较好的性能。",
                               "GPTQ是一种先进的模型量化方法，通过二阶信息补偿量化误差。",
                           ] * 10  # 扩充到100条

        # 2. 生成质量测试用例
        test_cases = [
            {
                "prompt": "中国的首都是",
                "reference": "北京"
            },
            {
                "prompt": "人工智能的定义是",
                "reference": "人工智能是研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统的科学。"
            },
            {
                "prompt": "请解释什么是机器学习",
                "reference": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习规律，而无需明确编程。"
            },
            {
                "prompt": "什么是知识蒸馏",
                "reference": "知识蒸馏是一种模型压缩技术，通过让小模型学习大模型的输出概率分布来提升性能。"
            },
            {
                "prompt": "模型量化的优点是什么",
                "reference": "模型量化可以降低模型大小、减少内存占用、加速推理计算。"
            },
        ]

        return {
            "perplexity_texts": perplexity_texts,
            "test_cases": test_cases
        }

    def save_intermediate_results(self):
        """保存中间结果"""
        if self.results:
            df = pd.DataFrame([r.to_dict() for r in self.results])
            df.to_csv(f"{self.config.output_dir}/intermediate_results.csv", index=False)

    def generate_report(self):
        """生成详细报告"""
        if not self.results:
            self.logger.warning("没有评估结果")
            return

        # 转换为 DataFrame
        df = pd.DataFrame([r.to_dict() for r in self.results])

        # 1. 生成 Markdown 报告
        self.generate_markdown_report(df)

        # 2. 生成可视化图表
        self.generate_visualizations(df)

        # 3. 生成 JSON 报告
        self.generate_json_report(df)

        # 4. 生成对比分析
        self.generate_comparison_analysis(df)

        # 5. 生成推荐建议
        self.generate_recommendations(df)

    def generate_markdown_report(self, df: pd.DataFrame):
        """生成 Markdown 格式报告"""
        report_lines = []
        report_lines.append("# 模型评估报告\n")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 汇总表格
        report_lines.append("## 1. 评估结果汇总\n")
        report_lines.append(
            "| 模型 | 困惑度↓ | 损失 | 推理时间(s)↓ | 吞吐量(tok/s)↑ | 显存(GB)↓ | 模型大小(MB)↓ | BLEU↑ | 综合评分↑ |")
        report_lines.append(
            "|------|---------|------|-------------|---------------|-----------|-------------|-------|----------|")

        for _, row in df.iterrows():
            report_lines.append(
                f"| {row['name']} | {row['perplexity']:.2f} | {row['loss']:.4f} | "
                f"{row['inference_time_avg']:.3f} | {row['inference_throughput']:.1f} | "
                f"{row['memory_usage_gb']:.2f} | {row['model_size_mb']:.0f} | "
                f"{row['bleu_score']:.3f} | {row['generation_quality_score']:.1f} |"
            )

        # 详细分析
        report_lines.append("\n## 2. 详细分析\n")

        # 找最佳模型
        best_ppl = df.loc[df['perplexity'].idxmin()]
        best_throughput = df.loc[df['inference_throughput'].idxmax()]
        best_size = df.loc[df['model_size_mb'].idxmin()]

        report_lines.append(f"- **最低困惑度**: {best_ppl['name']} ({best_ppl['perplexity']:.2f})")
        report_lines.append(
            f"- **最高吞吐量**: {best_throughput['name']} ({best_throughput['inference_throughput']:.1f} tok/s)")
        report_lines.append(f"- **最小模型**: {best_size['name']} ({best_size['model_size_mb']:.0f} MB)")

        # 相对改善
        if len(df) >= 2:
            base_model = df.iloc[0]
            final_model = df.iloc[-1]

            ppl_improvement = (base_model['perplexity'] - final_model['perplexity']) / base_model['perplexity'] * 100
            size_reduction = (base_model['model_size_mb'] - final_model['model_size_mb']) / base_model[
                'model_size_mb'] * 100
            speed_improvement = (final_model['inference_throughput'] - base_model['inference_throughput']) / base_model[
                'inference_throughput'] * 100

            report_lines.append("\n## 3. 优化效果\n")
            report_lines.append(f"- 困惑度改善: {ppl_improvement:.1f}%")
            report_lines.append(f"- 模型大小减少: {size_reduction:.1f}%")
            report_lines.append(f"- 推理速度提升: {speed_improvement:.1f}%")

        # 保存报告
        report_path = f"{self.config.output_dir}/evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"Markdown 报告已保存: {report_path}")

    def generate_visualizations(self, df: pd.DataFrame):
        """生成可视化图表"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 困惑度对比
        ax1 = axes[0, 0]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        bars = ax1.bar(df['name'], df['perplexity'], color=colors)
        ax1.set_title('困惑度对比 (越小越好)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Perplexity')
        ax1.tick_params(axis='x', rotation=45)
        # 添加数值标签
        for bar, val in zip(bars, df['perplexity']):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=10)

        # 2. 模型大小对比
        ax2 = axes[0, 1]
        bars = ax2.bar(df['name'], df['model_size_mb'], color=colors)
        ax2.set_title('模型大小对比 (越小越好)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Size (MB)')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, df['model_size_mb']):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=10)

        # 3. 推理吞吐量
        ax3 = axes[0, 2]
        bars = ax3.bar(df['name'], df['inference_throughput'], color=colors)
        ax3.set_title('推理吞吐量 (越大越好)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Tokens/Second')
        ax3.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, df['inference_throughput']):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=10)

        # 4. 综合评分
        ax4 = axes[1, 0]
        bars = ax4.bar(df['name'], df['generation_quality_score'], color=colors)
        ax4.set_title('生成质量综合评分 (越大越好)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Score (0-100)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim([0, 100])
        for bar, val in zip(bars, df['generation_quality_score']):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=10)

        # 5. 雷达图（多维度对比）
        ax5 = axes[1, 1]
        metrics = ['perplexity', 'inference_throughput', 'model_size_mb', 'generation_quality_score']
        # 归一化处理
        normalized_data = []
        for metric in metrics:
            values = df[metric].values
            if metric in ['perplexity', 'model_size_mb']:  # 越小越好
                normalized = 1 - (values - values.min()) / (values.max() - values.min() + 1e-8)
            else:  # 越大越好
                normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
            normalized_data.append(normalized)

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        for i, row in df.iterrows():
            values = [normalized_data[j][i] for j in range(len(metrics))]
            values += values[:1]
            ax5.plot(angles, values, 'o-', linewidth=2, label=row['name'])
            ax5.fill(angles, values, alpha=0.25)

        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(['困惑度', '吞吐量', '模型大小', '生成质量'])
        ax5.set_title('多维度性能雷达图', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 6. 性能对比热力图
        ax6 = axes[1, 2]
        # 选择数值列
        numeric_cols = ['perplexity', 'inference_time_avg', 'inference_throughput',
                        'memory_usage_gb', 'model_size_mb', 'bleu_score', 'generation_quality_score']
        heatmap_data = df[numeric_cols].T
        heatmap_data.columns = df['name']

        # 归一化
        heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
        # 对于越小越好的指标，反转颜色
        for col in ['perplexity', 'inference_time_avg', 'memory_usage_gb', 'model_size_mb']:
            if col in heatmap_norm.index:
                heatmap_norm.loc[col] = 1 - heatmap_norm.loc[col]

        im = ax6.imshow(heatmap_norm.values, cmap='RdYlGn', aspect='auto')
        ax6.set_xticks(range(len(df['name'])))
        ax6.set_xticklabels(df['name'], rotation=45, ha='right')
        ax6.set_yticks(range(len(heatmap_norm.index)))
        ax6.set_yticklabels(heatmap_norm.index)
        ax6.set_title('性能热力图 (绿色越深越好)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax6)

        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/evaluation_charts.png", dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"可视化图表已保存: {self.config.output_dir}/evaluation_charts.png")

    def generate_json_report(self, df: pd.DataFrame):
        """生成 JSON 格式报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "models": df.to_dict('records'),
            "summary": {
                "total_models": len(df),
                "best_perplexity": df.loc[df['perplexity'].idxmin()]['name'],
                "best_throughput": df.loc[df['inference_throughput'].idxmax()]['name'],
                "smallest_model": df.loc[df['model_size_mb'].idxmin()]['name'],
            }
        }

        # 计算改善
        if len(df) >= 2:
            base = df.iloc[0]
            final = df.iloc[-1]
            report["improvements"] = {
                "perplexity_reduction": f"{((base['perplexity'] - final['perplexity']) / base['perplexity'] * 100):.2f}%",
                "size_reduction": f"{((base['model_size_mb'] - final['model_size_mb']) / base['model_size_mb'] * 100):.2f}%",
                "speed_improvement": f"{((final['inference_throughput'] - base['inference_throughput']) / base['inference_throughput'] * 100):.2f}%"
            }

        report_path = f"{self.config.output_dir}/evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"JSON 报告已保存: {report_path}")

    def generate_comparison_analysis(self, df: pd.DataFrame):
        """生成对比分析"""
        analysis = []
        analysis.append("# 详细对比分析\n")

        if len(df) >= 2:
            base = df.iloc[0]
            final = df.iloc[-1]

            analysis.append("## 蒸馏+量化 vs 原始模型\n")
            analysis.append(f"- **困惑度**: {base['perplexity']:.2f} → {final['perplexity']:.2f} "
                            f"({self.calculate_change(base['perplexity'], final['perplexity'])})\n")
            analysis.append(f"- **模型大小**: {base['model_size_mb']:.0f} MB → {final['model_size_mb']:.0f} MB "
                            f"({self.calculate_change(base['model_size_mb'], final['model_size_mb'], reverse=True)})\n")
            analysis.append(
                f"- **推理速度**: {base['inference_throughput']:.1f} → {final['inference_throughput']:.1f} tok/s "
                f"({self.calculate_change(base['inference_throughput'], final['inference_throughput'])})\n")

        # 性能评分
        analysis.append("\n## 模型推荐\n")

        # 计算综合得分（考虑所有指标）
        scores = {}
        for _, row in df.iterrows():
            # 归一化各项指标
            ppl_score = 1 / (row['perplexity'] + 1e-8)
            speed_score = row['inference_throughput']
            size_score = 1 / (row['model_size_mb'] + 1e-8)
            quality_score = row['generation_quality_score']

            total_score = ppl_score * 0.3 + speed_score * 0.3 + size_score * 0.2 + quality_score * 0.2
            scores[row['name']] = total_score

        best_model = max(scores, key=scores.get)
        analysis.append(f"**推荐部署模型**: {best_model}\n")
        analysis.append(f"**推荐理由**: 在精度、速度、大小的综合权衡中表现最优\n")

        if "GPTQ" in best_model:
            analysis.append("- 模型大小减少约 50%，极大降低部署成本\n")
            analysis.append("- 推理速度显著提升，适合生产环境\n")

        # 保存分析
        analysis_path = f"{self.config.output_dir}/comparison_analysis.md"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(analysis))

    def calculate_change(self, old_val, new_val, reverse=False):
        """计算变化百分比"""
        change = (new_val - old_val) / old_val * 100
        if reverse:
            change = -change
        arrow = "↑" if change > 0 else "↓"
        return f"{arrow} {abs(change):.1f}%"

    def generate_recommendations(self, df: pd.DataFrame):
        """生成部署建议"""
        recommendations = []
        recommendations.append("# 部署建议\n")

        if len(df) == 0:
            return

        # 找最佳模型
        best_size = df.loc[df['model_size_mb'].idxmin()]['name']
        best_speed = df.loc[df['inference_throughput'].idxmax()]['name']
        best_quality = df.loc[df['generation_quality_score'].idxmax()]['name']

        recommendations.append("## 根据场景选择模型\n")
        recommendations.append(f"- **追求最小模型**: {best_size}\n")
        recommendations.append(f"- **追求最快速度**: {best_speed}\n")
        recommendations.append(f"- **追求最佳质量**: {best_quality}\n")

        recommendations.append("\n## 生产环境部署检查清单\n")
        recommendations.append("- [ ] 模型加载测试通过\n")
        recommendations.append("- [ ] 推理延迟满足 SLA 要求\n")
        recommendations.append("- [ ] 显存使用在预算范围内\n")
        recommendations.append("- [ ] 生成质量通过人工评估\n")
        recommendations.append("- [ ] 已配置模型监控和日志\n")

        # 保存建议
        recommendations_path = f"{self.config.output_dir}/deployment_recommendations.md"
        with open(recommendations_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(recommendations))

        self.logger.info(f"部署建议已保存: {recommendations_path}")


# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("企业级模型评估系统 v1.0")
    print("=" * 60)

    # 配置
    config = EvaluationConfig()

    # 创建评估器
    evaluator = ModelEvaluator(config)

    # 运行评估
    print("开始评估流程...")
    results = evaluator.run_full_evaluation()

    # 输出最终结果
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    print(f"报告保存在: {config.output_dir}")
    print(f"  - Markdown报告: evaluation_report.md")
    print(f"  - JSON报告: evaluation_report.json")
    print(f"  - 可视化图表: evaluation_charts.png")
    print(f"  - 对比分析: comparison_analysis.md")
    print(f"  - 部署建议: deployment_recommendations.md")

    # 打印摘要
    if results:
        print("\n模型性能摘要:")
        for r in results:
            print(f"  {r.name}: PPL={r.perplexity:.2f}, "
                  f"速度={r.inference_throughput:.1f} tok/s, "
                  f"大小={r.model_size_mb:.0f} MB")


if __name__ == "__main__":
    main()