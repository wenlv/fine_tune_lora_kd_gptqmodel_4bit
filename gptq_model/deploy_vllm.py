# deploy_vllm.py
from vllm import LLM, SamplingParams

# 加载模型（vLLM 自动识别 GPTQ）
llm = LLM(
    model="../output/Qwen1.5-1.8B-Chat_joint_kd_lora_merged_gptq_4bit",
    trust_remote_code=True,
    tensor_parallel_size=1,  # 单GPU
    gpu_memory_utilization=0.9,
    quantization="gptq",  # 指定量化类型
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    repetition_penalty=1.1,
)

# 批量推理
prompts = [
    "什么是人工智能？",
    "解释一下深度学习",
    "机器学习有哪些类型？"
]

outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")