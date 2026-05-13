# monitor.py
import psutil
import GPUtil
import torch
from prometheus_client import start_http_server, Gauge

# Prometheus 指标
REQUEST_COUNT = Gauge('model_requests_total', 'Total requests')
REQUEST_LATENCY = Gauge('model_request_latency_ms', 'Request latency')
GPU_MEMORY = Gauge('gpu_memory_used_gb', 'GPU memory used')
TOKENS_PER_SEC = Gauge('tokens_per_second', 'Inference tokens per second')


def monitor_resources():
    """监控系统资源"""
    # GPU监控
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
        GPU_MEMORY.set(gpu_memory)
        print(f"GPU Memory: {gpu_memory:.2f} GB")

    # CPU监控
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    print(f"CPU: {cpu_percent}%, Memory: {memory_percent}%")


# 启动监控服务
start_http_server(9090)