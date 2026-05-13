##构建运行：
#    # docker build -t llm-prod:v1 .
#    # docker run -d --gpus all -p 8000:8000 --name llm-prod llm-prod:v1
#
#FROM nvidia/cuda:12.1-base-ubuntu22.04
#
#WORKDIR /app
#
#RUN apt-get update && apt-get install -y python3 python3-pip
#
#COPY requirements2.txt .
#RUN pip3 install --no-cache-dir -r requirements2.txt
#
#COPY api_prod.py .
#COPY final_4bit_gptq /app/final_4bit_gptq
#
#EXPOSE 8000
#
#CMD ["python3","api_prod.py"]

# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY deploy_api.py .
COPY ./output /app/models

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "deploy_api.py"]