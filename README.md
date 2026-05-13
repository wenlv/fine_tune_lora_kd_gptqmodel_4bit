用conda管理python:
# 创建 py3.11 环境
conda create -n py311 python=3.11 -y

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py311

# 彻底清空坏包
pip freeze | xargs pip uninstall -y

pip install torch==2.4.0+cu121 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

pip install auto-gptq==0.7.1

pip install transformers==4.45.2  datasets  sentencepiece peft==0.11.1 accelerate==0.28.0


autodl数据盘不够用了
    #查看文件占用大小
    du -sh /root/* /root/.[!.]* 2>/dev/null | sort -h
    
    #查看autodl-tmp目录下的文件
    du -sh /root/autodl-tmp/* /root/autodl-tmp/.[!.]* 2>/dev/null | sort -h
    
    # 1. 查看回收站内容
    ls -la /root/autodl-tmp/.Trash-0/
    
    # 2. 查看回收站中文件的大小
    du -sh /root/autodl-tmp/.Trash-0/*
    
    # 3. 彻底删除回收站
    rm -rf /root/autodl-tmp/.Trash-0
    
    # 或者使用 AutoDL 的回收站清空命令（如果有）
    # rm -rf /root/autodl-tmp/.local/share/Trash
    ============================

autodl系统盘满了不够用了：
    先看磁盘占用（确认情况）
    df -h /

    conda 缓存（最大垃圾场）
    conda clean --all -y
    rm -rf /root/miniconda3/pkgs/*

    pip 缓存
    pip cache purge
    rm -rf ~/.cache/pip/*

    huggingface 缓存（模型缓存）
    rm -rf ~/.cache/huggingface/*

    torch 缓存
    rm -rf ~/.cache/torch/*

    临时文件 /tmp   
    rm -rf /tmp/*
    
    jupyter 回收站
    rm -rf /root/.local/share/Trash/*

    python 垃圾文件
    find /root -name "__pycache__" -type d -exec rm -rf {} +
    find /root -name "*.pyc" -delete

    docker 垃圾（如果有）
    docker system prune -a --volumes -f
    清理完再看空间
    df -h /
 
    


添加.env文件
cat > .env << EOF
MODEL_NAME=qwen2_7b
MODEL_NAME="Qwen1.5-1.8B-Chat"
TEACHER_MODEL_NAME="Qwen2.5-7B-Instruct"
OUTPUT_DIR=../output
EOF

查看是否添加成功
ls -a

download_filter_models.py:
    pip install sentence-transformers  modelscope speedtest  dotenv

fake_qat
    创建新的版本：fakequant
    conda create -n fakequant python=3.11 -y
    conda activate  fakequant


    查看conda有多少可用的版本
    conda env list
   
    conda删除指定的版本
    conda remove -n quant_new --all -y
    conda remove -n quantize --all -y
    
    # 1. 安装 PyTorch 2.6 + CUDA12.1（最稳定）
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
    
    # 2. 核心：必须用这个版本，自带 FakeQuantizationConfig
    pip install transformers==4.48.1
    pip install --upgrade transformers
    
    # 3. 大模型必备
    pip install accelerate==1.5.0
    
    # 4. 你代码里用到的包
    pip install python-dotenv datasets sentencepiece protobuf
gptq
    pip install


gptqmodel实现基座模型 → Fake 伪量化 → 知识蒸馏 (KD) → LoRA-QAT  → GPTQModel 4bit 量化 → PPL 评估  python=3.11
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126    
    pip install gptqmodel peft transformers  python-dotenv sentencepiece
    pip install bitsandbytes
    # 升级 optimum 到最新版本
    pip install --upgrade optimum>=1.24.0
    
    # 或者如果不需要 gptqmodel，直接卸载
    pip uninstall gptqmodel -y

评估维度
    pip install pandas matplotlib seaborn nltk rouge-score datasets tqdm
    python -c "import nltk; nltk.download('punkt')"
        
    cd evaluation_reports
    # 查看 Markdown 报告
    cat evaluation_report.md
    # 查看可视化图表
    open evaluation_charts.png



# 构建镜像
docker build -t quantized-llm:latest .

# 运行容器
docker run --gpus all -p 8000:8000 quantized-llm:latest
  