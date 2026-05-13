# download_models.py
from dotenv import load_dotenv
import os
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from modelscope import snapshot_download

load_dotenv()

print("="*60)
print("开始下载所需的模型...")
print("="*60)

models_list_path=os.getenv("MODEL_LIST_PATH")
base_model=os.getenv("MODEL_LIST_OUTPUT_PATH")

# 创建模型目录
os.makedirs(base_model, exist_ok=True)
local_dir=f"{base_model}"

# 下载失败的模型列表
fail_model_list=[]

# 从文件中读取要下载的模型信息
print('从文件中读取要下载的模型信息')

try:
    with open(models_list_path,'r',encoding='utf-8') as f:
        models_list=json.load(f)

    print(f'读取成功: 共找到 {len(models_list)} 个模型需要下载')
except Exception as e:
    print(f'模型信息读取失败: {str(e)}')
    exit(1)

def model_scope_load(model_id,local_dir,model_name):
    """从 ModelScope 下载模型（通用下载）"""
    model_dir = snapshot_download(
        model_id=model_id,  # 模型的唯一标识ID，例如 'qwen/Qwen1.5-1.8B-Chat'
        local_dir=local_dir,  # 可选，指定模型的下载根目录
        revision='master',  # 可选，指定版本，默认为 'master'
        ignore_patterns=["imgs/*", "*.DS_Store"]
    )
    print(f"ModelScope 下载成功: {model_name} -> {model_dir}")

def hugging_face_load(model_id,local_dir,model_name):
    """从 HuggingFace 下载分类模型并保存"""
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)

    print(f"HuggingFace 下载成功: {model_name}")

def check_quantize_config(local_dir):
    """手动创建手动创建 GPTQ 模型的 quantize_config.json"""
    quantize_config_path = os.path.join(local_dir, 'quantize_config.json')
    if not os.path.exists(quantize_config_path):
        print("     正在创建 GPTQ 配置文件 quantize_config.json...")
        config = {
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
            "true_sequential": True,
            "model_name_or_path": 'qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4'
        }

        config_path = os.path.join(quantize_config_path, 'quantize_config.json')
        with open(quantize_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ 配置文件创建完成: {quantize_config_path}")
    else:
        print("   ✅ GPTQ 配置文件已存在")


def load_model():
    # 批量下载所有模型
    print("开始批量下载模型...")

    for mode_info in models_list:
        model_name = mode_info.get('modelName', '未知模型')
        model_id = mode_info.get('modelID', '')

        print(f"正在下载: {model_name}")

        model_path = os.path.join(base_model, model_name)

        if not os.path.exists(model_path):
            try:
                # 优先从 ModelScope 下载
                model_scope_load(model_id,model_path,model_name)
            except Exception as e:
                print(f"ModelScope 下载失败，尝试 HuggingFace: {str(e)}")
                try:
                    # 降级从 HuggingFace 下载
                    hugging_face_load(model_id,model_path,model_name)
                except Exception as err:
                    fail_model_list.append(mode_info)
                    print(f"❌ {model_name} 模型下载失败: {str(err)}")
        else:
            print(f"✅ {model_name}模型已存在")

        # 特殊处理：GPTQ 模型自动生成配置文件
        if "GPTQ" in model_name or "Int4" in model_name:
            check_quantize_config(model_path)

        print("-" * 50)



# ===================== 执行下载 =====================
if __name__ == "__main__":
    load_model()

    # 最终统计
    print("\n" + "="*60)
    print("所有模型处理完成！")
    print("="*60)
    total = len(models_list)
    success = total - len(fail_model_list)
    fail = len(fail_model_list)
    print(f'总计：{total} 个 | 成功：{success} 个 | 失败：{fail} 个')

    if fail_model_list:
        print("\n❌ 下载失败的模型：")
        for m in fail_model_list:
            print(f"  - {m.get('modelName')}")
