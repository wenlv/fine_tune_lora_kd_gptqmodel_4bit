import json   # 导入 JSON 库，用于读取和写入 JSON/JSONL 文件
import os

def convert_to_chatml(example, system_prompt="你是一个乐于助人的助手。"):
    """
    把一条 instruction/input/output 格式的数据转换为 ChatML 格式
    参数:
        example: dict, 输入的一条样本，包含 instruction/input/output
        system_prompt: str, 系统角色提示语，默认是“你是一个乐于助人的助手。”
    返回:
        chatml_example: dict, 转换后的 ChatML 格式
    """
    # 构造 user 内容：如果有 input，就拼接到 instruction 后面
    if example.get("input") and example["input"].strip():
        user_content = f"{example['instruction']}\n输入: {example['input']}"
    else:
        user_content = example["instruction"]

    # 构造 ChatML 格式
    chatml_example = {
        "messages": [
            {"role": "system", "content": system_prompt},    # 系统角色提示
            {"role": "user", "content": user_content},       # 用户输入 (instruction + input)
            {"role": "assistant", "content": example["output"]}  # 助手回答
        ]
    }
    return chatml_example


def convert_jsonl_to_chatml(input_file, output_file, system_prompt="你是一个乐于助人的助手。"):
    """
    读取 instruction 格式的 JSONL 文件，转换为 ChatML 格式 JSONL 文件
    参数:
        input_file: str, 输入文件路径 (instruction/input/output 格式)
        output_file: str, 输出文件路径 (ChatML 格式)
        system_prompt: str, 系统提示词
    """
    # 打开输入文件和输出文件，逐行读取并写入
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:   # 逐行读取 JSONL
            example = json.loads(line.strip())   # 解析为字典
            chatml_example = convert_to_chatml(example, system_prompt)   # 转换为 ChatML
            fout.write(json.dumps(chatml_example, ensure_ascii=False) + "\n")  # 写入一行


if __name__ == "__main__":
    input_file = "../data/source_data/instruction_5000.jsonl"           # 输入文件路径
    output_file = "../data/data_cleaning/cleaning_instruction_5000_chatml.jsonl"   # 输出文件路径

    convert_jsonl_to_chatml(input_file, output_file)   # 调用函数执行转换
    print(f"转换完成！结果已保存到 {output_file}")   # 提示转换完成
