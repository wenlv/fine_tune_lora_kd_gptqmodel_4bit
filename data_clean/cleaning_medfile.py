import re   # 导入正则模块，用于模式匹配
import json # 导入 JSON 模块，用于读写 JSONL 文件


def parse_dialogue(dialogue_text: str):
    """把 Dialogue 部分解析成 ChatML 格式"""
    messages = []  
    # 用正则切分对话，保留“病人：”和“医生：”作为分隔符
    parts = re.split(r"(病人：|医生：)", dialogue_text.strip())
    # 定义角色映射：病人→user，医生→assistant
    role_map = {"病人：": "user", "医生：": "assistant"}

    current_role = None  # 当前说话角色
    for part in parts:   # 遍历每个分割结果
        part = part.strip()  # 去掉前后空格
        if not part:         # 空字符串跳过
            continue
        if part in role_map: # 如果是角色标签
            current_role = role_map[part]  # 设置当前角色
        else:
            if current_role: # 如果有角色
                # 添加消息（角色 + 内容）
                messages.append({"role": current_role, "content": part})
    return messages


def clean_text(text: str) -> str:
    """清洗 description 文本，去掉提示词"""
    # 去掉描述性的前缀
    text = re.sub(r"病情描述（.*?）[:：]", "", text)
    text = re.sub(r"曾经治疗情况.*?[:：]", "", text)
    text = re.sub(r"和效果[:：]", "", text)
    text = re.sub(r"想得到怎样的帮助[:：]?", "", text)

    # 去掉多余空格/换行
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def merge_sentences(text: str) -> str:
    """把多句合并成一句"""
    # 把句号/分号替换为逗号，避免拆开
    text = re.sub(r"[。；;，,]\s*", "，", text)
    return text


def convert_description_to_qa(description: str):
    """把 Description 转换成问答对"""
    messages = []

    # 提取疾病
    disease_match = re.search(r"疾病[:：](.*?)(?:内容|$)", description, re.S)
    disease = clean_text(disease_match.group(1)) if disease_match else ""

    # 提取病情描述
    cond_match = re.search(r"内容[:：](.*?)(?:曾经治疗情况|想得到怎样的帮助|$)", description, re.S)
    condition = clean_text(cond_match.group(1)) if cond_match else ""

    # 提取既往治疗
    prev_match = re.search(r"曾经治疗情况.*?(?:[:：]|)(.*?)(?:想得到怎样的帮助|$)", description, re.S)
    previous = clean_text(prev_match.group(1)) if prev_match else ""

    # 提取帮助诉求
    help_match = re.search(r"想得到怎样的帮助[:：]?(.*)", description, re.S)
    help_info = clean_text(help_match.group(1)) if help_match else ""

    # 合并句子
    disease = merge_sentences(disease)
    condition = merge_sentences(condition)
    previous = merge_sentences(previous)
    help_info = merge_sentences(help_info)

    # 组装问答
    if disease:
        messages.append({"role": "assistant", "content": "请问您主要的疾病是什么？"})
        messages.append({"role": "user", "content": disease})
    if condition:
        messages.append({"role": "assistant", "content": "请介绍一下发病时间、主要症状和就诊医院等情况。"})
        messages.append({"role": "user", "content": condition})
    if previous:
        messages.append({"role": "assistant", "content": "您之前接受过哪些治疗？效果如何？"})
        messages.append({"role": "user", "content": previous})
    if help_info:
        messages.append({"role": "assistant", "content": "您希望得到怎样的帮助？"})
        messages.append({"role": "user", "content": help_info})

    return messages


def parse_blocks(text: str):
    """解析整个文本，拆分成病例记录"""
    records = []
    # 按照 id=数字 分割成多个病例块
    blocks = re.split(r"\nid=\d+\n", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 提取 Description
        desc_match = re.search(r"Description\s*(.+?)\n\nDialogue", block, re.S)
        description = desc_match.group(1).strip() if desc_match else ""

        # 提取 Dialogue
        dialogue_match = re.search(r"Dialogue\s*(.+)", block, re.S)
        dialogue = dialogue_match.group(1).strip() if dialogue_match else ""

        records.append({
            "Description": description,
            "Dialogue": dialogue
        })
    return records


def convert_record(record: dict):
    """把一条记录转为 ChatML 格式"""
    description = record.get("Description", "")
    dialogue = record.get("Dialogue", "")

    # 添加 system 提示
    system_prompt = "你是一名专业医生，以下是病人的咨询，请给出耐心、专业的解答。"
    messages = [{"role": "system", "content": system_prompt}]

    # Description 转为问答
    if description:
        messages.extend(convert_description_to_qa(description))

    # 添加 Dialogue 转换结果
    messages.extend(parse_dialogue(dialogue))

    return {"messages": messages}


def main():
    # 输入文件（原始文本）
    input_file = "../data/source_data/medfile.txt"
    # 输出文件（JSONL 格式）
    output_file = "../data/data_cleaning/cleaning_medfile.jsonl"

    # 读取原始文本
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # 解析成结构化数据
    records = parse_blocks(text)

    # 最多保留 10000 条
    records = records[:10000]

    # 写入 JSONL 文件
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            chatml_obj = convert_record(record)
            f.write(json.dumps(chatml_obj, ensure_ascii=False) + "\n")

    # 打印提示
    print(f"✅ 已完成转换，共输出 {len(records)} 条数据，结果保存在 {output_file}")


if __name__ == "__main__":
    main()  # 运行主程序
