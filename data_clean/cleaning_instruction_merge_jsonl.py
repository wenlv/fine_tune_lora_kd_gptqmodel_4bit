# -*- coding: utf-8 -*-
import json

def merge_jsonl(file1, file2, output_file):
    """
    合并两个 JSONL 文件到一个新文件
    :param file1: 第一个输入 JSONL 文件
    :param file2: 第二个输入 JSONL 文件
    :param output_file: 输出合并后的 JSONL 文件
    """
    with open(output_file, "w", encoding="utf-8") as fout:
        # 逐行写入第一个文件
        with open(file1, "r", encoding="utf-8") as f1:
            for line in f1:
                line = line.strip()
                if line:  # 跳过空行
                    json_obj = json.loads(line)  # 确认是合法 JSON
                    fout.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

        # 逐行写入第二个文件
        with open(file2, "r", encoding="utf-8") as f2:
            for line in f2:
                line = line.strip()
                if line:
                    json_obj = json.loads(line)
                    fout.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"✅ 已合并 {file1} 和 {file2} 到 {output_file}")


if __name__ == "__main__":
    file1 = "../data/source_data/medfile_10000.jsonl"          # 输入文件1
    file2 = "../data/data_cleaning/cleaning_instruction_5000_chatml.jsonl"          # 输入文件2
    output_file = "../data/data_cleaning/cleaning_instruction_mixed_15000.jsonl"   # 输出文件
    merge_jsonl(file1, file2, output_file)
