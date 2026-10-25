# copy_top25_sample_score.py
# -*- coding: utf-8 -*-

import os
import re
import shutil
import torch
from tqdm import tqdm


SRC_DIR = "/media/user/d/xyk/data1/llava_instruct/qwen2.5vl/qwen2.5vl_7B_0_67999/0"
DST_DIR = "/media/user/d/xyk/data1/llava_instruct/sample_level_0.25_qwen2.5vl7B"

TOP_RATIO = 0.25


def ordered_list_files(path: str, pattern: str = r"data_(\d+)\.ckpt$"):
    files_with_numbers = []
    rgx = re.compile(pattern)

    for root, _, files in os.walk(path):
        for file in files:
            m = rgx.match(file)
            if m:
                files_with_numbers.append((int(m.group(1)), os.path.join(root, file)))

    files_with_numbers.sort(key=lambda x: x[0])
    return [fp for _, fp in files_with_numbers]


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    files = ordered_list_files(SRC_DIR)
    print(f"[Found] 原始数据数量: {len(files)}")

    scored_files = []
    missing_score_files = []

    for fp in tqdm(files, desc="Reading sample_score"):
        data = torch.load(fp, map_location="cpu")

        if "sample_score" not in data:
            missing_score_files.append(fp)
            continue

        score = float(data["sample_score"])
        scored_files.append((fp, score))

    print(f"[Valid] 有 sample_score 的数据数量: {len(scored_files)}")
    print(f"[Missing] 没有 sample_score 的数据数量: {len(missing_score_files)}")

    if len(scored_files) == 0:
        raise RuntimeError("没有任何 ckpt 包含 sample_score，请先完整运行打分脚本。")

    # 按 sample_score 从高到低排序
    scored_files.sort(key=lambda x: x[1], reverse=True)

    select_num = int(len(scored_files) * TOP_RATIO)
    select_num = max(1, select_num)

    selected_files = scored_files[:select_num]

    print(f"[Select] 选择前 {TOP_RATIO:.0%} 数据: {select_num} 条")
    print(f"[Top Score] 最高分: {selected_files[0][1]}")
    print(f"[Last Score] 入选最低分: {selected_files[-1][1]}")

    # 复制文件
    for fp, score in tqdm(selected_files, desc="Copying selected files"):
        filename = os.path.basename(fp)
        dst_path = os.path.join(DST_DIR, filename)
        shutil.copy2(fp, dst_path)

    # 保存一份入选清单，方便你检查
    list_path = os.path.join(DST_DIR, "selected_files.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for fp, score in selected_files:
            f.write(f"{os.path.basename(fp)}\t{score}\t{fp}\n")

    print("[Done] 复制完成")
    print(f"[Output Dir] {DST_DIR}")
    print(f"[Selected List] {list_path}")


if __name__ == "__main__":
    main()