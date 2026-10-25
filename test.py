import argparse
import torch
from transformers import AutoProcessor

import os
from datasets import load_dataset

# dataset = load_dataset("/media/user/e/xyk/eagle-eye2/eagle-eye2/textvqa.py", trust_remote_code=True, split="train")


import torch

path = "/media/user/d/xyk/data1/ScienceQA/qwen2.5vl_distill/0/data_0.ckpt"
data = torch.load(path, map_location="cpu")
print(data.keys())

# for k in ["pixel_values", "image_grid_thw"]:
#     x = data.get(k, None)
#     print(f"\n[{k}]")
#     print("type:", type(x))
#     if isinstance(x, torch.Tensor):
#         print("shape:", tuple(x.shape))
#         print("dtype:", x.dtype)
#         print("numel:", x.numel())
#         print("has_nan:", torch.isnan(x.float()).any().item())
#         print("has_inf:", torch.isinf(x.float()).any().item())
#         print("preview:", x.reshape(-1)[:20])
#     else:
#         print("value:", x)

# print(dataset)

# import torch, glob
# f = glob.glob("/media/user/e/xyk/data/qwen2.5vl_3B/qwen2.5vl_3B_0_67999/0/*.ckpt")[0]
# d = torch.load(f, map_location="cpu")
# print(d.keys())
# print(d["hidden_state"].shape)
# print(d["input_ids"].shape)
# print(d["loss_mask"].sum(), d["loss_mask"].shape)
# print(torch.isfinite(d["hidden_state"]).all())

# parser = argparse.ArgumentParser()
# parser.add_argument("--ckpt", default='/media/user/d/xyk/data1/llava_instruct/qwen2.5vl/qwen2.5vl_7B_0_67999/0/data_0.ckpt')
# parser.add_argument("--model_path", default='/media/user/d/xyk/model/Qwen2.5-VL-7B-Instruct')
# args = parser.parse_args()

# # 1️⃣ 读取 ckpt
# data = torch.load(args.ckpt, map_location="cpu")
# print(data.keys())

# import torch
# import os

# data_dir = "/media/user/d/xyk/data1/llava_instruct/qwen2.5vl/qwen2.5vl_7B_0_67999/0"

# for i in range(10):
#     path = os.path.join(data_dir, f"data_{i}.ckpt")
#     data = torch.load(path, map_location="cpu")

#     if "sample_score" in data:
#         print(f"data_{i}.ckpt sample_score =", data["sample_score"])
#     else:
#         print(f"data_{i}.ckpt 没有 sample_score 字段")
# random_select_25.py
# -*- coding: utf-8 -*-

# import os
# import re
# import random
# import shutil
# from tqdm import tqdm


# SRC_DIR = "/media/user/d/xyk/data1/llava_instruct/qwen2.5vl/qwen2.5vl_7B_0_67999/0"
# DST_DIR = "/media/user/d/xyk/data1/llava_instruct/random_0.25"

# RATIO = 0.25
# SEED = 42   # 固定随机种子，保证每次抽到的数据一致


# def list_data_files(path: str, pattern: str = r"data_(\d+)\.ckpt$"):
#     files_with_idx = []
#     rgx = re.compile(pattern)

#     for root, _, files in os.walk(path):
#         for file in files:
#             m = rgx.match(file)
#             if m:
#                 files_with_idx.append((int(m.group(1)), os.path.join(root, file)))

#     files_with_idx.sort(key=lambda x: x[0])
#     return [fp for _, fp in files_with_idx]


# def main():
#     os.makedirs(DST_DIR, exist_ok=True)

#     files = list_data_files(SRC_DIR)
#     print(f"[Found] 原始数据数量: {len(files)}")

#     random.seed(SEED)
#     select_num = int(len(files) * RATIO)

#     selected_files = random.sample(files, select_num)

#     print(f"[Select] 随机选择数量: {select_num}")
#     print(f"[Output] {DST_DIR}")

#     for fp in tqdm(selected_files, desc="Copying"):
#         filename = os.path.basename(fp)
#         dst_path = os.path.join(DST_DIR, filename)
#         shutil.copy2(fp, dst_path)

#     # 保存抽样清单，方便之后确认随机选了哪些
#     list_path = os.path.join(DST_DIR, "selected_files.txt")
#     with open(list_path, "w", encoding="utf-8") as f:
#         for fp in selected_files:
#             f.write(fp + "\n")

#     print("[Done] 随机 25% 数据复制完成")
#     print(f"[List] {list_path}")


# if __name__ == "__main__":
#     main()