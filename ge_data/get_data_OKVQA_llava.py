import argparse
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description="A-OKVQA -> LLaVA ckpt (gold answer + rationale)")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=10000)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument(
    "--outdir",
    type=str,
    default="/media/user/d/xyk/data1/A-OKVQA/llava_gold_answer",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/media/user/d/xyk/data1/A-OKVQA",
    help="HF datasets 格式的本地路径（你现在的 parquet / arrow 数据集目录）",
)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--num_proc", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

# -----------------------------
# Env
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

# -----------------------------
# Model / Processor
# -----------------------------
bigname = "/media/user/d/xyk/model/llava-1.5-7b-hf"

bigmodel = LlavaForConditionalGeneration.from_pretrained(
    bigname,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
bigprocessor = AutoProcessor.from_pretrained(bigname, use_fast=False)

# 和你 get_data_all_llava 对齐：patch_size 设定
bigprocessor.patch_size = bigmodel.config.vision_config.patch_size
tokenizer = bigprocessor.tokenizer


# -----------------------------
# Dataset builder (对齐 get_data_all_llava 的 conversation + loss_mask 逻辑)
# -----------------------------
def build_aokvqa_dataset(processor, split="train"):
    ds = load_dataset(args.dataset_path, split=split)
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(args.start, args.end))

    original_columns = ds.column_names
    num_proc = args.num_proc

    def preprocess_function(examples):
        new_examples = {
            "conversation": [],
            "input_ids": [],
            "pixel_values": [],
            "loss_mask": [],
        }

        # 这里 examples["image"][i] 已经是 PIL.Image（你验证过）
        for i in range(len(examples["question"])):
            image = examples["image"][i]
            question = examples["question"][i]

            # gold answer
            choices = examples["choices"][i]
            correct_idx = examples["correct_choice_idx"][i]
            answer = choices[correct_idx]

            # rationale list[str]
            rationales = examples.get("rationales", None)
            if rationales is None:
                rationale_text = ""
            else:
                # parquet 里 rationales 是 list[str]
                rs = rationales[i]
                rationale_text = " ".join(rs) if isinstance(rs, (list, tuple)) else str(rs)

            # ---- 关键：一定要 user + assistant 两回合，Answer/Reason 放 assistant ----
            # 并且 assistant 回合末尾加 eos_token（对齐 all_llava）
            con = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},                 # 这里不要塞 PIL
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"Answer: {answer}. Reason: {rationale_text}{tokenizer.eos_token}"}
                    ]
                }
            ]
            # 对齐 all_llava：去掉问题里可能自带的 <image>
            con[0]["content"][1]["text"] = re.sub(r"\n?<image>\n?", "", con[0]["content"][1]["text"])

            conversation = processor.apply_chat_template(con, tokenize=False, add_generation_prompt=False)
            inputs = processor(images=image, text=conversation, return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            pixel_values = inputs["pixel_values"]

            # ---- loss_mask：完全复制 get_data_all_llava 的逻辑（只对 assistant 部分算 loss）----
            loss_mask = torch.ones_like(input_ids)
            sep = " " + "ASSISTANT" + ": "

            turns = conversation.split(tokenizer.eos_token)
            cur_len = 1
            loss_mask[:cur_len] = 0

            for _, turn in enumerate(turns):
                if turn.strip() == "":
                    break

                turn_len = len(processor(images=image, text=turn, return_tensors="pt").input_ids[0])

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                instruction_len = len(
                    processor(images=image, text=parts[0], return_tensors="pt").input_ids[0]
                ) - 2

                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len

            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["pixel_values"].append(pixel_values)
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        batch_size=args.batch_size,
        remove_columns=original_columns,
        load_from_cache_file=False,
    )
    ds.set_format(type="torch")
    return ds


# -----------------------------
# Save ckpt (对齐你的 all_llava 输出字段风格)
# -----------------------------
@torch.no_grad()
def process_and_save(one_item, outdir):
    input_ids = one_item["input_ids"]  # (1, L)
    pixel_values = one_item["pixel_values"]  # (1, ...)
    loss_mask = one_item["loss_mask"]  # (1, L)

    # attention_mask：pad != 0
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    outs_big = bigmodel(
        input_ids=input_ids.cuda(non_blocking=True),
        pixel_values=pixel_values.cuda(non_blocking=True),
        attention_mask=attention_mask.cuda(non_blocking=True),
        output_hidden_states=True,
    )
    inputs_embeds = outs_big.hidden_states[0]
    hidden_state_big = outs_big.hidden_states[-1]

    td = {
        "input_ids": input_ids.cpu()[0],
        "inputs_embeds": inputs_embeds.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": loss_mask.cpu()[0],
    }

    os.makedirs(outdir, exist_ok=True)
    idx = len(os.listdir(outdir))
    torch.save(td, f"{outdir}/data_{idx}.ckpt")


# -----------------------------
# Main
# -----------------------------
ds = build_aokvqa_dataset(bigprocessor, split=args.split)
print(ds)

outdir = f"{args.outdir}/{args.index}"
for item in ds:
    process_and_save(item, outdir)