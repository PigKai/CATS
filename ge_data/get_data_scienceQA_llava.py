#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScienceQA parquet -> ckpt (LLaVA teacher states)

Output ckpt format (same as your Qwen2.5-VL script):
  {
    "input_ids":      Tensor [T] int64,
    "inputs_embeds":  Tensor [T,H] float16/float32,
    "hidden_state":   Tensor [T,H] float16/float32 (last layer),
    "loss_mask":      Tensor [T] int64 (0/1),
  }

Main differences vs Qwen2.5-VL:
- model/processor are LLaVA
- loss_mask uses a robust "prompt_len vs full_len" token split, not template parsing
"""

import argparse
import os
import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor

# LLaVA class name depends on transformers version
try:
    from transformers import LlavaForConditionalGeneration
    _LLAVA_CLS = LlavaForConditionalGeneration
except Exception:
    # fallback: some versions expose it under AutoModelForVision2Seq
    from transformers import AutoModelForVision2Seq
    _LLAVA_CLS = AutoModelForVision2Seq


# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description="ScienceQA -> ckpt (LLaVA)")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=10000)
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])

parser.add_argument("--outdir", type=str, default="/media/user/d/xyk/data1/ScienceQA/llava_gold_answer")
parser.add_argument("--parquet_path", type=str, default="/media/user/d/xyk/data1/ScienceQA/train-00000-of-00001-1028f23e353fbe3e.parquet")

# Change this to your LLaVA weights path
parser.add_argument("--model_path", type=str, default="/media/user/d/xyk/model/llava-1.5-7b-hf")

parser.add_argument("--max_len", type=int, default=2048)
parser.add_argument("--include_solution", action="store_true")
parser.add_argument("--skip_no_image", action="store_true", default=False, help="skip samples without image")

# dtype options for stability; default fp16 to match your current pipeline
parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))


def _get_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


DTYPE = _get_dtype(args.dtype)


# -----------------------------
# Model / Processor
# -----------------------------
bigmodel = _LLAVA_CLS.from_pretrained(
    args.model_path,
    device_map="auto",
    torch_dtype=DTYPE,
)
bigprocessor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
bigmodel.eval()


# -----------------------------
# Helpers
# -----------------------------
import io
from PIL import Image

def decode_image_from_parquet(ex):
    img_field = ex.get("image", None)
    if img_field is None:
        return None
        
    # 情况 1: 如果已经是 PIL Image 对象
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    
    # 情况 2: 处理你提供的 {'bytes': b'...', 'path': '...'} 字典格式
    if isinstance(img_field, dict):
        img_bytes = img_field.get("bytes", None)
        if img_bytes is not None:
            try:
                # 必须确保上面 import io 成功了
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"解码失败: {e}")
                return None
    
    # 情况 3: 字节流直接在字段里
    if isinstance(img_field, bytes):
        return Image.open(io.BytesIO(img_field)).convert("RGB")

    return None

def build_conversation_sqa(ex, include_solution=False):
    q = ex["question"]
    choices = ex["choices"]
    ans_idx = int(ex["answer"])
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    options_text = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])
    hint = ex.get("hint", "")

    prompt = ""
    if hint:
        prompt += f"Context: {hint}\n\n"
    prompt += f"Question: {q}\n\n"
    prompt += f"Options:\n{options_text}\n"
    prompt += (
        "Please select the correct answer from the options above. "
        "Only respond with the letter (A, B, C, or D) of your choice."
    )

    ans_letter = letters[ans_idx]
    ans_text = choices[ans_idx]
    if include_solution and ex.get("solution"):
        assistant_text = f"{ans_letter}. {ans_text}\nExplanation: {ex['solution']}"
    else:
        assistant_text = f"{ans_letter}. {ans_text}"

    # LLaVA / transformers chat template can consume this messages format
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    return messages


def apply_template(messages, add_generation_prompt: bool):
    """
    Return template string (not tokenized), using processor's chat template if present.
    """
    if hasattr(bigprocessor, "apply_chat_template"):
        return bigprocessor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    # Fallback: if no chat template exists, just concatenate (rare)
    # Note: you probably won't hit this for standard LLaVA checkpoints.
    s = ""
    for m in messages:
        role = m["role"]
        # flatten content
        parts = []
        for c in m.get("content", []):
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "image":
                parts.append("<image>")
        s += f"[{role}]\n" + "\n".join(parts) + "\n"
    if add_generation_prompt:
        s += "[assistant]\n"
    return s


def proc_tokenize(text: str, image: Image.Image | None):
    """
    Return input_ids 1D tensor for text (+ optional image).
    LLaVA processor typically expects images=image (not necessarily list).
    """
    if image is None:
        out = bigprocessor(text=text, return_tensors="pt")
    else:
        out = bigprocessor(text=text, images=image, return_tensors="pt")
    return out["input_ids"][0]


def build_loss_mask_from_prompt_and_full(prompt_str: str, full_str: str, image: Image.Image | None):
    """
    Build loss_mask by:
      - tokenize(prompt_str) => prompt_len
      - tokenize(full_str)   => full_ids
      - mask=0 for [0:prompt_len), mask=1 for [prompt_len:full_len)
    Then:
      - crop to max_len
      - set last token mask to 0 (avoid eos/end special token contributing to loss)
    """
    prompt_ids = proc_tokenize(prompt_str, image)
    full_ids = proc_tokenize(full_str, image)

    # truncate to max_len (consistent with teacher forward)
    full_ids = full_ids[: args.max_len]
    prompt_len = min(len(prompt_ids), len(full_ids))

    loss_mask = torch.zeros_like(full_ids)
    if prompt_len < len(full_ids):
        loss_mask[prompt_len:] = 1

    # avoid counting final special token (often eos)
    if len(loss_mask) > 0:
        loss_mask[-1] = 0

    return full_ids, loss_mask


@torch.no_grad()
def extract_teacher_states(image: Image.Image | None, input_ids_1d: torch.Tensor, loss_mask_1d: torch.Tensor):
    """
    Run teacher forward; return dict with:
      input_ids, inputs_embeds (hs[0]), hidden_state (last hs), loss_mask
    """
    input_ids = input_ids_1d.unsqueeze(0)  # [1,T]
    input_ids = input_ids[:, : args.max_len]
    attention_mask = torch.ones_like(input_ids)

    if image is None:
        outs = bigmodel(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            output_hidden_states=True,
        )
    else:
        # get vision inputs aligned with the SAME text
        proc = bigprocessor(text=bigprocessor.tokenizer.decode(input_ids_1d.tolist(), skip_special_tokens=False),
                            images=image, return_tensors="pt")
        # However, above decode->encode can drift for some tokenizers. Better: build proc from the same full_str externally.
        # So we instead compute pixel_values from (image, full_str) upstream when needed.
        # Here we assume we already rely on processor inside main for pixel_values (see below).
        raise RuntimeError("Internal error: multimodal path should pass pixel_values precomputed in main().")

    inputs_embeds = outs.hidden_states[0]      # [B,T,H]
    hidden_state = outs.hidden_states[-1]      # [B,T,H]

    td = {
        "input_ids": input_ids.cpu()[0],
        "inputs_embeds": inputs_embeds.cpu()[0],
        "hidden_state": hidden_state.cpu()[0],
        "loss_mask": loss_mask_1d[: input_ids.shape[1]].cpu(),
    }
    return td


@torch.no_grad()
def extract_teacher_states_mm(image: Image.Image, full_str: str, loss_mask_1d: torch.Tensor):
    """
    Multimodal teacher forward for LLaVA, using processor(text=..., images=...).
    """
    proc = bigprocessor(text=full_str, images=image, return_tensors="pt")
    input_ids = proc["input_ids"][:, : args.max_len]
    attention_mask = torch.ones_like(input_ids)

    # LLaVA in transformers uses pixel_values
    pixel_values = proc.get("pixel_values", None)
    if pixel_values is None:
        raise KeyError("processor output has no 'pixel_values' for multimodal input")

    outs = bigmodel(
        input_ids=input_ids.cuda(),
        pixel_values=pixel_values.cuda(),
        attention_mask=attention_mask.cuda(),
        output_hidden_states=True,
    )

    inputs_embeds = outs.hidden_states[0]     # [B,T,H]
    hidden_state = outs.hidden_states[-1]     # [B,T,H]

    td = {
        "input_ids": input_ids.cpu()[0],
        "inputs_embeds": inputs_embeds.cpu()[0],
        "hidden_state": hidden_state.cpu()[0],
        "loss_mask": loss_mask_1d[: input_ids.shape[1]].cpu(),
    }
    return td


def writedata(out_dir: str, idx: int, data_point: dict):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(data_point, os.path.join(out_dir, f"data_{idx}.ckpt"))


# -----------------------------
# Main
# -----------------------------
ds = load_dataset("parquet", data_files=args.parquet_path, split="train")
ds = ds.shuffle(seed=42)
ds = ds.select(range(args.start, min(args.end, len(ds))))

outdir = os.path.join(args.outdir, str(args.index))
os.makedirs(outdir, exist_ok=True)

written, skipped, failed_decode = 0, 0, 0
failed_nan = 0

for ex_i, ex in enumerate(tqdm(ds, desc="processing")):
    img = decode_image_from_parquet(ex)

    if img is None:
        failed_decode += 1
        if args.skip_no_image:
            skipped += 1
            continue
    
    if img is not None and written < 3:
        print(f"Successfully loaded image: {img.size} for index {ex_i}")

    messages = build_conversation_sqa(ex, include_solution=args.include_solution)

    # Insert image placeholder for chat template (same idea as your Qwen script)
    if img is not None:
        messages[0]["content"] = [{"type": "image"}] + messages[0]["content"]

    # full conversation string (with assistant answer)
    full_str = apply_template(messages, add_generation_prompt=False)

    # prompt-only string (up to assistant header)
    prompt_messages = [messages[0]]  # only user (with optional image placeholder)
    prompt_str = apply_template(prompt_messages, add_generation_prompt=True)

    # build input_ids + loss_mask (robust)
    input_ids_1d, loss_mask_1d = build_loss_mask_from_prompt_and_full(prompt_str, full_str, img)

    # teacher states
    if img is None:
        td = extract_teacher_states(None, input_ids_1d, loss_mask_1d)
    else:
        td = extract_teacher_states_mm(img, full_str, loss_mask_1d)

    # optional: skip NaN teacher states to keep dataset clean
    if torch.isnan(td["hidden_state"]).any() or torch.isinf(td["hidden_state"]).any():
        failed_nan += 1
        continue

    writedata(outdir, written, td)
    written += 1

print(f"Done: {outdir} | written={written} skipped={skipped} failed_decode={failed_decode} failed_nan={failed_nan}")