# quick_speed_scienceqa_refstyle.py
# -*- coding: utf-8 -*-

import argparse
import os
import time
from typing import Any, Dict, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ea_model import EaModel


def _to_pil(image_field):
    if image_field is None:
        return None
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB") if image_field.mode != "RGB" else image_field
    if isinstance(image_field, dict):
        if "image" in image_field and isinstance(image_field["image"], Image.Image):
            img = image_field["image"]
            return img.convert("RGB") if img.mode != "RGB" else img
        if "bytes" in image_field and image_field["bytes"] is not None:
            import io
            img = Image.open(io.BytesIO(image_field["bytes"]))
            return img.convert("RGB") if img.mode != "RGB" else img
    return None


def build_reason_text(ex: Dict[str, Any], use_lecture: bool, use_hint: bool, only_solution: bool) -> str:
    sol = (ex.get("solution", "") or "").strip()
    lec = (ex.get("lecture", "") or "").strip()
    hint = (ex.get("hint", "") or "").strip()

    if only_solution:
        return sol

    parts = []
    if sol:
        parts.append(sol)
    if use_lecture and lec:
        parts.append(f"Lecture: {lec}")
    if use_hint and hint:
        parts.append(f"Hint: {hint}")

    return "\n".join(parts).strip()


def build_prompt_text(question: str, choices, image_present: bool, reason_ctx: str) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_lines = []
    for j, c in enumerate(choices):
        key = letters[j] if j < len(letters) else str(j)
        choice_lines.append(f"({key}) {c}")
    choices_block = "\n".join(choice_lines)

    head = (
        "You are given an image and a multiple-choice question."
        if image_present
        else "You are given a multiple-choice question."
    )

    ctx = f"\n\nContext:\n{reason_ctx}" if reason_ctx else ""

    prompt = (
        f"{head}\n"
        f"Choose the best answer from the choices.\n\n"
        f"Question: {question}\n"
        f"Choices:\n{choices_block}"
        f"{ctx}\n\n"
        f"Please follow this format:\n"
        f"Answer: (X) <choice>\n"
        f"Reasoning: <short explanation>"
    )
    return prompt


def apply_llava_chat_template(processor, user_text: str, image_present: bool) -> str:
    tokenizer = getattr(processor, "tokenizer", None)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        if image_present:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}],
                }
            ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # fallback
    if image_present:
        return f"USER: <image>\n{user_text}\nASSISTANT:"
    else:
        return f"USER:\n{user_text}\nASSISTANT:"


def build_inputs_same_template(
    processor,
    item,
    device: str,
    use_lecture: bool,
    use_hint: bool,
    only_solution: bool,
) -> Tuple[Dict[str, torch.Tensor], str]:
    image = _to_pil(item.get("image", None))
    image_present = image is not None

    question = item.get("question", "")
    choices = item.get("choices", [])

    reason_ctx = build_reason_text(
        item,
        use_lecture=use_lecture,
        use_hint=use_hint,
        only_solution=only_solution,
    )
    user_text = build_prompt_text(
        question=question,
        choices=choices,
        image_present=image_present,
        reason_ctx=reason_ctx,
    )
    prompt = apply_llava_chat_template(processor, user_text, image_present=image_present)

    if image is not None:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")

    return inputs, prompt


def extract_accept_stats(idx: Any, new_tokens: int) -> Dict[str, float]:
    out = {"avg_accept_len": -1.0, "accept_ratio": -1.0, "steps": -1.0}
    if idx is None:
        return out

    seq = None
    if isinstance(idx, (list, tuple)):
        try:
            seq = [int(x) for x in idx]
        except Exception:
            seq = None
    elif torch.is_tensor(idx):
        try:
            seq = [int(x) for x in idx.detach().cpu().flatten().tolist()]
        except Exception:
            seq = None
    elif isinstance(idx, dict):
        for k in ["accepted", "accept", "accept_len", "accept_lens", "idx"]:
            if k in idx and isinstance(idx[k], (list, tuple)):
                seq = [int(x) for x in idx[k]]
                break

    if not seq:
        return out

    s = sum(seq)
    out["steps"] = float(len(seq))
    out["avg_accept_len"] = float(s / max(1, len(seq)))
    out["accept_ratio"] = float(s / max(1, new_tokens))
    return out


@torch.inference_mode()
def bench_one(fn_name: str, fn, inputs, max_new_tokens: int, temperature: float):
    prompt_len = int(inputs["input_ids"].shape[1])

    torch.cuda.synchronize()
    st = time.time()

    input_ids = torch.as_tensor(inputs["input_ids"]).cuda()
    pixel_values = None
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        pixel_values = torch.as_tensor(inputs["pixel_values"]).cuda()

    output_ids, _, idx = fn(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        log=True,
    )

    torch.cuda.synchronize()
    wall = time.time() - st

    gen_ids = output_ids[0][prompt_len:]
    new_tokens = int(gen_ids.shape[0])
    tps = new_tokens / max(1e-9, wall)
    acc = extract_accept_stats(idx, new_tokens)

    return {
        "fn": fn_name,
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "time": wall,
        "tok_per_s": tps,
        "avg_accept_len": acc["avg_accept_len"],
        "accept_ratio": acc["accept_ratio"],
        "steps": acc["steps"],
    }


def main():
    parser = argparse.ArgumentParser("Quick speed test on ScienceQA (naive vs eagle)")

    parser.add_argument("--base-model-path", type=str, default="/media/user/d/xyk/model/llava-1.5-7b-hf")
    parser.add_argument("--ea-model-path", type=str, default="/media/user/d/xyk/model/Eagle_qwen3B_draft/distill/ScienceQA/Stage1/state_25")

    # ScienceQA parquet
    parser.add_argument("--parquet_path", type=str, default="/media/user/d/xyk/data1/ScienceQA/test-00000-of-00001-f0e719df791966ff.parquet")

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=500)

    parser.add_argument("--gpu_index", type=str, default="0")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--total_token", type=int, default=64)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--attn_impl", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--cache_dir", type=str, default="/media/user/d/hf_cache")
    parser.add_argument("--warmup", type=int, default=10, help="warmup steps (prefer image samples)")

    # prompt 相关开关：完全对齐你给的 ScienceQA 版本
    parser.add_argument("--use_lecture", action="store_true")
    parser.add_argument("--use_hint", action="store_true")
    parser.add_argument("--only_solution", action="store_true")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_dataset(
        "parquet",
        data_files={"train": args.parquet_path},
        split="train",
        cache_dir=args.cache_dir,
    )
    ds = ds.select(range(args.start, min(args.start + args.n, len(ds))))

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=args.attn_impl,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
    ).eval()

    print("warmup ...")
    warmup_samples = [ds[i] for i in range(len(ds)) if ds[i].get("image", None) is not None][:args.warmup]
    if len(warmup_samples) < args.warmup:
        warmup_samples = [ds[i] for i in range(min(args.warmup, len(ds)))]

    for sample in warmup_samples:
        inputs, _ = build_inputs_same_template(
            model.processor,
            sample,
            device=device,
            use_lecture=args.use_lecture,
            use_hint=args.use_hint,
            only_solution=args.only_solution,
        )
        _ = model.naivegenerate(
            input_ids=torch.as_tensor(inputs["input_ids"]).cuda(),
            pixel_values=torch.as_tensor(inputs["pixel_values"]).cuda() if "pixel_values" in inputs else None,
            max_new_tokens=2,
            temperature=args.temperature,
            log=False,
        )
        _ = model.eagenerate(
            input_ids=torch.as_tensor(inputs["input_ids"]).cuda(),
            pixel_values=torch.as_tensor(inputs["pixel_values"]).cuda() if "pixel_values" in inputs else None,
            max_new_tokens=2,
            temperature=args.temperature,
            log=False,
        )

    print("Warmup done")

    total = {"naive_time": 0.0, "eagle_time": 0.0, "naive_tok": 0, "eagle_tok": 0}

    for i in tqdm(range(len(ds)), desc="Processing"):
        item = ds[i]
        inputs, _ = build_inputs_same_template(
            model.processor,
            item,
            device=device,
            use_lecture=args.use_lecture,
            use_hint=args.use_hint,
            only_solution=args.only_solution,
        )

        naive = bench_one("naivegenerate", model.naivegenerate, inputs, args.max_new_tokens, args.temperature)
        eagle = bench_one("eagenerate", model.eagenerate, inputs, args.max_new_tokens, args.temperature)

        total["naive_time"] += naive["time"]
        total["eagle_time"] += eagle["time"]
        total["naive_tok"] += naive["new_tokens"]
        total["eagle_tok"] += eagle["new_tokens"]

    naive_tps = total["naive_tok"] / max(1e-9, total["naive_time"])
    eagle_tps = total["eagle_tok"] / max(1e-9, total["eagle_time"])
    speedup_time = total["naive_time"] / max(1e-9, total["eagle_time"])
    speedup_tps = eagle_tps / max(1e-9, naive_tps)

    print("\n" + "=" * 60)
    print(f"Total samples: {len(ds)}")
    print(f"Average naive token/s: {naive_tps:.2f}")
    print(f"Average eagle token/s: {eagle_tps:.2f}")
    print(f"Speedup (time): {speedup_time:.3f}x")
    print(f"Speedup (token/s): {speedup_tps:.3f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()