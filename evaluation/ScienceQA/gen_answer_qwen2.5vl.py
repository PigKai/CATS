# quick_speed_scienceqa_qwen25vl_refstyle.py
# -*- coding: utf-8 -*-

import argparse
import inspect
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, Tuple, Optional

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

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
            img = Image.open(BytesIO(image_field["bytes"]))
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


def apply_qwen_chat_template(processor, user_text: str, image_present: bool) -> str:
    """
    优先走 Qwen2.5-VL / HF processor 的 chat template；
    若 processor 没有，则回退到 tokenizer.apply_chat_template；
    再不行则 fallback 成普通字符串模板。
    """
    tokenizer = getattr(processor, "tokenizer", None)

    if hasattr(processor, "apply_chat_template"):
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
                    "content": [
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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

    if image_present:
        return f"USER: <image>\n{user_text}\nASSISTANT:"
    return f"USER:\n{user_text}\nASSISTANT:"


def build_inputs_same_template(
    processor,
    item,
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

    prompt = apply_qwen_chat_template(processor, user_text, image_present=image_present)

    if image_present:
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


def get_model_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_inputs_for_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue

        if k == "pixel_values":
            out[k] = v.to(device=device, dtype=torch.float16)
        else:
            out[k] = v.to(device=device)
    return out


def call_generate(fn, **kwargs):
    """
    兼容不同 eagenerate / naivegenerate 签名：
    - 如果函数支持 image_grid_thw，就传
    - 如果不支持，就自动过滤掉
    """
    sig = inspect.signature(fn)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if has_var_kw:
        return fn(**kwargs)

    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**valid_kwargs)


@torch.inference_mode()
def bench_one(fn_name: str, fn, inputs, max_new_tokens: int, temperature: float, device: torch.device):
    prompt_len = int(inputs["input_ids"].shape[1])

    moved = prepare_inputs_for_device(inputs, device=device)
    input_ids = moved["input_ids"]
    pixel_values = moved.get("pixel_values", None)
    image_grid_thw = moved.get("image_grid_thw", None)

    if device.type == "cuda":
        torch.cuda.synchronize()
    st = time.time()

    ret = call_generate(
        fn,
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        log=True,
    )
    if isinstance(ret, tuple) and len(ret) == 4:
        output_ids, _, idx, accept_lens = ret
    else:
        output_ids, _, idx = ret
        accept_lens = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - st

    gen_ids = output_ids[0][prompt_len:]
    new_tokens = int(gen_ids.shape[0])
    tps = new_tokens / max(1e-9, wall)
    acc = extract_accept_stats(accept_lens, new_tokens)

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
    parser = argparse.ArgumentParser("Quick speed test on ScienceQA with Qwen2.5-VL (naive vs eagle)")

    parser.add_argument("--base-model-path", type=str, default="/media/user/d/xyk/model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--ea-model-path", type=str, default="/media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/ScienceQA/gain/state_5_0.5")

    # ScienceQA parquet
    parser.add_argument("--parquet_path", type=str, default="/media/user/d/xyk/data1/ScienceQA/test-00000-of-00001-f0e719df791966ff.parquet")

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=500)

    parser.add_argument("--gpu_index", type=str, default="3")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--total_token", type=int, default=64)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--attn_impl", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--cache_dir", type=str, default="/media/user/d/hf_cache")
    parser.add_argument("--warmup", type=int, default=10, help="warmup steps (prefer image samples)")

    # prompt 开关：保持和原 ScienceQA 版本一致
    parser.add_argument("--use_lecture", action="store_true")
    parser.add_argument("--use_hint", action="store_true")
    parser.add_argument("--only_solution", action="store_true")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

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

    device = get_model_device(model)
    print(f"model device: {device}")

    print("warmup ...")
    warmup_samples = [ds[i] for i in range(len(ds)) if ds[i].get("image", None) is not None][:args.warmup]
    if len(warmup_samples) < args.warmup:
        warmup_samples = [ds[i] for i in range(min(args.warmup, len(ds)))]

    for sample in warmup_samples:
        inputs, _ = build_inputs_same_template(
            model.processor,
            sample,
            use_lecture=args.use_lecture,
            use_hint=args.use_hint,
            only_solution=args.only_solution,
        )
        moved = prepare_inputs_for_device(inputs, device=device)

        _ = call_generate(
            model.naivegenerate,
            input_ids=moved["input_ids"],
            pixel_values=moved.get("pixel_values", None),
            image_grid_thw=moved.get("image_grid_thw", None),
            max_new_tokens=2,
            temperature=args.temperature,
            log=False,
        )
        _ = call_generate(
            model.eagenerate,
            input_ids=moved["input_ids"],
            pixel_values=moved.get("pixel_values", None),
            image_grid_thw=moved.get("image_grid_thw", None),
            max_new_tokens=2,
            temperature=args.temperature,
            log=False,
        )

    print("Warmup done")

    total = {
        "naive_time": 0.0,
        "eagle_time": 0.0,
        "naive_tok": 0,
        "eagle_tok": 0,
        "accept_sum": 0.0,
        "accept_steps": 0.0,
    }

    for i in tqdm(range(len(ds)), desc="Processing"):
        item = ds[i]
        inputs, _ = build_inputs_same_template(
            model.processor,
            item,
            use_lecture=args.use_lecture,
            use_hint=args.use_hint,
            only_solution=args.only_solution,
        )

        naive = bench_one(
            "naivegenerate",
            model.naivegenerate,
            inputs,
            args.max_new_tokens,
            args.temperature,
            device,
        )
        eagle = bench_one(
            "eagenerate",
            model.eagenerate,
            inputs,
            args.max_new_tokens,
            args.temperature,
            device,
        )
        if eagle["avg_accept_len"] > 0 and eagle["steps"] > 0:
            total["accept_sum"] += eagle["avg_accept_len"] * eagle["steps"]
            total["accept_steps"] += eagle["steps"]

        total["naive_time"] += naive["time"]
        total["eagle_time"] += eagle["time"]
        total["naive_tok"] += naive["new_tokens"]
        total["eagle_tok"] += eagle["new_tokens"]

    naive_tps = total["naive_tok"] / max(1e-9, total["naive_time"])
    eagle_tps = total["eagle_tok"] / max(1e-9, total["eagle_time"])
    speedup_time = total["naive_time"] / max(1e-9, total["eagle_time"])
    speedup_tps = eagle_tps / max(1e-9, naive_tps)
    avg_accept_len = total["accept_sum"] / max(1e-9, total["accept_steps"])

    print("\n" + "=" * 60)
    print(f"Total samples: {len(ds)}")
    print(f"Average naive token/s: {naive_tps:.2f}")
    print(f"Average eagle token/s: {eagle_tps:.2f}")
    print(f"Speedup (time): {speedup_time:.3f}x")
    print(f"Speedup (token/s): {speedup_tps:.3f}x")
    print(f"Average accept length: {avg_accept_len:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()