# distill_textvqa_qwen2vl.py
# -*- coding: utf-8 -*-

import argparse
import os
import random
import gc
import re
from io import BytesIO
from typing import Optional, Tuple, Dict, Any, List

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from PIL import Image

# 兼容不同 transformers 版本
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLForConditionalGeneration
except Exception:
    try:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLForConditionalGeneration
    except Exception:
        raise ImportError(
            "Cannot import Qwen2_5_VLForConditionalGeneration or Qwen2VLForConditionalGeneration. "
            "Please upgrade transformers to a version that supports Qwen2-VL / Qwen2.5-VL."
        )


def clean_hard_text(text: str) -> str:
    text = text.strip()

    bad_prefixes = [
        "自动生成的文本内容如下：",
        "以下是自动生成的文本内容：",
        "答案如下：",
        "自动生成的详细解释如下：",
    ]
    for p in bad_prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    m = re.match(r'^addCriterion\("image",\s*"(.*)"\)\s*$', text, flags=re.S)
    if m:
        text = m.group(1).strip()

    text = text.strip('"').strip()
    return text


# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=50000)   # -1 表示到最后
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])

parser.add_argument("--outdir", type=str, default="/media/user/e/xyk/data/TextVQA/qwen2.5vl_3B_distill")
parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
parser.add_argument("--model_path", type=str, default="/media/user/d/xyk/model/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--dataset_script", type=str, default="/media/user/e/xyk/eagle-eye2/eagle-eye2/textvqa.py")

# 生成硬标签参数
parser.add_argument("--gen_max_new_tokens", type=int, default=256)
parser.add_argument("--gen_num_beams", type=int, default=1)

# 稳健参数
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--save_conversation", action="store_true", help="额外把 conversation 文本存进 ckpt 便于排查")
parser.add_argument("--debug_samples", type=int, default=3, help="打印前几个样本做验证")
parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.gpu_index)

random.seed(args.seed)
torch.manual_seed(args.seed)

if args.torch_dtype == "float16":
    MODEL_DTYPE = torch.float16
elif args.torch_dtype == "bfloat16":
    MODEL_DTYPE = torch.bfloat16
else:
    MODEL_DTYPE = torch.float32


# -----------------------------
# Load model / processor
# -----------------------------
bigmodel = QwenVLForConditionalGeneration.from_pretrained(
    args.model_path,
    device_map="auto",
    torch_dtype=MODEL_DTYPE,
    attn_implementation="eager",
)

if "Qwen2.5-VL" in args.model_path:
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    bigprocessor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
else:
    bigprocessor = AutoProcessor.from_pretrained(args.model_path)

tokenizer = getattr(bigprocessor, "tokenizer", None)
if tokenizer is None:
    raise RuntimeError("AutoProcessor does not expose tokenizer. Please check your transformers version.")


def get_model_device() -> torch.device:
    for p in bigmodel.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MODEL_DEVICE = get_model_device()


def to_pil_image(img_field):
    if img_field is None:
        return None
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, dict):
        if img_field.get("image") is not None and isinstance(img_field["image"], Image.Image):
            return img_field["image"].convert("RGB")
        if img_field.get("bytes"):
            return Image.open(BytesIO(img_field["bytes"])).convert("RGB")
    return None


def build_messages(question: str, image_present: bool) -> List[Dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                },
            ],
        }
    ]

    user_content = [
        {
            "type": "text",
            "text": question,
        },
        {
            "type": "text",
            "text": "Perform an OCR task on the provided image. Please extract the text accurately and provide a detailed explanation of the process. Ensure the response is comprehensive and well-structured.",
        },
    ]

    if image_present:
        user_content.append({"type": "image"})

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )
    return messages


def build_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    if hasattr(bigprocessor, "apply_chat_template"):
        return bigprocessor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # fallback
    text_parts = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            for x in msg["content"]:
                if x["type"] == "text":
                    text_parts.append(f"<|im_start|>system\n{x['text']}<|im_end|>\n")
        elif role == "user":
            text_parts.append("<|im_start|>user\n")
            for x in msg["content"]:
                if x["type"] == "text":
                    text_parts.append(x["text"])
                    text_parts.append("\n")
                elif x["type"] == "image":
                    text_parts.append("<|vision_start|><|image_pad|><|vision_end|>\n")
            text_parts.append("<|im_end|>\n")
    text_parts.append("<|im_start|>assistant\n")
    return "".join(text_parts)


def move_inputs_to_model_device(inputs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if k == "pixel_values":
                out[k] = v.to(MODEL_DEVICE, dtype=MODEL_DTYPE)
            else:
                out[k] = v.to(MODEL_DEVICE)
        else:
            out[k] = v
    return out


def build_processor_inputs_from_messages(
    messages: List[Dict[str, Any]],
    image: Optional[Image.Image]
) -> Tuple[str, Dict[str, Any]]:
    prompt_input = build_prompt_from_messages(messages)

    if image is not None:
        images = [image]
        inputs = bigprocessor(
            images=images,
            text=prompt_input,
            return_tensors="pt",
        )
    else:
        inputs = bigprocessor(
            text=prompt_input,
            return_tensors="pt",
        )

    return prompt_input, inputs


def decode_text(token_ids: torch.Tensor) -> str:
    if token_ids.ndim > 1:
        token_ids = token_ids[0]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_hard_label(messages: List[Dict[str, Any]], image: Optional[Image.Image]) -> Tuple[str, str, Dict[str, Any]]:
    prompt_str, inputs = build_processor_inputs_from_messages(messages, image)
    inputs = move_inputs_to_model_device(inputs)

    prompt_len = inputs["input_ids"].shape[1]

    gen_ids = bigmodel.generate(
        **inputs,
        max_new_tokens=args.gen_max_new_tokens,
        do_sample=False,
        num_beams=args.gen_num_beams,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = gen_ids[0][prompt_len:]
    hard_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    hard_text = clean_hard_text(hard_text)
    return hard_text, prompt_str, inputs


@torch.no_grad()
def build_one_item(ex: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    image = to_pil_image(ex.get("image"))
    image_present = image is not None

    question = ex["question"]
    gt_answers = ex.get("answers", [])

    messages = build_messages(question, image_present=image_present)

    hard_label, prompt_str, _ = generate_hard_label(messages=messages, image=image)

    eos_text = tokenizer.eos_token if tokenizer.eos_token is not None else ""
    full_text = f"{prompt_str}{hard_label}{eos_text}"

    # 重新构造 full_text 与 prompt_only 的 processor 输入
    if image is not None:
        full_inputs = bigprocessor(
            images=[image],
            text=full_text,
            return_tensors="pt",
        )
        inst_inputs = bigprocessor(
            images=[image],
            text=prompt_str,
            return_tensors="pt",
        )
    else:
        full_inputs = bigprocessor(
            text=full_text,
            return_tensors="pt",
        )
        inst_inputs = bigprocessor(
            text=prompt_str,
            return_tensors="pt",
        )

    input_ids = full_inputs["input_ids"][0]
    inst_len = inst_inputs["input_ids"].shape[1]

    loss_mask = torch.zeros_like(input_ids)
    if input_ids.shape[0] > inst_len:
        loss_mask[inst_len:] = 1

    pixel_values = full_inputs.get("pixel_values", None)
    image_grid_thw = full_inputs.get("image_grid_thw", None)

    one_item = {
        "input_ids": input_ids.tolist(),
        "loss_mask": loss_mask.tolist(),
        "pixel_values": pixel_values.cpu() if pixel_values is not None else None,
        "image_grid_thw": image_grid_thw.cpu() if image_grid_thw is not None else None,
    }

    masked_text = tokenizer.decode(input_ids[inst_len:], skip_special_tokens=False)

    convo = (
        "=== FULL CONVERSATION ===\n"
        f"{full_text}\n\n"
        "=== MASKED (TRAINED) TEXT ===\n"
        f"{masked_text}\n\n"
        f"mask tokens: {int(loss_mask.sum().item())} / {input_ids.shape[0]}\n"
    )

    debug_info = {
        "question": question,
        "gt_answers": gt_answers,
        "prompt_str": prompt_str,
        "hard_label": hard_label,
        "full_text": full_text,
        "masked_text": masked_text,
        "input_len": int(input_ids.shape[0]),
        "inst_len": int(inst_len),
        "mask_sum": int(loss_mask.sum().item()),
        "pixel_values_shape": tuple(pixel_values.shape) if pixel_values is not None else None,
        "image_grid_thw_shape": tuple(image_grid_thw.shape) if image_grid_thw is not None else None,
    }
    return one_item, convo, debug_info


@torch.no_grad()
def process_and_save(one_item: Dict[str, Any], outdir: str, idx: int, conversation: Optional[str] = None):
    input_ids = torch.as_tensor(one_item["input_ids"]).unsqueeze(0).to(MODEL_DEVICE)
    loss_mask = torch.as_tensor(one_item["loss_mask"]).unsqueeze(0)

    pixel_values = one_item["pixel_values"]
    image_grid_thw = one_item["image_grid_thw"]

    model_inputs = {
        "input_ids": input_ids,
        "output_hidden_states": True,
    }

    if pixel_values is not None:
        pixel_values = pixel_values.to(MODEL_DEVICE, dtype=MODEL_DTYPE)
        model_inputs["pixel_values"] = pixel_values

    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(MODEL_DEVICE)
        model_inputs["image_grid_thw"] = image_grid_thw

    outs = bigmodel(**model_inputs)

    td = {
        "input_ids": input_ids.detach().cpu()[0],
        "inputs_embeds": outs.hidden_states[0].detach().cpu()[0],
        "hidden_state": outs.hidden_states[-1].detach().cpu()[0],
        "loss_mask": loss_mask.detach().cpu()[0],
        "pixel_values": pixel_values.detach().cpu() if pixel_values is not None else None,
        "image_grid_thw": image_grid_thw.detach().cpu() if image_grid_thw is not None else None,
    }

    if args.save_conversation and conversation is not None:
        td["conversation"] = conversation

    os.makedirs(outdir, exist_ok=True)
    save_path = os.path.join(outdir, f"data_{idx}.ckpt")
    torch.save(td, save_path)
    return save_path


def print_debug_sample(local_i: int, global_i: int, debug_info: Dict[str, Any]):
    print("\n" + "=" * 100)
    print(f"[DEBUG SAMPLE] local_i={local_i} | global_i={global_i}")
    print("-" * 100)
    print(f"[Question]\n{debug_info['question']}")
    print("-" * 100)
    print(f"[Gold answers]\n{debug_info['gt_answers']}")
    print("-" * 100)
    print(f"[Prompt]\n{debug_info['prompt_str']}")
    print("-" * 100)
    print(f"[Generated hard label]\n{debug_info['hard_label']}")
    print("-" * 100)
    print(f"[Masked trained text]\n{debug_info['masked_text']}")
    print("-" * 100)
    print(
        f"[Shapes] input_len={debug_info['input_len']} | "
        f"inst_len={debug_info['inst_len']} | "
        f"mask_sum={debug_info['mask_sum']} | "
        f"pixel_values_shape={debug_info['pixel_values_shape']} | "
        f"image_grid_thw_shape={debug_info['image_grid_thw_shape']}"
    )
    print("=" * 100 + "\n")


def inspect_saved_ckpt(path: str):
    print("\n" + "#" * 100)
    print(f"[CKPT VERIFY] {path}")
    td = torch.load(path, map_location="cpu")
    print(f"keys: {list(td.keys())}")
    for k, v in td.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        elif v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: type={type(v)}")
    if "input_ids" in td and "loss_mask" in td:
        input_ids = td["input_ids"]
        loss_mask = td["loss_mask"]
        active_pos = (loss_mask == 1).nonzero(as_tuple=False).flatten()
        if len(active_pos) > 0:
            start = int(active_pos[0].item())
            print("-" * 100)
            print("[Decoded trained tail]")
            print(tokenizer.decode(input_ids[start:], skip_special_tokens=False))
    print("#" * 100 + "\n")


def main():
    dataset = load_dataset(args.dataset_script, split=args.split)
    n = len(dataset)

    start = max(0, min(args.start, n))
    end = n if args.end is None or args.end < 0 else max(0, min(args.end, n))
    if end < start:
        end = start

    dataset = dataset.select(range(start, end))
    assert len(dataset) > 0, f"Empty dataset after select: start={start}, end={end}, n={n}"

    outdir = os.path.join(args.outdir, str(args.index))
    os.makedirs(outdir, exist_ok=True)

    first_save_path = None
    oom_count = 0

    for local_i in tqdm(range(len(dataset)), desc="Saving ckpt", total=len(dataset)):
        global_i = start + local_i
        save_path = os.path.join(outdir, f"data_{global_i}.ckpt")

        if os.path.exists(save_path):
            print(f"[Skip] already exists: {save_path}")
            continue

        ex = dataset[local_i]

        try:
            one_item, convo, debug_info = build_one_item(ex)

            if local_i < args.debug_samples:
                print_debug_sample(local_i, global_i, debug_info)

            save_path = process_and_save(one_item, outdir, idx=global_i, conversation=convo)

            if first_save_path is None:
                first_save_path = save_path

        except torch.cuda.OutOfMemoryError as e:
            oom_count += 1
            print(f"\n[OOM-SKIP] global_i={global_i}")
            print(f"[OOM-SKIP] question={ex.get('question', '')}")
            print(f"[OOM-SKIP] error={str(e)}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_count += 1
                print(f"\n[OOM-SKIP] global_i={global_i}")
                print(f"[OOM-SKIP] question={ex.get('question', '')}")
                print(f"[OOM-SKIP] error={str(e)}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if first_save_path is not None:
        inspect_saved_ckpt(first_save_path)

    print(f"Finished: {outdir}")
    print(f"Total OOM skipped samples: {oom_count}")


if __name__ == "__main__":
    main()