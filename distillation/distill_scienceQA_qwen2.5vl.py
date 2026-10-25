import argparse
import os
import torch
from io import BytesIO
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=20000)          # -1 表示到最后
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])

parser.add_argument("--outdir", type=str, default="/media/user/e/xyk/data/scienceQA/qwen2.5vl_3B_distill")
parser.add_argument("--parquet_path", type=str, default="/media/user/d/xyk/data1/ScienceQA/train-00000-of-00001-1028f23e353fbe3e.parquet")
parser.add_argument("--model_path", type=str, default="/media/user/d/xyk/model/Qwen2.5-VL-3B-Instruct")

# 生成硬标签参数
parser.add_argument("--gen_max_new_tokens", type=int, default=256)
parser.add_argument("--gen_num_beams", type=int, default=1)

# 数据字段控制（保留接口，下面按你要的“推理模板风格”生成）
parser.add_argument("--use_lecture", action="store_true")
parser.add_argument("--use_hint", action="store_true")
parser.add_argument("--only_solution", action="store_true")

# 一些稳健参数
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--save_conversation", action="store_true", help="额外把 conversation 文本存进 ckpt 便于排查")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.manual_seed(args.seed)

# -----------------------------
# Load model / processor
# -----------------------------
bigmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
bigprocessor = AutoProcessor.from_pretrained(args.model_path)
tokenizer = bigprocessor.tokenizer

# 兼容 device_map="auto"：取一个“实际权重所在”的 device
def get_model_device() -> torch.device:
    for p in bigmodel.parameters():
        if p.device.type != "meta":
            return p.device
    # 极端情况下全是 meta（基本不会），退化到 cuda:0
    return torch.device("cuda:0")

MODEL_DEVICE = get_model_device()


def to_pil_image(img_field):
    if img_field is None:
        return None
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, dict) and img_field.get("bytes"):
        return Image.open(BytesIO(img_field["bytes"])).convert("RGB")
    return None


def build_reason_text(ex: Dict[str, Any]) -> str:
    """按你原来逻辑保留 solution/lecture/hint 的拼接开关，但改成更像“官方推理模板”的内容组织。"""
    sol = (ex.get("solution", "") or "").strip()
    lec = (ex.get("lecture", "") or "").strip()
    hint = (ex.get("hint", "") or "").strip()

    if args.only_solution:
        return sol

    parts = []
    if sol:
        parts.append(sol)
    if args.use_lecture and lec:
        parts.append(f"Lecture: {lec}")
    if args.use_hint and hint:
        parts.append(f"Hint: {hint}")

    return "\n".join(parts).strip()


def build_prompt_text(question: str, choices, image_present: bool, reason_ctx: str) -> str:
    """
    采用更“稳”的 ScienceQA 推理模板风格：
    - 有/无图分别描述
    - 强制输出 Answer + Reasoning 两段
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_lines = []
    for j, c in enumerate(choices):
        key = letters[j] if j < len(letters) else str(j)
        choice_lines.append(f"({key}) {c}")
    choices_block = "\n".join(choice_lines)

    head = "You are given an image and a multiple-choice question." if image_present else \
           "You are given a multiple-choice question."

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


def apply_qwen_chat_template(user_text: str, image_present: bool) -> str:
    """
    尽量用 Qwen2.5-VL/HF 的 chat template：
    - 如果 tokenizer/processor 支持 apply_chat_template，就走它
    - 否则 fallback
    """
    if hasattr(bigprocessor, "apply_chat_template"):
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

        prompt = bigprocessor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    if hasattr(tokenizer, "apply_chat_template"):
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

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    if image_present:
        return f"USER: <image>\n{user_text}\nASSISTANT:"
    else:
        return f"USER:\n{user_text}\nASSISTANT:"


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if k == "pixel_values":
                out[k] = v.to(MODEL_DEVICE, dtype=torch.float16)
            else:
                out[k] = v.to(MODEL_DEVICE)
        else:
            out[k] = v
    return out


@torch.no_grad()
def generate_hard_label(prompt_str: str, image: Optional[Image.Image]) -> str:
    if image is not None:
        inputs = bigprocessor(
            text=[prompt_str],
            images=[image],
            return_tensors="pt",
            padding=True
        )
    else:
        inputs = bigprocessor(
            text=[prompt_str],
            return_tensors="pt",
            padding=True
        )

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
    return hard_text


@torch.no_grad()
def build_one_item(ex: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    image = to_pil_image(ex.get("image", None))
    image_present = image is not None

    question = ex["question"]
    choices = ex["choices"]
    reason_ctx = build_reason_text(ex)

    user_text = build_prompt_text(question, choices, image_present=image_present, reason_ctx=reason_ctx)

    # 1) prompt（用于 generate & 计算 inst_len），不含 eos
    prompt_str = apply_qwen_chat_template(user_text, image_present=image_present)

    # 2) 生成硬标签（assistant 文本）
    hard_label = generate_hard_label(prompt_str=prompt_str, image=image)

    # 3) full_text = prompt + hard_label + eos
    full_text = f"{prompt_str}{hard_label}{tokenizer.eos_token}"

    # 4) 编码 full_text 与 prompt_only 来构建 mask
    if image_present:
        full_inputs = bigprocessor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        inst_inputs = bigprocessor(
            text=[prompt_str],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        pixel_values = full_inputs["pixel_values"].to(torch.float16)

        # 保持 processor 原始返回形状，不要手动切
        image_grid_thw = full_inputs.get("image_grid_thw", None)

    else:
        full_inputs = bigprocessor(
            text=[full_text],
            return_tensors="pt",
            padding=True
        )
        inst_inputs = bigprocessor(
            text=[prompt_str],
            return_tensors="pt",
            padding=True
        )
        pixel_values = None
        image_grid_thw = None

    input_ids = full_inputs["input_ids"][0]
    inst_len = inst_inputs["input_ids"].shape[1]

    loss_mask = torch.zeros_like(input_ids)
    if input_ids.shape[0] > inst_len:
        loss_mask[inst_len:] = 1

    one_item = {
        "input_ids": input_ids.tolist(),
        "loss_mask": loss_mask.tolist(),
        "pixel_values": pixel_values.cpu().tolist() if pixel_values is not None else None,
        "image_grid_thw": image_grid_thw.cpu().tolist() if image_grid_thw is not None else None,
    }

    convo = (
        "=== FULL CONVERSATION ===\n"
        f"{full_text}\n\n"
        "=== MASKED (TRAINED) TEXT ===\n"
        f"{tokenizer.decode(input_ids[inst_len:], skip_special_tokens=False)}\n\n"
        f"mask tokens: {int(loss_mask.sum().item())} / {input_ids.shape[0]}\n"
    )
    return one_item, convo


@torch.no_grad()
def process_and_save(one_item: Dict[str, Any], outdir: str, idx: int, conversation: Optional[str] = None):
    input_ids = torch.as_tensor(one_item["input_ids"]).unsqueeze(0).to(MODEL_DEVICE)
    loss_mask = torch.as_tensor(one_item["loss_mask"]).unsqueeze(0)

    pixel_values = one_item["pixel_values"]
    if pixel_values is not None:
        pixel_values = torch.as_tensor(pixel_values, dtype=torch.float16).to(MODEL_DEVICE)

    image_grid_thw = one_item.get("image_grid_thw", None)
    if image_grid_thw is not None:
        image_grid_thw = torch.as_tensor(image_grid_thw).to(MODEL_DEVICE)

    if pixel_values is not None:
        outs = bigmodel(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True
        )
    else:
        outs = bigmodel(
            input_ids=input_ids,
            output_hidden_states=True
        )

    td = {
        "input_ids": input_ids.detach().cpu()[0],
        "inputs_embeds": outs.hidden_states[0].detach().cpu()[0],
        "hidden_state": outs.hidden_states[-1].detach().cpu()[0],
        "loss_mask": loss_mask.detach().cpu()[0],
    }
    if pixel_values is not None:
        td["pixel_values"] = pixel_values.detach().cpu()

    if image_grid_thw is not None:
        td["image_grid_thw"] = image_grid_thw.detach().cpu()

    if args.save_conversation and conversation is not None:
        td["conversation"] = conversation

    os.makedirs(outdir, exist_ok=True)
    torch.save(td, os.path.join(outdir, f"data_{idx}.ckpt"))

def main():
    ds = load_dataset("parquet", data_files={"train": args.parquet_path}, split="train")
    n = len(ds)

    start = max(0, min(args.start, n))
    end = n if args.end is None or args.end < 0 else max(0, min(args.end, n))
    if end < start:
        end = start

    ds = ds.select(range(start, end))
    assert len(ds) > 0, f"Empty dataset after select: start={start}, end={end}, n={n}"

    outdir = os.path.join(args.outdir, str(args.index))
    os.makedirs(outdir, exist_ok=True)

    for local_i in tqdm(range(len(ds)), desc="Saving ckpt", total=len(ds)):
        ex = ds[local_i]
        one_item, convo = build_one_item(ex)
        if local_i == 0:
            print(convo)
        process_and_save(one_item, outdir, idx=start + local_i, conversation=convo)

    print(f"Finished: {outdir}")


if __name__ == "__main__":
    main()