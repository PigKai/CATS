import argparse
import os
import torch
from io import BytesIO
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from datasets import load_dataset, DownloadConfig


# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=50000)          # -1 表示到最后
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])

parser.add_argument("--outdir", type=str, default="/media/user/d/xyk/data1/textVQA")
parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="TextVQA split to use")
parser.add_argument("--model_path", type=str, default="/media/user/d/xyk/model/llava-1.5-7b-hf")

# 生成硬标签参数
parser.add_argument("--gen_max_new_tokens", type=int, default=256)   
parser.add_argument("--gen_num_beams", type=int, default=1)

# 一些稳健参数
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--save_conversation", action="store_true", help="额外把 conversation 文本存进 ckpt 便于排查")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(args.seed)

# -----------------------------
# Load model / processor
# -----------------------------
bigmodel = LlavaForConditionalGeneration.from_pretrained(
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


def build_prompt_text(question: str) -> str:
    # 保持与你提供的 prompt 逻辑完全一致
    instruction = (
        "Perform an OCR task on the provided image. Please extract the text accurately "
        "and provide a detailed explanation of the process. Ensure the response is "
        "comprehensive and well-structured."
    )
    # 合并问题和 OCR 指令
    return f"{question}\n{instruction}"

def apply_llava_chat_template(user_text: str, image_present: bool) -> str:
    # 1. 定义 System Prompt (与你的参考代码一致)
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

    # 2. 定义 User Content
    user_content = [
        {"type": "text", "text": user_text},
    ]
    
    # 如果有图片，按照你给出代码的顺序，把图片放在最后 ({"type": "image"})
    if image_present:
        user_content.append({"type": "image"})
        
    messages.append({
        "role": "user",
        "content": user_content
    })

    # 3. 使用 tokenizer 的模板 (LLaVA-1.5 通常支持 apply_chat_template)
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    # Fallback: 如果 tokenizer 没有模板，手动构建 LLaVA-1.5 风格的字符串
    # 注意：LLaVA-1.5 官方原始权重不一定带有 system 模板，通常为 USER: <image>\n{text} ASSISTANT:
    system_str = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    if image_present:
        return f"{system_str} USER: <image>\n{user_text} ASSISTANT:"
    else:
        return f"{system_str} USER: {user_text} ASSISTANT:"


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
        inputs = bigprocessor(text=prompt_str, images=image, return_tensors="pt")
    else:
        inputs = bigprocessor(text=prompt_str, return_tensors="pt")

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

    image = to_pil_image(ex.get("image"))
    image_present = image is not None

    question = ex["question"]   # TextVQA 中的问题字段

    user_text = build_prompt_text(question)

    # 1) prompt（用于 generate & 计算 inst_len），不含 eos
    prompt_str = apply_llava_chat_template(user_text, image_present=image_present)

    # 2) 生成硬标签（assistant 文本）
    hard_label = generate_hard_label(prompt_str=prompt_str, image=image)

    # 3) full_text = prompt + hard_label + eos（只在末尾加一次 eos）
    full_text = f"{prompt_str}{hard_label}{tokenizer.eos_token}"

    # 4) 编码 full_text 与 prompt_only 来构建 mask
    if image_present:
        full_inputs = bigprocessor(text=full_text, images=image, return_tensors="pt")
        inst_inputs = bigprocessor(text=prompt_str, images=image, return_tensors="pt")
        pixel_values = full_inputs["pixel_values"][0].to(torch.float16)  # tensor
    else:
        full_inputs = bigprocessor(text=full_text, return_tensors="pt")
        inst_inputs = bigprocessor(text=prompt_str, return_tensors="pt")
        pixel_values = None

    input_ids = full_inputs["input_ids"][0]
    inst_len = inst_inputs["input_ids"].shape[1]

    loss_mask = torch.zeros_like(input_ids)
    if input_ids.shape[0] > inst_len:
        loss_mask[inst_len:] = 1

    one_item = {
        "input_ids": input_ids.tolist(),
        "loss_mask": loss_mask.tolist(),
        "pixel_values": pixel_values.cpu().tolist() if pixel_values is not None else None,
    }

    # conversation/debug
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
    loss_mask = torch.as_tensor(one_item["loss_mask"]).unsqueeze(0)  # mask 存 cpu 也行
    pixel_values = one_item["pixel_values"]
    if pixel_values is not None:
        pixel_values = torch.as_tensor(pixel_values).unsqueeze(0).to(MODEL_DEVICE, dtype=torch.float16)

    outs = bigmodel(input_ids=input_ids, pixel_values=pixel_values, output_hidden_states=True)

    td = {
        "input_ids": input_ids.detach().cpu()[0],
        "inputs_embeds": outs.hidden_states[0].detach().cpu()[0],
        "hidden_state": outs.hidden_states[-1].detach().cpu()[0],
        "loss_mask": loss_mask.detach().cpu()[0],
        "pixel_values": pixel_values.detach().cpu()[0] if pixel_values is not None else None,
    }
    if args.save_conversation and conversation is not None:
        td["conversation"] = conversation

    os.makedirs(outdir, exist_ok=True)
    torch.save(td, os.path.join(outdir, f"data_{idx}.ckpt"))


def main():

    dataset = load_dataset("/media/user/e/xyk/eagle-eye2/eagle-eye2/textvqa.py", trust_remote_code=True, split="train")
    n = len(dataset)

    start = max(0, min(args.start, n))
    end = n if args.end is None or args.end < 0 else max(0, min(args.end, n))
    if end < start:
        end = start

    dataset = dataset.select(range(start, end))
    assert len(dataset) > 0, f"Empty dataset after select: start={start}, end={end}, n={n}"

    outdir = os.path.join(args.outdir, str(args.index))
    os.makedirs(outdir, exist_ok=True)

    for local_i in tqdm(range(len(dataset)), desc="Saving ckpt", total=len(dataset)):
        ex = dataset[local_i]
        one_item, convo = build_one_item(ex)
        if local_i == 0:
            print(convo)  # 打印第一条对话用于检查
        process_and_save(one_item, outdir, idx=start + local_i, conversation=convo)

    print(f"Finished: {outdir}")


if __name__ == "__main__":
    main()