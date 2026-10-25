import os
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ------------------------------------------------
# Args
# ------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--src_dir",
    type=str,
    default="/media/user/d/xyk/data1/llava_instruct/qwen2.5vl/qwen2.5vl_7B_0_67999/0",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="/media/user/d/xyk/model/Qwen2.5-VL-7B-Instruct",
)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1, help="-1 means all")
parser.add_argument("--max_samples", type=int, default=-1, help="-1 means process all")
parser.add_argument(
    "--topk_ratio",
    type=float,
    default=0.1,
    help="top-k ratio in answer region for visual_static_mask",
)
parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--overwrite", action="store_true", help="真的原地覆盖保存时再打开")
parser.add_argument("--debug_first_n", type=int, default=3, help="只打印前几条样本的调试信息")
parser.add_argument(
    "--trust_remote_code",
    action="store_true",
    help="如本地模型需要 remote code，可打开",
)
args = parser.parse_args()


# ------------------------------------------------
# Dtype
# ------------------------------------------------
if args.dtype == "fp16":
    model_dtype = torch.float16
elif args.dtype == "bf16":
    model_dtype = torch.bfloat16
else:
    model_dtype = torch.float32


# ------------------------------------------------
# Load model / processor
# ------------------------------------------------
processor = AutoProcessor.from_pretrained(
    args.model_path,
    trust_remote_code=args.trust_remote_code,
)
tokenizer = processor.tokenizer

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=model_dtype,
    device_map=None,
    attn_implementation="eager",
    trust_remote_code=args.trust_remote_code,
)
model.eval().to(args.device)


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def list_ckpt_files(folder: str) -> List[str]:
    files = []
    pat = re.compile(r"^data_(\d+)\.ckpt$")
    for fn in os.listdir(folder):
        if pat.match(fn):
            files.append(os.path.join(folder, fn))
    files.sort(key=lambda x: int(re.search(r"data_(\d+)\.ckpt", os.path.basename(x)).group(1)))
    return files


def find_image_token_ids(tok) -> List[int]:
    """
    尝试找到 Qwen2.5-VL 的图像相关 token id
    这里保留多候选写法，尽量兼容不同 tokenizer 配置。
    """
    candidates = [
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<image>",
        "<img>",
        "<|img|>",
    ]
    ids = []

    for s in candidates:
        try:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid != tok.unk_token_id:
                ids.append(tid)
        except Exception:
            pass

        try:
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1 and enc[0] != tok.unk_token_id:
                ids.append(enc[0])
        except Exception:
            pass

    uniq = []
    for x in ids:
        if x not in uniq:
            uniq.append(x)
    return uniq


IMAGE_TOKEN_IDS = find_image_token_ids(tokenizer)
print("Detected image token ids:", IMAGE_TOKEN_IDS)


def load_ckpt(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    return data


def get_answer_positions(loss_mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(loss_mask == 1, as_tuple=True)[0]


def count_image_tokens(input_ids: torch.Tensor, image_token_ids: List[int]) -> int:
    cnt = 0
    for tid in image_token_ids:
        cnt += int((input_ids == tid).sum().item())
    return cnt


def build_without_image_input_ids(
    input_ids_with: torch.Tensor,
    loss_mask_with: torch.Tensor,
    image_token_ids: List[int],
):
    """
    基于已有 with-image input_ids 构造 without-image input_ids：
    - prompt 区域删掉 image special tokens
    - answer 区域保持不变
    """
    answer_pos = get_answer_positions(loss_mask_with)
    if answer_pos.numel() == 0:
        return input_ids_with.clone(), 0, 0, 0

    ans_start = int(answer_pos[0].item())

    prompt_ids = input_ids_with[:ans_start]
    answer_ids = input_ids_with[ans_start:]

    prompt_img_cnt = count_image_tokens(prompt_ids, image_token_ids)

    if len(image_token_ids) == 0:
        # 如果没识别出图像 token，就保持原样
        return input_ids_with.clone(), len(prompt_ids), len(prompt_ids), prompt_img_cnt

    keep_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    for tid in image_token_ids:
        keep_mask &= (prompt_ids != tid)

    prompt_wo = prompt_ids[keep_mask]
    input_ids_wo = torch.cat([prompt_wo, answer_ids], dim=0)

    return input_ids_wo, len(prompt_ids), len(prompt_wo), prompt_img_cnt


def to_pil_image_from_tensor(img: torch.Tensor) -> Image.Image:
    """
    将常见格式的图像 tensor 转成 PIL.Image
    支持:
      - (3, H, W)
      - (H, W, 3)
      - uint8 / float
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"pixel_values is not a torch.Tensor, got {type(img)}")

    x = img.detach().cpu()

    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 3:
        raise ValueError(f"Expected raw image tensor with 3 dims, got shape={tuple(x.shape)}")

    # (3, H, W) -> (H, W, 3)
    if x.shape[0] in [1, 3]:
        x = x.permute(1, 2, 0).contiguous()
    elif x.shape[-1] in [1, 3]:
        pass
    else:
        raise ValueError(f"Cannot interpret image tensor shape={tuple(img.shape)} as raw image")

    x = x.float()

    # 自动判断范围
    if x.max() <= 1.5:
        x = x * 255.0

    x = x.clamp(0, 255).byte().numpy()

    if x.shape[-1] == 1:
        x = x[..., 0]
        pil = Image.fromarray(x, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(x, mode="RGB")

    return pil


def prepare_qwen_image_inputs_from_ckpt(
    data: Dict[str, Any],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    """
    返回:
      qwen_pixel_values, image_grid_thw, source_desc

    优先级:
      1) 若 ckpt 里已经有 Qwen 风格字段: pixel_values + image_grid_thw
      2) 若只有原始图像 tensor (3,H,W 或 H,W,3)，用 processor.image_processor 现算
      3) 否则返回 None, None
    """
    pixel_values = data.get("pixel_values", None)
    image_grid_thw = data.get("image_grid_thw", None)

    # 情况1：已经是 Qwen 处理后的字段
    if pixel_values is not None and image_grid_thw is not None:
        pv = pixel_values
        grid = image_grid_thw

        if isinstance(pv, torch.Tensor):
            pv = pv.cpu()
        else:
            pv = torch.tensor(pv)

        if isinstance(grid, torch.Tensor):
            grid = grid.long().cpu()
        else:
            grid = torch.tensor(grid, dtype=torch.long)

        if pv.ndim == 3:
            pv = pv.unsqueeze(0)
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)

        return pv, grid, "ckpt:qwen_preprocessed"

    # 情况2：尝试把 ckpt 里的 pixel_values 当成原始图像
    if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
        try:
            pil_img = to_pil_image_from_tensor(pixel_values)
            img_out = processor.image_processor(images=pil_img, return_tensors="pt")

            pv = img_out["pixel_values"]
            grid = img_out.get("image_grid_thw", None)

            if grid is None:
                raise RuntimeError("processor.image_processor did not return image_grid_thw")

            return pv.cpu(), grid.long().cpu(), "ckpt:raw_image_reprocessed"
        except Exception as e:
            raise RuntimeError(
                f"Found pixel_values in ckpt, but failed to convert it into Qwen inputs. "
                f"Original shape={tuple(pixel_values.shape)} | error={repr(e)}"
            )

    # 情况3：没有图
    return None, None, "no_image"


@torch.no_grad()
def forward_with_image(
    input_ids: torch.Tensor,
    qwen_pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
):
    kwargs = {
        "input_ids": input_ids.unsqueeze(0).to(args.device),
    }

    if qwen_pixel_values is not None:
        kwargs["pixel_values"] = qwen_pixel_values.to(args.device, dtype=model_dtype)

    if image_grid_thw is not None:
        kwargs["image_grid_thw"] = image_grid_thw.to(args.device)

    outs = model(**kwargs)
    return outs.logits[0].detach().cpu()  # (L, V)


@torch.no_grad()
def forward_without_image(
    input_ids: torch.Tensor,
):
    kwargs = {
        "input_ids": input_ids.unsqueeze(0).to(args.device),
    }
    outs = model(**kwargs)
    return outs.logits[0].detach().cpu()  # (L, V)


def compute_answer_logprobs(
    logits: torch.Tensor,
    answer_token_ids: torch.Tensor,
    answer_start_idx: int,
) -> torch.Tensor:
    """
    对 answer 的每个 token y_t，取 teacher forcing 下的 log p(y_t | prefix)
    token 位置 j 的标签由 logits[j-1] 预测
    """
    A = answer_token_ids.shape[0]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    scores = []
    for i in range(A):
        pos = answer_start_idx + i
        if pos - 1 < 0 or pos - 1 >= logits.shape[0]:
            scores.append(torch.tensor(float("-inf")))
            continue
        token_id = int(answer_token_ids[i].item())
        scores.append(log_probs[pos - 1, token_id])

    return torch.stack(scores, dim=0)


def build_visual_static_mask(
    visual_dep_score: torch.Tensor,
    loss_mask: torch.Tensor,
    topk_ratio: float,
) -> torch.Tensor:
    mask = torch.zeros_like(loss_mask, dtype=torch.long)
    ans_pos = get_answer_positions(loss_mask)

    if ans_pos.numel() == 0:
        return mask

    ans_scores = visual_dep_score[ans_pos]
    valid_mask = torch.isfinite(ans_scores)
    valid_idx = ans_pos[valid_mask]

    if valid_idx.numel() == 0:
        return mask

    valid_scores = visual_dep_score[valid_idx]
    k = max(1, int(round(valid_scores.numel() * topk_ratio)))
    k = min(k, valid_scores.numel())

    top_idx_local = torch.topk(valid_scores, k=k, largest=True).indices
    chosen_pos = valid_idx[top_idx_local]
    mask[chosen_pos] = 1
    return mask


def decode_token(tid: int) -> str:
    try:
        return tokenizer.decode([tid], skip_special_tokens=False)
    except Exception:
        return f"<decode_error:{tid}>"


def process_one_ckpt(path: str):
    data = load_ckpt(path)

    input_ids_with = data["input_ids"].long().cpu()
    loss_mask_with = data["loss_mask"].long().cpu()

    answer_pos_with = get_answer_positions(loss_mask_with)

    if answer_pos_with.numel() == 0:
        data["visual_dep_score"] = torch.zeros_like(loss_mask_with, dtype=torch.float32)
        data["visual_static_mask"] = torch.zeros_like(loss_mask_with, dtype=torch.long)
        return data, {
            "answer_count": 0,
            "skip_reason": "no_answer"
        }

    ans_start_with = int(answer_pos_with[0].item())
    answer_token_ids = input_ids_with[answer_pos_with]
    answer_len = int(answer_token_ids.shape[0])

    input_ids_wo, prompt_len_with, prompt_len_wo, prompt_img_cnt = build_without_image_input_ids(
        input_ids_with=input_ids_with,
        loss_mask_with=loss_mask_with,
        image_token_ids=IMAGE_TOKEN_IDS,
    )

    ans_start_wo = int(input_ids_wo.shape[0] - answer_len)

    qwen_pixel_values, image_grid_thw, image_source = prepare_qwen_image_inputs_from_ckpt(data)

    logits_with = forward_with_image(
        input_ids=input_ids_with,
        qwen_pixel_values=qwen_pixel_values,
        image_grid_thw=image_grid_thw,
    )
    logits_wo = forward_without_image(input_ids_wo)

    logp_with = compute_answer_logprobs(
        logits=logits_with,
        answer_token_ids=answer_token_ids,
        answer_start_idx=ans_start_with,
    )
    logp_wo = compute_answer_logprobs(
        logits=logits_wo,
        answer_token_ids=answer_token_ids,
        answer_start_idx=ans_start_wo,
    )

    dep = logp_with - logp_wo

    visual_dep_score = torch.zeros_like(loss_mask_with, dtype=torch.float32)
    visual_dep_score[answer_pos_with[:dep.shape[0]]] = dep.float()

    visual_static_mask = build_visual_static_mask(
        visual_dep_score=visual_dep_score,
        loss_mask=loss_mask_with,
        topk_ratio=args.topk_ratio,
    )

    data["visual_dep_score"] = visual_dep_score
    data["visual_static_mask"] = visual_static_mask

    # 可选：把实际用于 Qwen 的视觉字段也存回去，方便后续复用
    if qwen_pixel_values is not None and image_grid_thw is not None:
        data["image_grid_thw"] = image_grid_thw.cpu()
        # 这里不强制覆盖原 pixel_values，避免把原始图像直接替换掉
        # 若原来没有 image_grid_thw，可写回供后续脚本直接使用

    debug_info = {
        "input_len_with": int(input_ids_with.shape[0]),
        "input_len_wo": int(input_ids_wo.shape[0]),
        "prompt_len_with": int(prompt_len_with),
        "prompt_len_wo": int(prompt_len_wo),
        "prompt_image_token_count": int(prompt_img_cnt),
        "answer_count": int(answer_pos_with.numel()),
        "ans_start_with": int(ans_start_with),
        "ans_start_wo": int(ans_start_wo),
        "logp_with": logp_with,
        "logp_wo": logp_wo,
        "dep": dep,
        "answer_pos_with": answer_pos_with,
        "input_ids_with": input_ids_with,
        "visual_static_mask": visual_static_mask,
        "image_source": image_source,
        "has_image_grid_thw": image_grid_thw is not None,
        "qwen_pixel_values_shape": None if qwen_pixel_values is None else tuple(qwen_pixel_values.shape),
        "image_grid_thw_shape": None if image_grid_thw is None else tuple(image_grid_thw.shape),
    }
    return data, debug_info


def print_debug(path: str, debug_info: Dict[str, Any]):
    print("\n" + "=" * 100)
    print(f"FILE: {path}")
    print(f"input_len_with       : {debug_info['input_len_with']}")
    print(f"input_len_wo         : {debug_info['input_len_wo']}")
    print(f"prompt_len_with      : {debug_info['prompt_len_with']}")
    print(f"prompt_len_wo        : {debug_info['prompt_len_wo']}")
    print(f"prompt_image_tokens  : {debug_info['prompt_image_token_count']}")
    print(f"answer_count         : {debug_info['answer_count']}")
    print(f"ans_start_with       : {debug_info['ans_start_with']}")
    print(f"ans_start_wo         : {debug_info['ans_start_wo']}")
    print(f"image_source         : {debug_info.get('image_source', 'unknown')}")
    print(f"has_image_grid_thw   : {debug_info.get('has_image_grid_thw', False)}")
    print(f"qwen_pixel_values_sh : {debug_info.get('qwen_pixel_values_shape', None)}")
    print(f"image_grid_thw_sh    : {debug_info.get('image_grid_thw_shape', None)}")

    answer_pos = debug_info["answer_pos_with"]
    input_ids = debug_info["input_ids_with"]
    logp_with = debug_info["logp_with"]
    logp_wo = debug_info["logp_wo"]
    dep = debug_info["dep"]
    vis_mask = debug_info["visual_static_mask"]

    show_n = min(40, answer_pos.numel())
    print("\nFirst answer tokens debug:")
    for i in range(show_n):
        p = int(answer_pos[i].item())
        tid = int(input_ids[p].item())
        tok = decode_token(tid)
        print(
            f"[{i:02d}] pos={p:4d} | tid={tid:6d} | tok={repr(tok):20s} "
            f"| logp_with={float(logp_with[i]):9.4f} "
            f"| logp_wo={float(logp_wo[i]):9.4f} "
            f"| dep={float(dep[i]):9.4f} "
            f"| vis_mask={int(vis_mask[p])}"
        )

    selected = torch.nonzero(vis_mask == 1, as_tuple=True)[0].tolist()
    print("\nSelected visual_static_mask positions:")
    print(selected[:100], "..." if len(selected) > 100 else "")

    if answer_pos.numel() > 0:
        ans_scores = dep[torch.isfinite(dep)]
        if ans_scores.numel() > 0:
            print(f"\nDep stats on answer:")
            print(f"  min  = {float(ans_scores.min()):.4f}")
            print(f"  max  = {float(ans_scores.max()):.4f}")
            print(f"  mean = {float(ans_scores.mean()):.4f}")


def main():
    files = list_ckpt_files(args.src_dir)
    if len(files) == 0:
        raise RuntimeError(f"No ckpt files found in {args.src_dir}")

    start = max(0, args.start)
    end = len(files) if args.end < 0 else min(args.end, len(files))
    files = files[start:end]

    if args.max_samples > 0:
        files = files[:args.max_samples]

    print(f"Will process {len(files)} files from {args.src_dir}")
    print(f"Overwrite mode: {args.overwrite}")
    print(f"Debug first n: {args.debug_first_n}")
    print(f"Model path: {args.model_path}")

    for idx, path in enumerate(tqdm(files, desc="Processing ckpt")):
        new_data, debug_info = process_one_ckpt(path)

        if idx < args.debug_first_n:
            print_debug(path, debug_info)

        if args.overwrite:
            torch.save(new_data, path)

    if args.overwrite:
        print("\nAll selected files have been overwritten.")
    else:
        print("\n[Dry-run] Not saved. Use --overwrite to really write fields into ckpt.")

    print("Done.")


if __name__ == "__main__":
    main()