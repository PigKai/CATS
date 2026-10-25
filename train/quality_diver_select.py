# add_sample_score_with_draft.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoConfig


torch.backends.cuda.matmul.allow_tf32 = True


# =========================================================
# 默认路径
# =========================================================

DEFAULT_PROJECT_ROOT = "/media/user/e/xyk/eagle-eye2/eagle-eye2"

DEFAULT_DATA_DIR = (
    "/media/user/d/xyk/data1/llava_instruct/qwen2.5vl/"
    "qwen2.5vl_7B_0_67999/0"
)

DEFAULT_MODEL_PATH = "/media/user/d/xyk/model/Qwen2.5-VL-7B-Instruct"

DEFAULT_DRAFT_STATE_PATH = (
    "/media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/"
    "endoSpec_distill/ScienceQA/stage1/state_20"
)

DEFAULT_CONFIG_PATH = (
    "/media/user/e/xyk/eagle-eye2/eagle-eye2/train/"
    "qwenvl2_5_7B_config.json"
)


# =========================================================
# 固定字段
# =========================================================

INPUT_HIDDEN_KEY = "hidden_state"
TARGET_HIDDEN_KEY = "hidden_state"
VISUAL_SCORE_KEY = "visual_dep_score"


# =========================================================
# sample_score 计算权重
# =========================================================

VISUAL_WEIGHT = 0.6
HARD_WEIGHT = 0.4

# SR 小于该值，认为 draft model 在这个 token 上比较困难
SR_THRESHOLD = 0.8

# 视觉分数取 top 20% token 的均值，避免被大量普通 token 稀释
VISUAL_TOP_RATIO = 0.2


# =========================================================
# 1. 文件读取
# =========================================================

def ordered_list_files(path: str, pattern: str = r"data_(\d+)\.ckpt$") -> List[str]:
    files_with_numbers: List[Tuple[int, str]] = []
    rgx = re.compile(pattern)

    for root, _, files in os.walk(path):
        for file in files:
            m = rgx.match(file)
            if m:
                files_with_numbers.append((int(m.group(1)), os.path.join(root, file)))

    files_with_numbers.sort(key=lambda x: x[0])
    return [fp for _, fp in files_with_numbers]


# =========================================================
# 2. Dataset / Collator
#    基本复用你原训练代码的数据组织方式
# =========================================================

class QwenHiddenStateDataset(Dataset):
    def __init__(self, file_list: List[str], max_len: int):
        self.files = file_list
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        file_path = self.files[index]
        data = torch.load(file_path, map_location="cpu")

        if INPUT_HIDDEN_KEY not in data:
            raise KeyError(f"{file_path} missing {INPUT_HIDDEN_KEY}")

        if TARGET_HIDDEN_KEY not in data:
            raise KeyError(f"{file_path} missing {TARGET_HIDDEN_KEY}")

        if "input_ids" not in data:
            raise KeyError(f"{file_path} missing input_ids")

        if "loss_mask" not in data:
            raise KeyError(f"{file_path} missing loss_mask")

        if VISUAL_SCORE_KEY not in data:
            raise KeyError(f"{file_path} missing {VISUAL_SCORE_KEY}")

        input_hidden_state = data[INPUT_HIDDEN_KEY][: self.max_len][None, :]
        target_hidden_state = data[TARGET_HIDDEN_KEY][: self.max_len][None, :]
        input_ids = data["input_ids"][: self.max_len][None, :]

        lm = data["loss_mask"][: self.max_len][None, :]
        loss_mask = lm[0].tolist()

        if len(loss_mask) > 0:
            loss_mask[-1] = 0

        inputs_embeds = None
        if "inputs_embeds" in data:
            inputs_embeds = data["inputs_embeds"][: self.max_len][None, :]

        visual_dep_score = data[VISUAL_SCORE_KEY][: self.max_len]
        if isinstance(visual_dep_score, torch.Tensor):
            visual_dep_score = visual_dep_score.float().view(-1)
        else:
            visual_dep_score = torch.tensor(visual_dep_score, dtype=torch.float32).view(-1)

        if visual_dep_score.shape[0] > 0:
            visual_dep_score[-1] = 0.0

        length = input_hidden_state.shape[1]
        attention_mask = [1] * length

        # 和你原始训练代码保持一致：
        # input_ids / target / inputs_embeds 都右移一位
        input_ids_target = torch.cat(
            [
                input_ids[:, 1:],
                torch.zeros((1, 1), dtype=input_ids.dtype),
            ],
            dim=1,
        )

        target = torch.cat(
            [
                target_hidden_state[:, 1:, :],
                torch.zeros(
                    (1, 1, target_hidden_state.shape[2]),
                    dtype=target_hidden_state.dtype,
                ),
            ],
            dim=1,
        )

        if inputs_embeds is not None:
            inputs_embeds_target = torch.cat(
                [
                    inputs_embeds[:, 1:, :],
                    torch.zeros(
                        (1, 1, inputs_embeds.shape[2]),
                        dtype=inputs_embeds.dtype,
                    ),
                ],
                dim=1,
            )
        else:
            inputs_embeds_target = None

        return {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "target": target,
            "hidden_state_big": input_hidden_state,
            "input_ids": input_ids_target,
            "inputs_embeds": inputs_embeds_target,
            "index": index,
            "visual_dep_score": visual_dep_score,
        }


class DataCollatorWithPadding:
    def _pad3d(self, x: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = x.shape
        if n == N:
            return x

        pad = torch.zeros(B, N - n, S, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def _pad2d(self, x: torch.Tensor, N: int) -> torch.Tensor:
        B, n = x.shape
        if n == N:
            return x

        pad = torch.zeros(B, N - n, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(item["hidden_state_big"].shape[1] for item in features)

        batch_input_ids = torch.cat(
            [self._pad2d(item["input_ids"], max_len) for item in features],
            dim=0,
        )

        batch_hidden_states = torch.cat(
            [self._pad3d(item["hidden_state_big"], max_len) for item in features],
            dim=0,
        )

        batch_target = torch.cat(
            [self._pad3d(item["target"], max_len) for item in features],
            dim=0,
        )

        batch_loss_mask = torch.tensor(
            [
                item["loss_mask"] + [0] * (max_len - len(item["loss_mask"]))
                for item in features
            ],
            dtype=torch.long,
        )

        batch_attention_mask = torch.tensor(
            [
                item["attention_mask"] + [0] * (max_len - len(item["attention_mask"]))
                for item in features
            ],
            dtype=torch.long,
        )

        has_embeds = any(item.get("inputs_embeds", None) is not None for item in features)
        batch_inputs_embeds = None

        if has_embeds:
            embeds_list = []
            for item in features:
                emb = item.get("inputs_embeds", None)
                if emb is None:
                    emb = torch.zeros_like(item["hidden_state_big"])
                embeds_list.append(self._pad3d(emb, max_len))

            batch_inputs_embeds = torch.cat(embeds_list, dim=0)

        batch_visual_dep_score = torch.stack(
            [
                torch.cat(
                    [
                        item["visual_dep_score"],
                        torch.zeros(
                            max_len - item["visual_dep_score"].shape[0],
                            dtype=torch.float32,
                        ),
                    ],
                    dim=0,
                )
                for item in features
            ],
            dim=0,
        )

        batch_indices = [item["index"] for item in features]

        out = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "index": batch_indices,
            "visual_dep_score": batch_visual_dep_score,
        }

        if batch_inputs_embeds is not None:
            out["inputs_embeds"] = batch_inputs_embeds

        return out


# =========================================================
# 3. Qwen2.5-VL lm_head
# =========================================================

def build_qwen_lm_head(basepath: str) -> nn.Linear:
    baseconfig = AutoConfig.from_pretrained(basepath, trust_remote_code=True)

    if hasattr(baseconfig, "text_config") and hasattr(baseconfig.text_config, "hidden_size"):
        hidden_size = baseconfig.text_config.hidden_size
        vocab_size = baseconfig.text_config.vocab_size
    else:
        hidden_size = getattr(baseconfig, "hidden_size")
        vocab_size = getattr(baseconfig, "vocab_size")

    head = nn.Linear(hidden_size, vocab_size, bias=False)

    candidate_keys = [
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
        "language_model.model.lm_head.weight",

        # 有些模型 lm_head 和 embedding 权重绑定
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "embed_tokens.weight",
        "language_model.embed_tokens.weight",
    ]

    idx_path = os.path.join(basepath, "model.safetensors.index.json")

    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            index_json = json.load(f)

        weight_map = index_json["weight_map"]
        key = next((k for k in candidate_keys if k in weight_map), None)

        if key is None:
            raise KeyError(f"None of {candidate_keys} found in safetensors index.")

        head_path = weight_map[key]

        with safe_open(os.path.join(basepath, head_path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key).float()

    else:
        idx_path = os.path.join(basepath, "pytorch_model.bin.index.json")

        if not os.path.exists(idx_path):
            raise FileNotFoundError(
                f"Neither model.safetensors.index.json nor pytorch_model.bin.index.json exists under {basepath}"
            )

        with open(idx_path, "r", encoding="utf-8") as f:
            index_json = json.load(f)

        weight_map = index_json["weight_map"]
        key = next((k for k in candidate_keys if k in weight_map), None)

        if key is None:
            raise KeyError(f"None of {candidate_keys} found in pytorch_model.bin index.")

        head_path = weight_map[key]
        weights = torch.load(os.path.join(basepath, head_path), map_location="cpu")
        tensor = weights[key].float()

    head.weight.data = tensor
    head.eval()

    for p in head.parameters():
        p.requires_grad = False

    return head


# =========================================================
# 4. 加载 EAGLE draft model
# =========================================================

def import_eagle_model(project_root: str):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from eagle_eye.model.cnets import Model
        from eagle_eye.model.configs import EConfig
    except Exception:
        from model.cnets import Model
        from model.configs import EConfig

    return Model, EConfig


def load_state_dict_from_accelerate_state(state_dir: str) -> Dict[str, torch.Tensor]:
    """
    读取 accelerate.save_state 保存出来的 model 权重。

    通常 state_20 目录里会有：
      - model.safetensors
      - model_1.safetensors
      - optimizer.bin
      - scheduler.bin
      - random_states_0.pkl

    这里优先读取 model.safetensors。
    """

    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
    ]

    for name in candidates:
        path = os.path.join(state_dir, name)

        if not os.path.exists(path):
            continue

        if name.endswith(".safetensors"):
            return safe_load_file(path, device="cpu")

        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict) and "state_dict" in obj:
            return obj["state_dict"]

        return obj

    raise FileNotFoundError(
        f"Cannot find model.safetensors / pytorch_model.bin / model.bin under {state_dir}"
    )


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}

    for k, v in state_dict.items():
        new_k = k

        if new_k.startswith("module."):
            new_k = new_k[len("module.") :]

        if new_k.startswith("model."):
            # 注意：这里只在出现明显多包一层 model. 的情况下可能需要。
            # 如果你的 Model 本身参数名就是 model.xxx，strict=False 会兜底。
            pass

        new_state[new_k] = v

    return new_state


def build_and_load_draft_model(
    project_root: str,
    config_path: str,
    model_path: str,
    draft_state_path: str,
    device: torch.device,
):
    Model, EConfig = import_eagle_model(project_root)

    config = EConfig.from_pretrained(config_path)
    draft_model = Model(config, load_emb=True, path=model_path)

    state_dict = load_state_dict_from_accelerate_state(draft_state_path)
    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = draft_model.load_state_dict(state_dict, strict=False)

    print(f"[Draft Model] loaded from: {draft_state_path}")
    print(f"[Draft Model] missing keys: {len(missing)}")
    print(f"[Draft Model] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[Draft Model] first missing keys:", missing[:10])

    if len(unexpected) > 0:
        print("[Draft Model] first unexpected keys:", unexpected[:10])

    draft_model.to(device)
    draft_model.eval()

    for p in draft_model.parameters():
        p.requires_grad = False

    return draft_model


# =========================================================
# 5. forward 和 SR 计算
# =========================================================

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    return out


def model_forward(model, batch: Dict[str, Any]) -> torch.Tensor:
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }

    if "inputs_embeds" in batch and batch["inputs_embeds"] is not None:
        kwargs["inputs_embeds"] = batch["inputs_embeds"]

    return model(batch["hidden_states"], **kwargs)


@torch.no_grad()
def compute_surprise_ratio_by_chunks(
    head: nn.Module,
    target_hidden: torch.Tensor,
    predict_hidden: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    严格等价于你原代码里的：

        target_head = head(target)
        target_p = softmax(target_head)
        true_token_ids = argmax(target_p)

        out_head_for_sr = head(predict)
        probs_student = softmax(out_head_for_sr)

        true_token_probs = gather(probs_student, true_token_ids)
        student_max_probs = max(probs_student)

        surprise_ratio = true_token_probs / student_max_probs

    这里为了省显存，不显式保存完整 softmax。
    因为：

        softmax(logit_true) / max(softmax(logits))
        = exp(logit_true - max_logit)

    所以 SR 数值等价。
    """

    B, S, H = target_hidden.shape
    sr_out = []

    target_flat = target_hidden.reshape(B * S, H)
    predict_flat = predict_hidden.reshape(B * S, H)

    total = target_flat.shape[0]

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)

        target_chunk = target_flat[start:end].float()
        predict_chunk = predict_flat[start:end].float()

        target_logits = head(target_chunk)
        true_token_ids = torch.argmax(target_logits, dim=-1, keepdim=True)

        draft_logits = head(predict_chunk)

        true_token_logits = torch.gather(
            draft_logits,
            dim=-1,
            index=true_token_ids,
        ).squeeze(-1)

        max_logits = draft_logits.max(dim=-1).values

        sr = torch.exp(true_token_logits - max_logits)
        sr = torch.nan_to_num(sr, nan=1.0, posinf=1.0, neginf=0.0)
        sr = torch.clamp(sr, min=0.0, max=1.0)

        sr_out.append(sr.detach().cpu())

        del target_chunk
        del predict_chunk
        del target_logits
        del draft_logits

    sr = torch.cat(sr_out, dim=0)
    sr = sr.view(B, S)

    return sr


# =========================================================
# 6. sample-level 原始分数
# =========================================================

@torch.no_grad()
def compute_batch_raw_scores(
    batch: Dict[str, Any],
    draft_model: nn.Module,
    head: nn.Module,
    device: torch.device,
    chunk_size: int,
    use_bf16: bool,
) -> List[Dict[str, float]]:
    batch = move_batch_to_device(batch, device)

    if device.type == "cuda" and use_bf16:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        predict = model_forward(draft_model, batch)

    if not torch.isfinite(predict).all():
        print("[Warning] predict has non-finite values.")

    sr = compute_surprise_ratio_by_chunks(
        head=head,
        target_hidden=batch["target"],
        predict_hidden=predict,
        chunk_size=chunk_size,
    )

    raw_loss_mask = batch["loss_mask"].detach().cpu().bool()
    visual_dep_score = batch["visual_dep_score"].detach().cpu().float()

    results = []

    for i in range(raw_loss_mask.shape[0]):
        valid_mask = raw_loss_mask[i]

        # -----------------------------
        # 指标 1：视觉相关分数
        # -----------------------------
        visual_valid = visual_dep_score[i][valid_mask]
        visual_valid = visual_valid[torch.isfinite(visual_valid)]

        if visual_valid.numel() == 0:
            visual_raw_score = 0.0
        else:
            topk = max(1, int(visual_valid.numel() * VISUAL_TOP_RATIO))
            topk = min(topk, visual_valid.numel())
            visual_raw_score = float(torch.topk(visual_valid, k=topk).values.mean().item())

        # -----------------------------
        # 指标 2：draft model 困难度
        # -----------------------------
        sr_valid = sr[i][valid_mask]
        sr_valid = sr_valid[torch.isfinite(sr_valid)]

        if sr_valid.numel() == 0:
            hard_raw_score = 0.0
        else:
            # SR 越小，draft model 越不认可 target token
            hard_mean = float((1.0 - sr_valid).mean().item())
            hard_ratio = float((sr_valid < SR_THRESHOLD).float().mean().item())

            hard_raw_score = 0.5 * hard_mean + 0.5 * hard_ratio

        results.append(
            {
                "visual_raw_score": float(visual_raw_score),
                "hard_raw_score": float(hard_raw_score),
            }
        )

    del batch
    del predict
    del sr

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# =========================================================
# 7. rank normalize + 写回 sample_score
# =========================================================

def rank_normalize(values: List[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)

    if len(arr) <= 1:
        return np.ones_like(arr, dtype=np.float64)

    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)

    return ranks / max(1, len(arr) - 1)


def write_sample_scores(files: List[str], sample_scores: Dict[str, float]):
    for fp in tqdm(files, desc="Writing sample_score"):
        data = torch.load(fp, map_location="cpu")

        # 只写入这一个字段
        data["sample_score"] = float(sample_scores[fp])

        torch.save(data, fp)


# =========================================================
# 8. 主流程
# =========================================================

def add_sample_score(args):
    device = torch.device(args.device)

    print(f"[Data Dir] {args.data_dir}")
    print(f"[Model Path] {args.model_path}")
    print(f"[Draft State] {args.draft_state_path}")
    print(f"[Config Path] {args.config_path}")
    print(f"[Device] {device}")

    files = ordered_list_files(args.data_dir)

    if len(files) == 0:
        raise RuntimeError(f"No data_*.ckpt found under: {args.data_dir}")

    print(f"[Data] found {len(files)} ckpt files")

    print("[Load] loading Qwen2.5-VL lm_head ...")
    head = build_qwen_lm_head(args.model_path)
    head.to(device)
    head.eval()
    print("[Load] lm_head loaded")

    print("[Load] loading fixed EAGLE draft model ...")
    draft_model = build_and_load_draft_model(
        project_root=args.project_root,
        config_path=args.config_path,
        model_path=args.model_path,
        draft_state_path=args.draft_state_path,
        device=device,
    )
    print("[Load] draft model loaded")

    dataset = QwenHiddenStateDataset(
        file_list=files,
        max_len=args.max_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    records = []

    for batch in tqdm(loader, desc="Scoring"):
        batch_indices = batch["index"]

        try:
            batch_scores = compute_batch_raw_scores(
                batch=batch,
                draft_model=draft_model,
                head=head,
                device=device,
                chunk_size=args.chunk_size,
                use_bf16=args.use_bf16,
            )

            for idx, score_dict in zip(batch_indices, batch_scores):
                fp = files[int(idx)]

                records.append(
                    {
                        "file": fp,
                        "visual_raw_score": score_dict["visual_raw_score"],
                        "hard_raw_score": score_dict["hard_raw_score"],
                    }
                )

        except Exception as e:
            print(f"[Error] batch failed: {repr(e)}")

            for idx in batch_indices:
                fp = files[int(idx)]

                records.append(
                    {
                        "file": fp,
                        "visual_raw_score": 0.0,
                        "hard_raw_score": 0.0,
                    }
                )

    # =====================================================
    # 全局 rank 归一化
    # =====================================================
    visual_raw_scores = [r["visual_raw_score"] for r in records]
    hard_raw_scores = [r["hard_raw_score"] for r in records]

    visual_rank_scores = rank_normalize(visual_raw_scores)
    hard_rank_scores = rank_normalize(hard_raw_scores)

    sample_scores = {}

    for i, r in enumerate(records):
        final_score = (
            VISUAL_WEIGHT * float(visual_rank_scores[i])
            + HARD_WEIGHT * float(hard_rank_scores[i])
        )

        sample_scores[r["file"]] = float(final_score)

    # =====================================================
    # 只写入 sample_score 一个字段
    # =====================================================
    write_sample_scores(files, sample_scores)

    print("[Done] 所有 ckpt 已新增 sample_score 字段。")
    print("[Done] sample_score 越高，表示该样本越值得被选入训练。")


# =========================================================
# 9. main
# =========================================================

def main():
    parser = argparse.ArgumentParser("Add sample_score with fixed EAGLE draft model")

    parser.add_argument(
        "--project_root",
        type=str,
        default=DEFAULT_PROJECT_ROOT,
        help="eagle-eye2/eagle-eye2 项目根目录",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="原始 data_*.ckpt 目录",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Qwen2.5-VL-7B-Instruct 模型目录",
    )

    parser.add_argument(
        "--draft_state_path",
        type=str,
        default=DEFAULT_DRAFT_STATE_PATH,
        help="固定 EAGLE draft model 的 state 目录",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="EAGLE draft model config 路径",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="建议保持 1，避免显存爆",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="lm_head 分块计算 token 数。显存不够就改成 16 或 32。",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=True,
        help="cuda 下默认使用 bf16 autocast",
    )

    args = parser.parse_args()

    add_sample_score(args)


if __name__ == "__main__":
    main()