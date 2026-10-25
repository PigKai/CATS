from __future__ import annotations
import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --- Paths (keep your sys.path hooks, but gate them) ---
# Sys.path will be set after parsing via --prefix to allow custom roots.

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import deepspeed
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open

# --------------------------- Helpers ---------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "t", "y", "yes"}

@dataclass
class TrainConfig:
    lr: float = 5e-5
    bs: int = 16
    gradient_accumulation_steps: int = 1
    datapath: str = ""  # set via --tmpdir; do not hardcode paths
    is_warmup: bool = True
    num_epochs: int = 20
    num_warmup_steps: int = 2000
    total_steps: int = 800000
    p_w: float = 0.1
    v_w: float = 1.0
    num_workers: int = 8
    embeding: bool = True
    act: str = "No"
    data_noise: bool = True
    noise: str = "uniform"  # or "gaussian"
    mean: float = 0.0
    std: float = 0.2
    residual: str = "true,norm"
    max_len: int = 2048
    config_path: str = ""  # derived from --prefix at runtime
    b1: float = 0.9
    b2: float = 0.95
    grad_clip: float = 0.5

# --------------------------- Argparse ---------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='EAGLE Train (Refactored)')
    parser.add_argument('--basepath', type=str, required=True)
    parser.add_argument('--tmpdir', type=str, required=True)
    parser.add_argument('--cpdir', type=str, required=True)
    parser.add_argument('--resume_path', type=str, help='Path to .bin or Deepspeed ckpt dir (for scoring or training)')
    parser.add_argument('--use_resume_for_training', type=str2bool,
                        help='True: resume_path used for filtering and TRAINING; False: resume_path only for FIRST filtering; training from scratch')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--epochs', type=int, required=True)

    parser.add_argument('--filter_start_epoch', type=int, nargs='+')
    parser.add_argument('--quantile', type=float, nargs='+')
    parser.add_argument('--index', type=int, nargs='+', choices=[0, 1],
                        help=('Metric index per prune step. Supported: 1 (1 - cosine(p, u)) and 0 (Random baseline that still builds Subset/DataLoader).'))

    parser.add_argument('--time_out_path', type=str, required=True)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_mode', type=str)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to EAGLE config directory. If not provided, will try to infer from prefix.')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='若设置，仅使用前 N 条训练数据做快速测试；不设则使用完整数据。例如 --max_train_samples 1000')

    # Let DeepSpeed inject its args last
    parser = deepspeed.add_config_arguments(parser)
    return parser

# --------------------------- IO utils ---------------------------

def list_files(path: str) -> List[str]:
    out = []
    for root, _, files in os.walk(path, followlinks=True):
        for f in files:
            out.append(os.path.join(root, f))
    return out

def list_files_sorted(path: str) -> List[str]:
    files = list_files(path)
    files.sort()
    return files

# --------------------------- Dataset ---------------------------

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean, self.std = mean, std
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = data["hidden_state_big"]
        data["hidden_state_big"] = x + (torch.randn_like(x) * self.std + self.mean)
        return data

class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = data["hidden_state_big"]
        noise = (torch.rand_like(x) - 0.5) * self.std * 512 / x.shape[1]
        data["hidden_state_big"] = x + noise
        return data

class CustomDataset(Dataset):
    def __init__(self, datapath: Sequence[str], transform=None, max_len: int = 2048):
        self.data = list(datapath)
        self.transform = transform
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.load(self.data[index])
        hidden_state = data['hidden_state'][: self.max_len][None, :]
        input_ids = data['input_ids'][: self.max_len][None, :]
        loss_mask = data['loss_mask'][: self.max_len][None, :]
        inputs_embeds = data["inputs_embeds"][: self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask_list = loss_mask[0].tolist()
        loss_mask_list[-1] = 0
        input_ids_target = input_ids[:, 1:]
        input_ids_target = torch.cat((input_ids_target, torch.tensor([[0]])), dim=1)

        inputs_embeds_target = inputs_embeds[:, 1:, :]
        zeropadding = torch.zeros(1, 1, inputs_embeds_target.shape[2])
        inputs_embeds_target = torch.cat((inputs_embeds_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        target = torch.cat((target, torch.zeros(1, 1, target.shape[2])), dim=1)
        item = {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask_list,
            "target": target,
            "hidden_state_big": hidden_state,
            "input_ids": input_ids_target,
            "inputs_embeds": inputs_embeds_target,
        }
        if self.transform:
            item = self.transform(item)
        return item

class DataCollatorWithPadding:
    @staticmethod
    def paddingtensor(tensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = tensors.shape
        out = tensors.new_zeros(B, N - n, S)
        return torch.cat((tensors, out), dim=1)
    @staticmethod
    def paddingtensor2D(tensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n = tensors.shape
        out = tensors.new_zeros(B, N - n, dtype=tensors.dtype)
        return torch.cat((tensors, out), dim=1)
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])

        batch_inputs_embeds = torch.cat([self.paddingtensor(item["inputs_embeds"], max_length) for item in features])
        
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor([item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor([item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        return {
            "input_ids": batch_input_ids,
            "inputs_embeds": batch_inputs_embeds,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }

# --------------------------- Model utils ---------------------------
# Model/EConfig will be imported in main() after sys.path is extended.


def load_lm_head_from_base(basepath: str) -> nn.Linear:
    """Safely load lm_head weights as frozen linear layer."""
    try:
        with open(os.path.join(basepath, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            emb_path = index_json["weight_map"][
                "language_model.model.embed_tokens.weight"
            ]
        with safe_open(
            os.path.join(basepath, emb_path), framework="pt", device="cpu"
        ) as f:
            tensor_slice = f.get_slice(
                "language_model.model.embed_tokens.weight"
            )
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        with open(os.path.join(basepath, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            emb_path = index_json["weight_map"][
                "language_model.model.embed_tokens.weight"
            ]
        weights = torch.load(os.path.join(basepath, emb_path))
        tensor = weights["language_model.model.embed_tokens.weight"].float()
    
    head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
    head.weight.data = tensor
    for p in head.parameters():
        p.requires_grad_(False)
    return head

# --------------------------- Scoring metrics ---------------------------

@torch.no_grad()
def score_batch_tokens(index: int,
                       head_engine: deepspeed.DeepSpeedEngine,
                       predict_valid: torch.Tensor,
                       target_valid: torch.Tensor) -> torch.Tensor:
    """Return per-token score for the ONLY supported metric index=1.
    Metric 1: 1 - cosine_similarity(p, u), where p = softmax(head(target_valid)), u = uniform.
    """
    if index != 1:
        raise ValueError(f"Unsupported metric index: {index}. Only 1 is supported here.")
    p = torch.softmax(head_engine(target_valid), dim=-1)
    u = torch.full_like(p, 1.0 / p.shape[-1])
    s = 1 - F.cosine_similarity(p, u, dim=-1)
    return s

# --------------------------- Pruning ---------------------------
# ===================================================================
# 3. DATA FILTERING LOGIC (prune_train_loader_once) — batch-level, 0/1 only
#   index=0: Random baseline (walk full flow; no forward; select random batches)
#   index=1: Flatness = 1 - cosine(p, u), p = softmax(head(target)); keep *smaller* batch scores
#   quantile = retain ratio (by number of batches)
# ===================================================================
@torch.no_grad()
def prune_train_loader_once(dataset, model_engine, head_engine,
                            batch_size, num_workers, rank, world_size,
                            quantile, index, cpdir, pruning_step,
                            score=None):
    t0 = time.time()

    if quantile >= 1.0:
        if rank == 0:
            print(f"[PRUNE] Step {pruning_step}: Skipping, quantile={quantile}>=1.0")
        return None, None, 0.0

    N_total = len(dataset)
    if rank == 0:
        print(f"[PRUNE] Starting Step {pruning_step} with Index={index}, Quantile={quantile} on {N_total} samples.")

    cache_dir = os.path.join(cpdir, "_prune_cache")
    os.makedirs(cache_dir, exist_ok=True)
    idx_path = os.path.join(cache_dir, f"keep_indices_step{pruning_step}_idx{index}.npy")

    # Common loader (walked for both 0/1 to keep flow consistent)
    score_loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=DataCollatorWithPadding(),
        drop_last=False,
        persistent_workers=(num_workers > 0)
    )

    final_indices_to_keep = []

    # ---------------- Random baseline (index=0): walk full flow, no forward ----------------
    if index == 0:
        if rank == 0:
            # Enumerate loader to keep flow identical (no compute)
            num_batches_scored = 0
            for _ in tqdm(score_loader, desc=f"Enumerate Step {pruning_step}", disable=(rank != 0)):
                num_batches_scored += 1

            # Randomly pick k batches then map to dataset indices
            k = max(1, int(round(quantile * max(1, num_batches_scored))))
            rng = np.random.default_rng(seed=0)
            batch_indices_to_keep = sorted(rng.choice(num_batches_scored, size=k, replace=False).tolist())

            current_dataset_indices = score_loader.dataset.indices if isinstance(score_loader.dataset, Subset) \
                                      else list(range(len(score_loader.dataset)))
            for b_idx in batch_indices_to_keep:
                start = b_idx * score_loader.batch_size
                end = min((b_idx + 1) * score_loader.batch_size, len(current_dataset_indices))
                final_indices_to_keep.extend(current_dataset_indices[start:end])

            final_indices_to_keep = sorted(final_indices_to_keep)

            tmp_idx = idx_path + ".tmp"
            with open(tmp_idx, "wb") as f:
                np.save(f, np.asarray(final_indices_to_keep, dtype=np.int64), allow_pickle=False)
            os.replace(tmp_idx, idx_path)

            kept = len(final_indices_to_keep)
            print(f"[PRUNE DONE] Step {pruning_step} (Random): kept {kept}/{N_total} ({kept/max(1,N_total):.2%})")

    # ---------------- Cosine flatness (index=1): batch-level mean of token scores ----------------
    elif index == 1:
        device = getattr(model_engine, "device", torch.device(f"cuda:{getattr(model_engine, 'local_rank', rank)}"))
        was_training = getattr(model_engine.module, "training", True)
        model_engine.eval()
        head_engine.eval()

        if rank == 0:
            conf_list = []
            TOK_CHUNK = 64

            for bidx, data in enumerate(tqdm(score_loader, desc=f"Scoring Step {pruning_step}", disable=(rank != 0))):
                # Keep forward call to preserve flow symmetry; must pass inputs_embeds to match training
                _ = model_engine(
                    data["hidden_states"].to(device, non_blocking=True),
                    input_ids=data["input_ids"].to(device, non_blocking=True),
                    inputs_embeds=data["inputs_embeds"].to(device, non_blocking=True),
                    attention_mask=data["attention_mask"].to(device, non_blocking=True)
                )

                target = data["target"].to(device, non_blocking=True)
                mask = data["loss_mask"].to(device, non_blocking=True).bool()

                mask_flat = mask.view(-1)
                target_valid = target.view(-1, target.shape[-1])[mask_flat]

                # Free ASAP
                del target, mask, mask_flat

                if target_valid.shape[0] == 0:
                    conf_avg = torch.tensor(0.0, device=device, dtype=torch.float32)
                else:
                    n_valid = int(target_valid.shape[0])
                    total_score = 0.0
                    head_dtype = next(head_engine.parameters()).dtype

                    with torch.inference_mode():
                        for s in range(0, n_valid, TOK_CHUNK):
                            e = min(s + TOK_CHUNK, n_valid)
                            hs_chunk_p = target_valid[s:e].to(head_dtype)           # [m, H]，与 head 一致避免 dtype 报错
                            logits_p = head_engine(hs_chunk_p)                      # [m, V]
                            p = torch.softmax(logits_p, dim=-1)                     # [m, V]
                            # cos(p, u) where u is uniform: 1 / (sqrt(V) * ||p||)
                            p_norm = torch.linalg.norm(p, dim=-1).clamp_min(1e-12)  # [m]
                            inv_sqrt_vocab = 1.0 / math.sqrt(p.shape[-1])
                            score_chunk = 1.0 - (inv_sqrt_vocab / p_norm)           # [m]
                            total_score += score_chunk.float().sum().item()
                            del hs_chunk_p, logits_p, p, p_norm, score_chunk

                    conf_avg = torch.tensor(total_score / max(1, n_valid), device=device, dtype=torch.float32)

                conf_list.append(conf_avg.detach().float().cpu())

            conf_data = torch.stack(conf_list) if conf_list else torch.zeros(0)
            num_batches_scored = len(conf_data)

            if num_batches_scored == 0:
                batch_indices_to_keep = []
            else:
                k = max(1, int(round(quantile * num_batches_scored)))
                idx_sorted = torch.argsort(conf_data, descending=False)  # keep smaller (flatter) batches
                batch_indices_to_keep = idx_sorted[:k].tolist()

            current_dataset_indices = score_loader.dataset.indices if isinstance(score_loader.dataset, Subset) \
                                      else list(range(len(score_loader.dataset)))

            for b_idx in batch_indices_to_keep:
                start = b_idx * score_loader.batch_size
                end = min((b_idx + 1) * score_loader.batch_size, len(current_dataset_indices))
                final_indices_to_keep.extend(current_dataset_indices[start:end])

            final_indices_to_keep = sorted(final_indices_to_keep)

            tmp_idx = idx_path + ".tmp"
            with open(tmp_idx, "wb") as f:
                np.save(f, np.asarray(final_indices_to_keep, dtype=np.int64), allow_pickle=False)
            os.replace(tmp_idx, idx_path)

            kept = len(final_indices_to_keep)
            print(f"[PRUNE DONE] Step {pruning_step}: scored={N_total}, kept={kept}/{N_total} ({kept/max(1,N_total):.2%})")

        if was_training:
            model_engine.train()

    else:
        if rank == 0:
            raise ValueError(f"Unsupported metric index: {index}. Use 0 (Random) or 1 (Cosine flatness).")

    # All ranks wait/load exactly like your second code (no timeout_s here)
    if rank != 0:
        t0w = time.time()
        wait_timeout = int(os.environ.get("PRUNE_WAIT_TIMEOUT", "86400"))
        while not os.path.exists(idx_path):
            time.sleep(0.5)
            if time.time() - t0w > wait_timeout:
                raise RuntimeError(f"Timeout waiting for keep indices at {idx_path}.")

    torch.distributed.barrier()
    final_indices_to_keep = np.load(idx_path).tolist()

    ds_micro_bs = model_engine.train_micro_batch_size_per_gpu()
    original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    pruned_dataset = Subset(original_dataset, final_indices_to_keep)

    sampler = DistributedSampler(pruned_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    pruned_loader = DataLoader(
        pruned_dataset, batch_size=ds_micro_bs, sampler=sampler,
        num_workers=num_workers, pin_memory=True, collate_fn=DataCollatorWithPadding(),
        drop_last=True, persistent_workers=(num_workers > 0)
    )
    t1 = time.time()
    return pruned_loader, sampler, (t1 - t0)

# --------------------------- Loss / Metrics ---------------------------

def top_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True) for k in topk]


def compute_loss(target, predict, head_engine, loss_mask, criterion, v_w: float, p_w: float):
    if loss_mask.dim() == 3:
        loss_mask = loss_mask.squeeze(-1)
    H = predict.shape[-1]
    active_mask = loss_mask.view(-1) > 0
    predict_sel = predict.view(-1, H)[active_mask]
    target_sel = target.view(-1, H)[active_mask]
    head_dtype = next(head_engine.parameters()).dtype
    predict_sel = predict_sel.to(head_dtype)
    target_sel = target_sel.to(head_dtype)
    out_head = head_engine(predict_sel)
    target_head = head_engine(target_sel)
    out_logq = nn.LogSoftmax(dim=-1)(out_head)
    target_p = nn.Softmax(dim=-1)(target_head)
    kl = -target_p * out_logq
    ploss = kl.sum(dim=-1).mean()
    vloss = criterion(predict_sel, target_sel).mean()
    loss = v_w * vloss + p_w * ploss
    return vloss, ploss, loss, out_head, target_head, active_mask

# --------------------------- Main ---------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Extend sys.path according to --prefix
    prefix = args.prefix.rstrip('/')
    flat_root = f"{prefix}/Flatness"
    for p in (flat_root, f"{flat_root}/eagle", f"{flat_root}/data"):
        if p not in sys.path:
            sys.path.append(p)

    # Now that paths are set, import project modules
    from model.cnets import Model
    from model.configs import EConfig

    # Init distributed first (so rank is known before printing)
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # Validate prune args
    if args.filter_start_epoch:
        if not (len(args.filter_start_epoch) == len(args.quantile or []) == len(args.index or [])):
            if rank == 0:
                print("\033[91mError: --filter_start_epoch, --quantile, --index must have same length.\033[0m")
            sys.exit(1)

    # Reproducibility
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(0)
    np.random.seed(0)

    # Accelerator for metric gather
    accelerator = Accelerator(mixed_precision="fp16")

    # WANDB (no hardcoded API keys; respect env)
    if rank == 0:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
                       config={"epochs": args.epochs})
        except Exception as e:
            print(f"[WARN] W&B init failed: {e}")

    # Build train config (respect CLI)
    tcfg = TrainConfig(datapath=args.tmpdir, num_epochs=args.epochs)

    # Load datasets
    datapath = list_files_sorted(tcfg.datapath)
    train_files = datapath[: int(len(datapath) * 0.95)]
    test_files = datapath[int(len(datapath) * 0.95):]
    if getattr(args, 'max_train_samples', None) is not None and args.max_train_samples > 0:
        n_orig = len(train_files)
        train_files = train_files[: args.max_train_samples]
        if rank == 0:
            print(f"[DEBUG] 仅使用前 {len(train_files)} 条训练数据做测试（完整为 {n_orig}），后续可去掉 --max_train_samples 使用完整数据")

    transform = None
    if tcfg.data_noise:
        if tcfg.noise == "uniform":
            transform = AddUniformNoise(std=tcfg.std)
        elif tcfg.noise == "gaussian":
            transform = AddGaussianNoise(mean=tcfg.mean, std=tcfg.std)

    traindataset = CustomDataset(train_files, transform=transform, max_len=tcfg.max_len)
    testdataset = CustomDataset(test_files, max_len=tcfg.max_len)

    if rank == 0:
        os.makedirs(args.cpdir, exist_ok=True)
        os.makedirs(os.path.dirname(args.time_out_path), exist_ok=True)
        with open(args.time_out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch", "did_prune", "kept_samples_ratio", "filter_time_sec", "train_time_sec",
                "avg_load_time_per_batch_sec", "epoch_loss", "epoch_acc", "train_acc_percent",
                "top1_acc", "top2_acc", "top3_acc", "max_mem_allocated_mb", "max_mem_reserved_mb",
                "cur_mem_allocated_mb", "cur_mem_reserved_mb", "avg_mem_allocated_mb_per_batch",
                "avg_mem_reserved_mb_per_batch",
            ])

    # Config & model
    if args.config_path:
        config_path = args.config_path
    else:
        # Try to infer config path from prefix
        possible_paths = [
            f"{prefix}/Flatness/eagle/train/EAGLE-LLaMA3-Instruct-8B",
            f"{prefix}/train/llava_7B_config.json",
            f"{prefix}/train/qwenvl2_5_7B_config.json",
        ]
        config_path = None
        for pp in possible_paths:
            if os.path.exists(pp):
                config_path = pp
                break
        if config_path is None:
            if rank == 0:
                print(f"[WARN] Could not find config at any of: {possible_paths}")
                print(f"[WARN] Using default: {possible_paths[0]}")
            config_path = possible_paths[0]
    
    if rank == 0:
        print(f"[CONFIG] Loading config from: {config_path}")
    config = EConfig.from_pretrained(config_path)
    model = Model(config, path=args.basepath, load_emb=True)

    # Resume-for-training behavior
    if args.use_resume_for_training and args.resume_path and os.path.isfile(args.resume_path) and args.resume_path.endswith('.bin'):
        if rank == 0:
            print(f"===> Loading TRAIN weights from {args.resume_path}")
        model.load_state_dict(torch.load(args.resume_path, map_location="cpu"), strict=False)
    else:
        if rank == 0:
            print("From Zero for TRAINING!!!")

    # Initialize engines
    criterion = nn.SmoothL1Loss(reduction="none")
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(), training_data=traindataset,
        collate_fn=DataCollatorWithPadding())

    device = getattr(model_engine, "device", torch.device("cuda", rank))
    head = load_lm_head_from_base(args.basepath)
    head = head.to(device, dtype=torch.bfloat16)
    head.eval()
    # head 仅用于推理（scoring/loss），参数已冻结，不交给 DeepSpeed，避免 ZeRO 对空可训练参数列表报错
    head_engine = head

    # Training state
    pruned_sampler = None
    num_pruning_steps_completed = 0
    original_train_size = len(traindataset)
    first_prune_used_resume = False

    _wall_clock_start = time.time()
    sum_train_time = 0.0

    for epoch in range(tcfg.num_epochs):
        did_prune, kept_ratio_str, filter_time_sum = 0, "", 0.0

        # Cascading pruning inside epoch
        if args.filter_start_epoch:
            while (
                num_pruning_steps_completed < len(args.filter_start_epoch)
                and (epoch + 1) == args.filter_start_epoch[num_pruning_steps_completed]
            ):
                q = float(args.quantile[num_pruning_steps_completed])
                idx = int(args.index[num_pruning_steps_completed])

                if rank == 0:
                    print(f"[PRUNE] Epoch {epoch+1}: step {num_pruning_steps_completed} (idx={idx}, q={q})")

                # Choose engine for scoring on FIRST prune (if not using resume for training)
                engine_for_scoring = model_engine
                temp_engine = None
                temp_model = None
                if (not args.use_resume_for_training) and (not first_prune_used_resume) and args.resume_path\
                        and os.path.isfile(args.resume_path) and args.resume_path.endswith('.bin') and rank == 0:
                    from model.cnets import Model as _Model
                    temp_model = _Model(config, path=args.basepath, load_emb=True)
                    print(f"===> [SCORING-ONLY] Loading resume weights from {args.resume_path}")
                    temp_model.load_state_dict(torch.load(args.resume_path, map_location="cpu"), strict=False)
                    temp_engine, _, _, _ = deepspeed.initialize(
                        args=args, model=temp_model, model_parameters=temp_model.parameters(),
                        collate_fn=DataCollatorWithPadding())
                    engine_for_scoring = temp_engine

                tl, pruned_sampler_new, ft = prune_train_loader_once(
                    traindataset, engine_for_scoring, head_engine,
                    batch_size=tcfg.bs, num_workers=tcfg.num_workers,
                    rank=rank, world_size=world_size,
                    quantile=q, index=idx, cpdir=args.cpdir, pruning_step=num_pruning_steps_completed,
                )

                # Cleanup temp scorer after first prune
                if (not args.use_resume_for_training) and (not first_prune_used_resume):
                    if rank == 0 and temp_engine is not None:
                        del temp_engine
                    if rank == 0 and temp_model is not None:
                        del temp_model
                    torch.cuda.empty_cache()
                    first_prune_used_resume = True

                filter_time_sum += float(ft or 0.0)

                if tl is not None:
                    train_loader = tl
                    traindataset = tl.dataset
                    pruned_sampler = pruned_sampler_new
                    did_prune = 1
                    if rank == 0:
                        kept = len(traindataset)
                        kept_ratio_str = f"{kept / max(1, original_train_size):.4f}"

                torch.cuda.empty_cache()
                num_pruning_steps_completed += 1

        if num_pruning_steps_completed > 0 and pruned_sampler is not None:
            pruned_sampler.set_epoch(epoch)

        # Train epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)
        t_train_start = time.time()

        top_ks = [0.0, 0.0, 0.0]
        correct = total = 0
        epoch_loss = 0.0
        num_batches = 0
        load_time_sum = 0.0
        mem_alloc_sum = mem_resrv_sum = 0.0

        t_load_start = time.time()
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(rank != 0)):
            t_load_end = time.time()
            load_time_sum += (t_load_end - t_load_start)

            hidden_states = data["hidden_states"].to(device, non_blocking=True)
            input_ids = data["input_ids"].to(device, non_blocking=True)
            inputs_embeds = data["inputs_embeds"].to(device, non_blocking=True)
            attention_mask = data["attention_mask"].to(device, non_blocking=True)
            target = data["target"].to(device, non_blocking=True)
            loss_mask = data["loss_mask"].to(device, non_blocking=True)

            predict = model_engine(
                hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            vloss, ploss, loss, out_head, target_head, _ = compute_loss(
                target, predict, head_engine, loss_mask, criterion, tcfg.v_w, tcfg.p_w)

            model_engine.backward(loss)
            model_engine.step()

            if torch.cuda.is_available():
                mem_alloc_sum += torch.cuda.memory_allocated(device) / (1024 ** 2)
                mem_resrv_sum += torch.cuda.memory_reserved(device) / (1024 ** 2)

            if out_head is not None:
                with torch.no_grad():
                    _, pred_tok = out_head.max(dim=-1)
                    _, tgt_tok = target_head.max(dim=-1)
                    Ntok = out_head.shape[0]
                    cc = (pred_tok == tgt_tok).sum().item()
                    tk = top_accuracy(out_head, tgt_tok, (1, 2, 3))
                    top_ks[0] += tk[0].item(); top_ks[1] += tk[1].item(); top_ks[2] += tk[2].item()
                    total += Ntok; correct += cc

            epoch_loss += loss.item(); num_batches += 1
            t_load_start = time.time()

        train_time = time.time() - t_train_start

        # Gather metrics
        dev = device
        # Reuse existing accelerator (initialized earlier)
        correct_t, total_t = torch.tensor(correct, device=dev), torch.tensor(total, device=dev)
        t1_t, t2_t, t3_t = [torch.tensor(x, device=dev) for x in top_ks]
        correct_all, total_all, t1_all, t2_all, t3_all = accelerator.gather_for_metrics(
            (correct_t, total_t, t1_t, t2_t, t3_t))
        correct_sum, total_sum = correct_all.sum().item(), total_all.sum().item()
        top1_sum, top2_sum, top3_sum = t1_all.sum().item(), t2_all.sum().item(), t3_all.sum().item()

        epoch_loss /= max(1, num_batches)
        epoch_acc = (correct_sum / (total_sum + 1e-5)) if total_sum > 0 else 0.0
        top1 = (top1_sum / (total_sum + 1e-5))
        top2 = (top2_sum / (total_sum + 1e-5))
        top3 = (top3_sum / (total_sum + 1e-5))

        avg_load_time = (load_time_sum / max(1, num_batches)) if num_batches > 0 else 0.0

        if torch.cuda.is_available():
            max_alloc_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
            max_resrv_mb = torch.cuda.max_memory_reserved(dev) / (1024 ** 2)
            cur_alloc_mb = torch.cuda.memory_allocated(dev) / (1024 ** 2)
            cur_resrv_mb = torch.cuda.memory_reserved(dev) / (1024 ** 2)
        else:
            max_alloc_mb = max_resrv_mb = cur_alloc_mb = cur_resrv_mb = 0.0

        # Log & CSV
        if rank == 0:
            try:
                import wandb
                wandb.log({
                    "train/epochacc": epoch_acc,
                    "train/epochloss": epoch_loss,
                    "train/top1": top1,
                    "train/top2": top2,
                    "train/top3": top3,
                    "sys/max_alloc_mb": max_alloc_mb,
                    "sys/max_resrv_mb": max_resrv_mb,
                })
            except Exception:
                pass

            with open(args.time_out_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch + 1, did_prune, kept_ratio_str, f"{filter_time_sum:.2f}", f"{train_time:.2f}", f"{avg_load_time:.6f}",
                    f"{epoch_loss:.6f}", f"{epoch_acc:.6f}", f"{epoch_acc * 100:.2f}", f"{top1:.6f}", f"{top2:.6f}", f"{top3:.6f}",
                    f"{max_alloc_mb:.2f}", f"{max_resrv_mb:.2f}", f"{cur_alloc_mb:.2f}", f"{cur_resrv_mb:.2f}",
                    f"{(mem_alloc_sum / max(1, num_batches)):.2f}", f"{(mem_resrv_sum / max(1, num_batches)):.2f}",
                ])

        sum_train_time += train_time

        # Checkpoints (16-bit weights for deepspeed engine)
        save_dir = f"{args.cpdir}/state_{epoch}"
        model_engine.save_16bit_model(save_dir)
        # EaModel.from_pretrained(ea_model_path=save_dir) 需要目录下有 config.json
        if rank == 0:
            config_src = config_path if os.path.isfile(config_path) else os.path.join(config_path, "config.json")
            if os.path.exists(config_src):
                shutil.copy(config_src, os.path.join(save_dir, "config.json"))
        if (epoch + 1) % 10 == 0:
            deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=save_dir)

    # Final timing
    if rank == 0:
        total_wall_clock = time.time() - _wall_clock_start
        save_path = os.path.join(args.cpdir, "total_training_time.txt")
        with open(save_path, "w") as f:
            f.write(f"total_train_time_sec={sum_train_time:.2f}\n")
            f.write(f"total_train_time_hms={str(timedelta(seconds=int(sum_train_time)))}\n")
            f.write(f"total_wall_clock_sec={total_wall_clock:.2f}\n")
            f.write(f"total_wall_clock_hms={str(timedelta(seconds=int(total_wall_clock)))}\n")
        with open(args.time_out_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([]); w.writerow(["TOTALS"])
            w.writerow(["total_train_time_sec", f"{sum_train_time:.2f}"])
            w.writerow(["total_train_time_hms", str(timedelta(seconds=int(sum_train_time)))] )
            w.writerow(["total_wall_clock_sec", f"{total_wall_clock:.2f}"])
            w.writerow(["total_wall_clock_hms", str(timedelta(seconds=int(total_wall_clock)))])
        print(f"[TIME] Total train time: {sum_train_time:.2f}s ({str(timedelta(seconds=int(sum_train_time)))})")


if __name__ == "__main__":
    main()
