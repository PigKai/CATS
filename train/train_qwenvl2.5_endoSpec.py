import argparse
import collections
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup


torch.backends.cuda.matmul.allow_tf32 = True

# project root: .../eagle-eye2/eagle-eye2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)


# -----------------------------
# File listing
# -----------------------------
def normal_list_files(path: str) -> List[str]:
    datapath: List[str] = []
    for root, _, files in os.walk(path):
        for file in files:
            datapath.append(os.path.join(root, file))
    return datapath



def ordered_list_files(path: str, pattern: str = r"data_(\d+)\.ckpt$") -> List[str]:
    rgx = re.compile(pattern)
    files_with_numbers: List[Tuple[int, str]] = []
    for root, _, files in os.walk(path):
        for file in files:
            m = rgx.match(file)
            if not m:
                continue
            files_with_numbers.append((int(m.group(1)), os.path.join(root, file)))
    files_with_numbers.sort(key=lambda x: x[0])
    return [fp for _, fp in files_with_numbers]


# -----------------------------
# Noise augment
# -----------------------------
class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        tensor = sample["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        sample["hidden_state_big"] = tensor + noise
        return sample


class AddUniformNoise:
    def __init__(self, std: float = 0.0):
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        tensor = sample["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / max(1, tensor.shape[1])
        sample["hidden_state_big"] = tensor + noise
        return sample


# -----------------------------
# Dataset / Collator
# -----------------------------
class QwenHiddenStateDataset(Dataset):
    """
    Each ckpt sample is expected to contain:
      - hidden_state: (seq, hidden)
      - input_ids: (seq,)
      - loss_mask: (seq,)
    Optionally:
      - inputs_embeds: (seq, hidden)
    """

    def __init__(
        self,
        file_list: List[str],
        max_len: int,
        transform=None,
        cache_loss_mask: bool = False,
    ):
        self.files = file_list
        self.max_len = max_len
        self.transform = transform
        self.cache_loss_mask = cache_loss_mask
        self.mask_cache: Dict[int, List[int]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = torch.load(self.files[index], map_location="cpu")
        new: Dict[str, Any] = {}

        hidden_state = data["hidden_state"][: self.max_len][None, :]
        input_ids = data["input_ids"][: self.max_len][None, :]

        if self.cache_loss_mask and index in self.mask_cache:
            loss_mask = self.mask_cache[index]
        else:
            lm = data["loss_mask"][: self.max_len][None, :]
            loss_mask = lm[0].tolist()
            if len(loss_mask) > 0:
                loss_mask[-1] = 0
            if self.cache_loss_mask:
                self.mask_cache[index] = loss_mask

        inputs_embeds = None
        if "inputs_embeds" in data:
            inputs_embeds = data["inputs_embeds"][: self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length

        input_ids_target = torch.cat(
            [input_ids[:, 1:], torch.zeros((1, 1), dtype=input_ids.dtype)], dim=1
        )
        target = torch.cat(
            [
                hidden_state[:, 1:, :],
                torch.zeros((1, 1, hidden_state.shape[2]), dtype=hidden_state.dtype),
            ],
            dim=1,
        )

        if inputs_embeds is not None:
            inputs_embeds_target = torch.cat(
                [
                    inputs_embeds[:, 1:, :],
                    torch.zeros((1, 1, inputs_embeds.shape[2]), dtype=inputs_embeds.dtype),
                ],
                dim=1,
            )
        else:
            inputs_embeds_target = None

        new["attention_mask"] = attention_mask
        new["loss_mask"] = loss_mask
        new["target"] = target
        new["hidden_state_big"] = hidden_state
        new["input_ids"] = input_ids_target
        new["inputs_embeds"] = inputs_embeds_target
        new["index"] = index

        if self.transform:
            new = self.transform(new)
        return new


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

        batch_input_ids = torch.cat([self._pad2d(item["input_ids"], max_len) for item in features], dim=0)
        batch_hidden_states = torch.cat([self._pad3d(item["hidden_state_big"], max_len) for item in features], dim=0)
        batch_target = torch.cat([self._pad3d(item["target"], max_len) for item in features], dim=0)

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

        batch_loss_mask = torch.tensor(
            [item["loss_mask"] + [0] * (max_len - len(item["loss_mask"])) for item in features],
            dtype=torch.long,
        )
        batch_attention_mask = torch.tensor(
            [item["attention_mask"] + [0] * (max_len - len(item["attention_mask"])) for item in features],
            dtype=torch.long,
        )
        batch_indices = [item["index"] for item in features]

        out = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "index": batch_indices,
        }
        if batch_inputs_embeds is not None:
            out["inputs_embeds"] = batch_inputs_embeds
        return out


# -----------------------------
# Metrics / Loss
# -----------------------------
def top_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            res.append(correct[:k].reshape(-1).float().sum(0, keepdim=True))
        return res


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    inputs_embeds = data.get("inputs_embeds", None)
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, sl = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    hidden_states_headout = head(hidden_states)

    for i in range(bs):
        for j in range(sl):
            single_hidden_states = hidden_states[i, :j]
            single_input_ids = input_ids[i, :j]

            single_hidden_states = single_hidden_states[None, :, :]
            single_input_ids = single_input_ids[None, :]

            if inputs_embeds is not None:
                single_inputs_embeds = inputs_embeds[i, :j][None, :, :]
            else:
                single_inputs_embeds = None

            for k in range(max_length):
                if single_hidden_states.shape[1] == 0:
                    break
                if loss_mask[i, single_hidden_states.shape[1] - 1] == 0:
                    break

                tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                target_in_token = torch.argmax(tmp_in_target_headout)
                target_out_token = torch.argmax(tmp_out_target_headout)
                tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]

                if not (target_in_token == tmp_token):
                    break

                model_kwargs = {"input_ids": single_input_ids}
                if single_inputs_embeds is not None:
                    model_kwargs["inputs_embeds"] = single_inputs_embeds
                out_hidden = model(single_hidden_states, **model_kwargs)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                total[k] += 1
                if token == target_out_token:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

                single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                single_input_ids = torch.cat(
                    (single_input_ids, torch.tensor([[token]], device=single_input_ids.device)), dim=1
                )

                if single_inputs_embeds is not None and hasattr(model, "embed_tokens"):
                    next_embed = model.embed_tokens(token.unsqueeze(0)).unsqueeze(0)
                    single_inputs_embeds = torch.cat((single_inputs_embeds, next_embed), dim=1)

    acc = [correct[i] / total[i] if total[i] > 0 else 0.0 for i in range(len(correct))]
    return acc



def compute_v_p_loss(
    *,
    head: nn.Module,
    criterion: nn.Module,
    target_hidden: torch.Tensor,
    target_p: torch.Tensor,
    predict_hidden: torch.Tensor,
    loss_mask_3d: torch.Tensor,
):
    out_head = head(predict_hidden)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask_3d * plogp, dim=2)) / (loss_mask_3d.sum() + 1e-5)

    vloss = criterion(predict_hidden, target_hidden)
    vloss = torch.sum(torch.mean(loss_mask_3d * vloss, dim=2)) / (loss_mask_3d.sum() + 1e-5)
    return vloss, ploss, out_head


# -----------------------------
# Qwen2.5-VL LM head builder
# -----------------------------
def build_qwen_lm_head(basepath: str) -> Tuple[nn.Linear, AutoConfig]:
    baseconfig = AutoConfig.from_pretrained(basepath)

    if hasattr(baseconfig, "text_config") and hasattr(baseconfig.text_config, "hidden_size"):
        hidden_size = baseconfig.text_config.hidden_size
        vocab_size = baseconfig.text_config.vocab_size
    else:
        hidden_size = getattr(baseconfig, "hidden_size")
        vocab_size = getattr(baseconfig, "vocab_size")

    head = nn.Linear(hidden_size, vocab_size, bias=False)

    candidate_keys = [
        # explicit lm_head
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
        "language_model.model.lm_head.weight",

        # tied embedding fallback
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "embed_tokens.weight",
        "language_model.embed_tokens.weight",
    ]

    tensor = None
    idx_path = os.path.join(basepath, "model.safetensors.index.json")

    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            index_json = json.load(f)

        weight_map = index_json["weight_map"]
        key = next((k for k in candidate_keys if k in weight_map), None)

        if key is None:
            debug_keys = [
                k for k in weight_map.keys()
                if ("lm_head" in k) or ("embed_tokens" in k)
            ]
            print("DEBUG basepath =", basepath)
            print("DEBUG head/embed-like keys =", debug_keys[:100])
            raise KeyError(
                f"None of {candidate_keys} found in safetensors index. "
                f"Found related keys: {debug_keys[:50]}"
            )

        weight_file = weight_map[key]
        with safe_open(os.path.join(basepath, weight_file), framework="pt", device="cpu") as f:
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
            debug_keys = [
                k for k in weight_map.keys()
                if ("lm_head" in k) or ("embed_tokens" in k)
            ]
            print("DEBUG basepath =", basepath)
            print("DEBUG head/embed-like keys =", debug_keys[:100])
            raise KeyError(
                f"None of {candidate_keys} found in pytorch_model.bin index. "
                f"Found related keys: {debug_keys[:50]}"
            )

        weight_file = weight_map[key]
        weights = torch.load(os.path.join(basepath, weight_file), map_location="cpu")
        tensor = weights[key].float()

    if tensor.shape != head.weight.data.shape:
        raise ValueError(
            f"Weight shape mismatch: loaded {tuple(tensor.shape)} vs head {tuple(head.weight.data.shape)}"
        )

    head.weight.data.copy_(tensor)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    print(f"[build_qwen_lm_head] loaded weight from key: {key}")
    return head, baseconfig



def model_forward(model, batch: Dict[str, Any]) -> torch.Tensor:
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    if "inputs_embeds" in batch and batch["inputs_embeds"] is not None:
        kwargs["inputs_embeds"] = batch["inputs_embeds"]
    return model(batch["hidden_states"], **kwargs)


# -----------------------------
# Stage1 / Stage2 runners
# -----------------------------
def run_stage1(
    args,
    accelerator: Accelerator,
    model: nn.Module,
    head: nn.Module,
    criterion: nn.Module,
    aug,
    file_lister,
    wandb,
):
    if accelerator.is_main_process:
        os.makedirs(args.stage1_outdir, exist_ok=True)

    stage1_files = file_lister(args.stage1_datadir)
    if len(stage1_files) == 0:
        raise RuntimeError(f"[Stage1] No ckpt files found under: {args.stage1_datadir}")

    stage1_train = stage1_files[: int(0.95 * len(stage1_files))]
    stage1_test = stage1_files[int(0.95 * len(stage1_files)) :]

    train_ds = QwenHiddenStateDataset(stage1_train, max_len=args.max_len, transform=aug)
    test_ds = QwenHiddenStateDataset(stage1_test, max_len=args.max_len, transform=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.stage1_bs,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.stage1_bs,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.stage1_lr, betas=(0.9, 0.95))

    steps_per_epoch = int(np.ceil(len(train_ds) / args.stage1_bs))
    update_steps_per_epoch = int(np.ceil(steps_per_epoch / accelerator.gradient_accumulation_steps))
    total_steps = max(1, args.stage1_epochs * update_steps_per_epoch)
    warmup_steps = int(args.stage1_warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model_p, head_p, optimizer_p, train_loader_p, test_loader_p, scheduler_p = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )

    start_epoch = 0
    if args.resume_stage1_state is not None:
        accelerator.print(f"[Stage1] Resuming from: {args.resume_stage1_state}")
        accelerator.load_state(args.resume_stage1_state)
        match = re.search(r"state_(\d+)", args.resume_stage1_state)
        if match:
            start_epoch = int(match.group(1))
            accelerator.print(f"[Stage1] Resume start epoch = {start_epoch}")

    for epoch in range(start_epoch, args.stage1_epochs):
        model_p.train()

        correct = 0.0
        total = 0.0
        epoch_loss = 0.0
        num_batches = 0

        for step_idx, batch in enumerate(tqdm(train_loader_p, disable=not accelerator.is_local_main_process), start=0):
            with accelerator.accumulate(model_p):
                optimizer_p.zero_grad(set_to_none=True)
                predict = model_forward(model_p, batch)

                with torch.no_grad():
                    target_head = head_p(batch["target"])
                    target_p = torch.softmax(target_head, dim=2).detach()

                loss_mask_3d = batch["loss_mask"][:, :, None].to(predict.dtype)

                vloss, ploss, out_head = compute_v_p_loss(
                    head=head_p,
                    criterion=criterion,
                    target_hidden=batch["target"],
                    target_p=target_p,
                    predict_hidden=predict,
                    loss_mask_3d=loss_mask_3d,
                )
                loss = args.v_w * vloss + args.p_w * ploss

                if accelerator.is_local_main_process and step_idx % 200 == 0:
                    accelerator.print(
                        f"[S1] ep={epoch+1} step={step_idx} "
                        f"loss={loss.item()} vloss={vloss.item()} ploss={ploss.item()} "
                        f"finite={torch.isfinite(loss).item()}"
                    )

                if not torch.isfinite(loss):
                    accelerator.print(
                        f"[S1 NaN] ep={epoch+1} step={step_idx} "
                        f"vloss={vloss.item()} ploss={ploss.item()} "
                        f"predict_finite={torch.isfinite(predict).all().item()} "
                        f"target_head_finite={torch.isfinite(target_head).all().item()} "
                        f"out_head_finite={torch.isfinite(out_head).all().item()}"
                    )
                    continue


                accelerator.backward(loss)
                accelerator.clip_grad_value_(model_p.parameters(), args.grad_clip)
                optimizer_p.step()
                scheduler_p.step()

            with torch.no_grad():
                _, pred_ids = out_head.max(dim=2)
                _, tgt_ids = target_head.max(dim=2)
                ct = loss_mask_3d.sum().item()
                cc = ((pred_ids == tgt_ids) * loss_mask_3d.squeeze(-1)).sum().item()
                total += ct
                correct += cc
                epoch_loss += loss.item()
                num_batches += 1

            if accelerator.is_main_process and wandb is not None and ct > 0:
                log_lr = optimizer_p.optimizer.param_groups[0]["lr"] if hasattr(optimizer_p, "optimizer") else optimizer_p.param_groups[0]["lr"]
                wandb.log(
                    {
                        "stage1/lr": log_lr,
                        "stage1/vloss": float(vloss.item()),
                        "stage1/ploss": float(ploss.item()),
                        "stage1/loss": float(loss.item()),
                        "stage1/acc_step": float(cc / ct),
                    }
                )

        correct_t = accelerator.gather_for_metrics(torch.tensor(correct, device=accelerator.device)).sum().item()
        total_t = accelerator.gather_for_metrics(torch.tensor(total, device=accelerator.device)).sum().item()
        epoch_loss = epoch_loss / max(1, num_batches)

        if accelerator.is_local_main_process:
            accelerator.print(
                f"[Stage1] Epoch {epoch + 1}/{args.stage1_epochs} | loss={epoch_loss:.4f} | "
                f"acc={100 * correct_t / max(1, total_t):.2f}%"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "stage1/epoch_loss": float(epoch_loss),
                        "stage1/epoch_acc": float(correct_t / max(1, total_t)),
                    }
                )

        if (epoch + 1) % args.save_freq == 0 and accelerator.is_local_main_process:
            accelerator.save_state(output_dir=os.path.join(args.stage1_outdir, f"state_{epoch + 1}"))

    if accelerator.is_local_main_process:
        accelerator.save_state(output_dir=os.path.join(args.stage1_outdir, "state_final"))

    return stage1_train, model_p, head_p, optimizer_p



def run_stage2(
    args,
    accelerator: Accelerator,
    model_p: nn.Module,
    head_p: nn.Module,
    optimizer_p,
    criterion: nn.Module,
    file_lister,
    stage1_train_list: Optional[List[str]],
    wandb,
):
    if accelerator.is_main_process:
        os.makedirs(args.stage2_outdir, exist_ok=True)
        os.makedirs(args.result_dir, exist_ok=True)

    data2_files = file_lister(args.stage2_datadir)
    if len(data2_files) == 0:
        raise RuntimeError(f"[Stage2] No ckpt files found under: {args.stage2_datadir}")

    data2_train = data2_files[: int(0.95 * len(data2_files))]
    data2_test = data2_files[int(0.95 * len(data2_files)) :]

    if stage1_train_list is not None:
        data1_train = stage1_train_list
    else:
        data1_files = file_lister(args.stage1_datadir)
        if len(data1_files) == 0:
            raise RuntimeError(f"[Stage2] No ckpt files found under: {args.stage1_datadir} (Stage2 replay needs data1)")
        data1_train = data1_files[: int(0.95 * len(data1_files))]

    if len(data1_train) > len(data2_train):
        accelerator.print(f"[Stage2] Truncating Stage1 replay data from {len(data1_train)} to {len(data2_train)}")
        data1_train = data1_train[: len(data2_train)]

    combined_train = data2_train
    distill_boundary_idx = 0

    train_ds = QwenHiddenStateDataset(combined_train, max_len=args.max_len, transform=None, cache_loss_mask=False)
    test_ds = QwenHiddenStateDataset(data2_test, max_len=args.max_len, transform=None, cache_loss_mask=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.stage2_bs,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer_p.state = collections.defaultdict(dict)
    for pg in optimizer_p.param_groups:
        pg["lr"] = args.stage2_lr
        pg["initial_lr"] = args.stage2_lr

    steps_per_epoch = int(np.ceil(len(train_ds) / args.stage2_bs))
    update_steps_per_epoch = int(np.ceil(steps_per_epoch / accelerator.gradient_accumulation_steps))
    total_steps = max(1, args.stage2_epochs * update_steps_per_epoch)

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer_p,
        num_warmup_steps=min(args.stage2_warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    train_loader_p, test_loader_p, scheduler2_p = accelerator.prepare(train_loader, test_loader, scheduler2)

    if args.resume_stage2_state is not None:
        accelerator.print(f"[Stage2] Resuming from: {args.resume_stage2_state}")
        accelerator.load_state(args.resume_stage2_state)
    elif args.init_stage2_from_stage1_state is not None:
        accelerator.print(f"[Stage2] Initializing from stage1 state: {args.init_stage2_from_stage1_state}")
        accelerator.load_state(args.init_stage2_from_stage1_state)

    for group in optimizer_p.param_groups:
        for p in group["params"]:
            state = optimizer_p.state[p]
            if len(state) > 0 and "step" in state:
                state["step"] = torch.tensor(0.0, device=p.device)

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer_p,
        num_warmup_steps=min(args.stage2_warmup_steps, total_steps),
        num_training_steps=total_steps,
    )
    scheduler2_p = accelerator.prepare(scheduler2)

    accelerator.print(
        f"[Stage2] Scheduler has been reset. Warmup steps: {min(args.stage2_warmup_steps, total_steps)}, "
        f"Total steps: {total_steps}"
    )
    for pg in optimizer_p.param_groups:
        pg["lr"] = args.stage2_lr

    global_sr_mean = 1.0
    total_token_list: List[float] = []
    selected_token_list: List[float] = []
    epoch_time_list: List[float] = []

    start_epoch = 0
    if args.resume_stage2_state is not None:
        match = re.search(r"state_(\d+)", args.resume_stage2_state)
        if match:
            start_epoch = int(match.group(1))

    for epoch in range(start_epoch, args.stage2_epochs):
        model_p.train()
        start_time = time.time()

        correct = 0.0
        total = 0.0
        epoch_loss = 0.0
        num_batches = 0
        epoch_threshold_sum = 0.0

        for step_idx, batch in enumerate(
            tqdm(train_loader_p, disable=not accelerator.is_local_main_process), start=0
        ):
            with accelerator.accumulate(model_p):
                optimizer_p.zero_grad(set_to_none=True)
                predict = model_forward(model_p, batch)

                with torch.no_grad():
                    target_head = head_p(batch["target"])
                    target_p = torch.softmax(target_head / args.teacher_temp, dim=2).detach()

                    out_head_for_sr = head_p(predict)
                    probs_student = torch.softmax(out_head_for_sr, dim=-1)

                    true_token_ids = torch.argmax(target_p, dim=-1, keepdim=True)
                    # if accelerator.is_local_main_process and step_idx % 200 == 0:
                    #     finite_predict = torch.isfinite(predict)
                    #     if finite_predict.any():
                    #         accelerator.print(
                    #             f"[PredictStats] step={step_idx} "
                    #             f"predict_min={predict[finite_predict].min().item():.6e} "
                    #             f"predict_max={predict[finite_predict].max().item():.6e} "
                    #             f"predict_mean={predict[finite_predict].mean().item():.6e}"
                    #         )
                    #     else:
                    #         accelerator.print(f"[PredictStats] step={step_idx} predict has no finite values")

                    #     finite_logits = torch.isfinite(out_head_for_sr)
                    #     if finite_logits.any():
                    #         accelerator.print(
                    #             f"[LogitStats] step={step_idx} "
                    #             f"logit_min={out_head_for_sr[finite_logits].min().item():.6e} "
                    #             f"logit_max={out_head_for_sr[finite_logits].max().item():.6e} "
                    #             f"logit_mean={out_head_for_sr[finite_logits].mean().item():.6e}"
                    #         )
                    #     else:
                    #         accelerator.print(f"[LogitStats] step={step_idx} out_head_for_sr has no finite values")
                    true_token_probs = torch.gather(probs_student, 2, true_token_ids).squeeze(-1)
                    student_max_probs, _ = probs_student.max(dim=-1)

                    surprise_ratio = true_token_probs / (student_max_probs + 1e-8)
                    raw_loss_mask = batch["loss_mask"]

                    for i in range(raw_loss_mask.shape[0]):
                        total_token_list.append(float(raw_loss_mask[i].sum().item()))

                    valid_sr = surprise_ratio[raw_loss_mask.bool()]
                    if valid_sr.numel() > 0:
                        batch_mean = float(valid_sr.mean().item())
                        global_sr_mean = args.ema_momentum * global_sr_mean + (1 - args.ema_momentum) * batch_mean
                    if accelerator.is_local_main_process and step_idx % 200 == 0:
                        valid_mask = torch.isfinite(valid_sr)
                        accelerator.print(
                            f"[SRTrace3] step={step_idx} "
                            f"valid_sr_count={valid_sr.numel()} "
                            f"valid_sr_finite_count={valid_mask.sum().item()} "
                            f"valid_sr_mean={(valid_sr[valid_mask].mean().item() if valid_mask.any() else float('nan'))}"
                        )
                    current_threshold = global_sr_mean * args.difficulty_scale
                    epoch_threshold_sum += current_threshold

                    if args.disable_dynamic_mask:
                        dynamic_mask = raw_loss_mask.clone()
                        selection_ratio = float(dynamic_mask.sum().item() / (raw_loss_mask.sum().item() + 1e-8))
                        replay_tokens = float(raw_loss_mask.sum().item())
                        hard_new_tokens = 0.0
                        is_replay_sample = torch.zeros_like(raw_loss_mask, dtype=torch.bool)
                    else:
                        is_hard_token = surprise_ratio < current_threshold

                        batch_indices = batch["index"]
                        is_replay_sample = torch.tensor(
                            [idx < distill_boundary_idx for idx in batch_indices],
                            device=raw_loss_mask.device,
                            dtype=torch.bool,
                        ).unsqueeze(1)

                        final_select_mask = is_hard_token | is_replay_sample
                        dynamic_mask = raw_loss_mask * final_select_mask.long()

                        total_active_tokens = raw_loss_mask.sum().item()
                        selected_active_tokens = dynamic_mask.sum().item()
                        selection_ratio = selected_active_tokens / (total_active_tokens + 1e-8)
                        replay_tokens = (raw_loss_mask * is_replay_sample.long()).sum().item()
                        hard_new_tokens = (
                            raw_loss_mask * (~is_replay_sample).long() * is_hard_token.long()
                        ).sum().item()

                    for i in range(dynamic_mask.shape[0]):
                        selected_token_list.append(float(dynamic_mask[i].sum().item()))

                    loss_mask_3d = dynamic_mask[:, :, None].to(predict.dtype)
                    if accelerator.is_local_main_process and step_idx % 200 == 0:
                        total_active_tokens = raw_loss_mask.sum().item()
                        selected_active_tokens = dynamic_mask.sum().item()

                        if args.disable_dynamic_mask:
                            hard_token_count = 0
                            replay_sample_count = 0
                            replay_token_count = 0
                            hard_new_token_count = 0
                        else:
                            hard_token_count = is_hard_token.long().sum().item()
                            replay_sample_count = is_replay_sample.long().sum().item()
                            replay_token_count = (raw_loss_mask * is_replay_sample.long()).sum().item()
                            hard_new_token_count = (
                                raw_loss_mask * (~is_replay_sample).long() * is_hard_token.long()
                            ).sum().item()

                        # accelerator.print(
                        #     f"[MaskDebug] step={step_idx} "
                        #     f"sr_mean={global_sr_mean:.6f} "
                        #     f"thr={current_threshold:.6f} "
                        #     f"selected={selected_active_tokens}/{total_active_tokens} "
                        #     f"hard_token_count={hard_token_count} "
                        #     f"replay_samples={replay_sample_count} "
                        #     f"replay_tokens={replay_token_count} "
                        #     f"hard_new_tokens={hard_new_token_count}"
                        # )

                        if not args.disable_dynamic_mask and raw_loss_mask.shape[0] > 0:
                            hard_pos = torch.nonzero(
                                (raw_loss_mask * (~is_replay_sample).long() * is_hard_token.long())[0],
                                as_tuple=False
                            ).squeeze(-1)

                            if hard_pos.numel() > 0:
                                show_n = min(10, hard_pos.numel())
                                accelerator.print(
                                    f"[HardTokenPos] first {show_n} hard token positions: {hard_pos[:show_n].tolist()}"
                                )
                            else:
                                accelerator.print("[HardTokenPos] no hard token selected in sample 0")

                vloss, ploss, out_head = compute_v_p_loss(
                    head=head_p,
                    criterion=criterion,
                    target_hidden=batch["target"],
                    target_p=target_p,
                    predict_hidden=predict,
                    loss_mask_3d=loss_mask_3d,
                )
                loss = args.v_w * vloss + args.p_w * ploss
                if accelerator.is_local_main_process and step_idx % 200 == 0:
                    accelerator.print(
                        f"[LossDebug] step={step_idx} "
                        f"vloss={float(vloss):.6f} "
                        f"ploss={float(ploss):.6f} "
                        f"loss={float(loss):.6f} "
                        f"loss_finite={torch.isfinite(loss).item()}"
                    )

                accelerator.backward(loss)
                grad_norm = 0.0
                if accelerator.sync_gradients:
                    for p in model_p.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5

                accelerator.clip_grad_value_(model_p.parameters(), args.grad_clip)
                optimizer_p.step()
                scheduler2_p.step()

                current_lr = optimizer_p.param_groups[0]["lr"]
                with torch.no_grad():
                    total_tokens = raw_loss_mask.sum().item()
                    selected_tokens = dynamic_mask.sum().item()
                    s_ratio = selected_tokens / (total_tokens + 1e-8)
                    replay_count = (raw_loss_mask * is_replay_sample.long()).sum().item()
                    hard_new_count = max(0.0, selected_tokens - replay_count)

                # if num_batches % args.debug_print_freq == 0 and accelerator.is_local_main_process:
                #     print(f"\n>>> DEBUG STEP {num_batches} <<<")
                #     print(f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
                #     print(f"Selection Ratio: {s_ratio:.2%} (Total: {selected_tokens:.0f}/{total_tokens:.0f})")
                #     print(f"Replay Tokens: {replay_count:.0f} | Hard New Tokens: {hard_new_count:.0f}")
                #     print(f"Threshold: {current_threshold:.4f} | SR Mean: {global_sr_mean:.4f}")
                #     print("-" * 30)

            with torch.no_grad():
                report_mask_3d = batch["loss_mask"][:, :, None].to(predict.dtype)
                _, pred_ids = out_head.max(dim=2)
                _, tgt_ids = target_head.max(dim=2)
                ct = report_mask_3d.sum().item()
                cc = ((pred_ids == tgt_ids) * report_mask_3d.squeeze(-1)).sum().item()

                total += ct
                correct += cc
                epoch_loss += loss.item()
                num_batches += 1

            if accelerator.is_main_process and wandb is not None and ct > 0:
                wandb.log(
                    {
                        "stage2/lr": optimizer_p.param_groups[0]["lr"],
                        "stage2/vloss": float(vloss.item()),
                        "stage2/ploss": float(ploss.item()),
                        "stage2/loss": float(loss.item()),
                        "stage2/acc_step": float(cc / ct),
                        "stage2/grad_norm": float(grad_norm),
                        "stage2/selection_ratio": float(selection_ratio),
                        "stage2/hard_new_tokens_count": float(hard_new_tokens),
                        "stage2/replay_tokens_count": float(replay_tokens),
                        "stage2/global_sr_mean": float(global_sr_mean),
                    }
                )

        epoch_time = time.time() - start_time
        epoch_time_list.append(epoch_time)

        correct_t = accelerator.gather_for_metrics(torch.tensor(correct, device=accelerator.device)).sum().item()
        total_t = accelerator.gather_for_metrics(torch.tensor(total, device=accelerator.device)).sum().item()
        epoch_loss = epoch_loss / max(1, num_batches)
        avg_threshold = epoch_threshold_sum / max(1, num_batches)

        if accelerator.is_local_main_process:
            accelerator.print(
                f"[Stage2] Epoch {epoch + 1}/{args.stage2_epochs} | loss={epoch_loss:.4f} | "
                f"acc={100 * correct_t / max(1, total_t):.2f}% | thr={avg_threshold:.4f} | "
                f"sr_mean={global_sr_mean:.4f} | time={epoch_time:.1f}s"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "stage2/epoch_loss": float(epoch_loss),
                        "stage2/epoch_acc": float(correct_t / max(1, total_t)),
                        "stage2/avg_threshold": float(avg_threshold),
                        "stage2/epoch_time": float(epoch_time),
                    }
                )

        if (epoch + 1) % args.save_freq == 0 and accelerator.is_local_main_process:
            accelerator.save_state(output_dir=os.path.join(args.stage2_outdir, f"state_{epoch + 1}"))

        if (epoch + 1) % args.eval_freq == 0:
            evaluate_stage2(
                args=args,
                accelerator=accelerator,
                model_p=model_p,
                head_p=head_p,
                criterion=criterion,
                test_loader_p=test_loader_p,
                wandb=wandb,
                epoch=epoch,
            )

    if accelerator.is_local_main_process:
        accelerator.save_state(output_dir=os.path.join(args.stage2_outdir, "state_final"))
        with open(os.path.join(args.result_dir, "total_token_result.txt"), "w", encoding="utf-8") as f:
            f.write(str(total_token_list))
        with open(os.path.join(args.result_dir, "selected_token_result.txt"), "w", encoding="utf-8") as f:
            f.write(str(selected_token_list))
        with open(os.path.join(args.result_dir, "usage_time_result.txt"), "w", encoding="utf-8") as f:
            f.write(str(epoch_time_list))



def evaluate_stage2(args, accelerator, model_p, head_p, criterion, test_loader_p, wandb, epoch):
    model_p.eval()
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0.0
    num_batches = 0
    k_acc = [[] for _ in range(5)]

    for batch_idx, data in enumerate(tqdm(test_loader_p, disable=not accelerator.is_local_main_process)):
        with torch.no_grad():
            if batch_idx < 10:
                acces = getkacc(model_p, data, head_p, max_length=5)
                for i in range(len(acces)):
                    k_acc[i].append(acces[i])

            predict = model_forward(model_p, data)
            target_head = head_p(data["target"])
            target_p = torch.softmax(target_head, dim=2).detach()
            out_head = head_p(predict)
            out_logp = nn.LogSoftmax(dim=2)(out_head)
            loss_mask = data["loss_mask"][:, :, None].to(predict.dtype)
            plogp = target_p * out_logp
            ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
            vloss = criterion(predict, data["target"])
            vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
            loss = args.v_w * vloss + args.p_w * ploss

            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze(-1)).sum().item()

            out_head_flat = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target_flat = target.view(-1)[loss_mask.view(-1) == 1]
            if out_head_flat.shape[0] > 0:
                topkacc = top_accuracy(out_head_flat, target_flat, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]

            total += ct
            correct += cc
            epoch_loss += loss.item()
            num_batches += 1

    mean_acces = []
    for i in k_acc:
        mean_acc = np.array(i).mean() if len(i) > 0 else 0.0
        mean_acc = torch.tensor(mean_acc, device=accelerator.device)
        mean_acces.append(mean_acc)

    mean_acces = accelerator.gather_for_metrics(mean_acces)
    if accelerator.is_local_main_process and wandb is not None:
        for idx, acc_tensor in enumerate(mean_acces):
            wandb.log({f"test/{idx}_acc": acc_tensor.mean().item()})

    correct, total = torch.tensor(correct, device=accelerator.device), torch.tensor(total, device=accelerator.device)
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    top_3acc = accelerator.gather_for_metrics(top_3acc)

    if accelerator.is_local_main_process and wandb is not None:
        for idx, acc_tensor in enumerate(top_3acc):
            wandb.log({f"test/top_{idx + 1}_acc": acc_tensor.sum().item() / max(1, total)})

    epoch_loss /= max(1, num_batches)
    if accelerator.is_local_main_process:
        print(f"[Eval] Epoch [{epoch + 1}/{args.stage2_epochs}], Loss: {epoch_loss:.4f}")
        print(f"[Eval] Accuracy: {100 * correct / max(1, total):.2f}%")
        if wandb is not None:
            wandb.log({"test/epochacc": correct / max(1, total), "test/epochloss": epoch_loss})

    model_p.train()


# -----------------------------
# Main
# -----------------------------
def main():
    os.environ.setdefault("WANDB_MODE", "offline")

    parser = argparse.ArgumentParser("train_qwen2.5vl_endoSpec")

    # base/model
    parser.add_argument("--basepath", type=str, default="/media/user/d/xyk/model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--configpath", type=str, default="/media/user/e/xyk/eagle-eye2/eagle-eye2/train/qwenvl2_5_7B_config.json")
    parser.add_argument("--seed", type=int, default=0)

    # runtime
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=2048)

    # data
    parser.add_argument("--stage1_datadir", type=str, default="/media/user/d/xyk/data1/llava_instruct/sample_level_0.25_qwen2.5vl7B")
    parser.add_argument("--stage2_datadir", type=str, default="/media/user/d/xyk/data1/textVQA/qwen2.5vl_distillation/0")
    parser.add_argument("--ordered_files", action="store_true")

    # noise
    parser.add_argument("--data_noise", action="store_true")
    parser.add_argument("--noise_type", type=str, default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--noise_mean", type=float, default=0.0)

    # stage1
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage1_bs", type=int, default=1)
    parser.add_argument("--stage1_lr", type=float, default=3e-5)
    parser.add_argument("--stage1_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--stage1_outdir", type=str, default="/media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/qua_diver_0.25")

    # stage2
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--stage2_bs", type=int, default=1)
    parser.add_argument("--stage2_lr", type=float, default=8e-6)
    parser.add_argument("--stage2_warmup_steps", type=int, default=500)
    parser.add_argument("--stage2_outdir", type=str, default="/media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/TextVQA/Stage2")

    # loss weights / clip
    parser.add_argument("--p_w", type=float, default=0.1)
    parser.add_argument("--v_w", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=0.5)

    # EndoSpec dynamic mask
    parser.add_argument("--teacher_temp", type=float, default=0.5)
    parser.add_argument("--ema_momentum", type=float, default=0.99)
    parser.add_argument("--difficulty_scale", type=float, default=0.8)
    parser.add_argument("--disable_dynamic_mask", action="store_true")

    # save/log/eval
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--debug_print_freq", type=int, default=20)
    parser.add_argument("--result_dir", type=str, default="./result")
    parser.add_argument("--log_project", type=str, default="eagle-eye")
    parser.add_argument("--log_entity", type=str, default=None)

    # stage control
    parser.add_argument("--run_stage1", action="store_true")
    parser.add_argument("--run_stage2", action="store_true")
    parser.add_argument("--only_stage1", action="store_true")
    parser.add_argument("--only_stage2", action="store_true")

    # resume / init
    parser.add_argument("--resume_stage1_state", type=str, default=None)
    parser.add_argument("--resume_stage2_state", type=str, default=None)
    parser.add_argument("--init_stage2_from_stage1_state", type=str, default="/media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/ScienceQA/stage1/state_20")

    args = parser.parse_args()

    if args.only_stage1:
        args.run_stage1, args.run_stage2 = True, False
    elif args.only_stage2:
        args.run_stage1, args.run_stage2 = False, True
    else:
        if not args.run_stage1 and not args.run_stage2:
            args.run_stage1, args.run_stage2 = True, True

    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        try:
            import wandb
            if args.log_entity is None:
                wandb.init(project=args.log_project, config=vars(args))
            else:
                wandb.init(project=args.log_project, entity=args.log_entity, config=vars(args))
        except Exception:
            wandb = None
    else:
        wandb = None

    try:
        from eagle_eye.model.cnets import Model
        from eagle_eye.model.configs import EConfig
    except Exception:
        from model.cnets import Model
        from model.configs import EConfig

    head, _ = build_qwen_lm_head(args.basepath)
    config = EConfig.from_pretrained(args.configpath)
    model = Model(config, load_emb=True, path=args.basepath)
    criterion = nn.SmoothL1Loss(reduction="none")

    aug = None
    if args.data_noise:
        if args.noise_type == "uniform":
            aug = AddUniformNoise(std=args.noise_std)
        else:
            aug = AddGaussianNoise(mean=args.noise_mean, std=args.noise_std)

    file_lister = ordered_list_files if args.ordered_files else normal_list_files

    stage1_train_list = None
    model_p = head_p = optimizer_p = None

    if args.run_stage1:
        stage1_train_list, model_p, head_p, optimizer_p = run_stage1(
            args=args,
            accelerator=accelerator,
            model=model,
            head=head,
            criterion=criterion,
            aug=aug,
            file_lister=file_lister,
            wandb=wandb,
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.95))
        model_p, head_p, optimizer_p = accelerator.prepare(model, head, optimizer)

    if args.run_stage2:
        run_stage2(
            args=args,
            accelerator=accelerator,
            model_p=model_p,
            head_p=head_p,
            optimizer_p=optimizer_p,
            criterion=criterion,
            file_lister=file_lister,
            stage1_train_list=stage1_train_list,
            wandb=wandb,
        )


if __name__ == "__main__":
    main()
