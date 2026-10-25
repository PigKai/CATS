# train_llava_dndoSpec.py (improved)
import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import collections
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/media/user/d/xyk/model/llava-1.5-7b-hf")
# project root: .../eagle-eye2/eagle-eye2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

torch.backends.cuda.matmul.allow_tf32 = True


# -----------------------------
# File listing
# -----------------------------
def normal_list_files(path: str) -> List[str]:
    """List all files under path (recursive), no ordering guarantee."""
    datapath: List[str] = []
    for root, _, files in os.walk(path):
        for file in files:
            datapath.append(os.path.join(root, file))
    return datapath


def ordered_list_files(path: str, pattern: str = r"data_(\d+)\.ckpt$") -> List[str]:
    """List files matching data_{id}.ckpt ordered by id."""
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
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        sample["hidden_state_big"] = tensor + noise
        return sample


# -----------------------------
# Dataset / Collator
# -----------------------------
class LlavaHiddenStateDataset(Dataset):
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

        hidden_state = data["hidden_state"][: self.max_len][None, :]  # (1, L, H)
        input_ids = data["input_ids"][: self.max_len][None, :]        # (1, L)

        if self.cache_loss_mask and index in self.mask_cache:
            loss_mask = self.mask_cache[index]
        else:
            lm = data["loss_mask"][: self.max_len][None, :]
            loss_mask = lm[0].tolist()
            loss_mask[-1] = 0  # eos
            if self.cache_loss_mask:
                self.mask_cache[index] = loss_mask

        inputs_embeds = None
        if "inputs_embeds" in data:
            inputs_embeds = data["inputs_embeds"][: self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length

        # shift to build target (next hidden)
        input_ids_target = torch.cat([input_ids[:, 1:], torch.tensor([[0]])], dim=1)
        target = torch.cat([hidden_state[:, 1:, :], torch.zeros(1, 1, hidden_state.shape[2])], dim=1)

        if inputs_embeds is not None:
            inputs_embeds_target = torch.cat(
                [inputs_embeds[:, 1:, :], torch.zeros(1, 1, inputs_embeds.shape[2])], dim=1
            )
        else:
            inputs_embeds_target = None

        new["attention_mask"] = attention_mask
        new["loss_mask"] = loss_mask
        new["target"] = target
        new["hidden_state_big"] = hidden_state
        new["input_ids"] = input_ids_target
        new["inputs_embeds"] = inputs_embeds_target
        new["index"] = index  # stage2 replay boundary判断用

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


def compute_v_p_loss(
    *,
    head: nn.Module,
    criterion: nn.Module,
    target_hidden: torch.Tensor,
    target_p: torch.Tensor,
    predict_hidden: torch.Tensor,
    loss_mask_3d: torch.Tensor,
):
    out_head = head(predict_hidden)  # (B, L, V)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask_3d * plogp, dim=2)) / (loss_mask_3d.sum() + 1e-5)

    vloss = criterion(predict_hidden, target_hidden)
    vloss = torch.sum(torch.mean(loss_mask_3d * vloss, dim=2)) / (loss_mask_3d.sum() + 1e-5)
    return vloss, ploss, out_head


# -----------------------------
# LLaVA LM head builder
# -----------------------------
def build_llava_lm_head(basepath: str) -> Tuple[nn.Linear, AutoConfig]:
    baseconfig = AutoConfig.from_pretrained(basepath)

    if hasattr(baseconfig, "text_config") and hasattr(baseconfig.text_config, "hidden_size"):
        hidden_size = baseconfig.text_config.hidden_size
    else:
        hidden_size = getattr(baseconfig, "hidden_size")

    vocab_size = getattr(baseconfig, "vocab_size")
    head = nn.Linear(hidden_size, vocab_size, bias=False)

    weight_key = "language_model.lm_head.weight"
    tensor = None

    idx_path = os.path.join(basepath, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            index_json = json.loads(f.read())
        head_path = index_json["weight_map"][weight_key]
        with safe_open(os.path.join(basepath, head_path), framework="pt", device="cpu") as f:
            sl = f.get_slice(weight_key)
            _, hidden_dim2 = sl.get_shape()
            tensor = sl[:, :hidden_dim2].float()
    else:
        idx_path = os.path.join(basepath, "pytorch_model.bin.index.json")
        with open(idx_path, "r") as f:
            index_json = json.loads(f.read())
        head_path = index_json["weight_map"][weight_key]
        weights = torch.load(os.path.join(basepath, head_path), map_location="cpu")
        tensor = weights[weight_key].float()

    head.weight.data = tensor
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head, baseconfig


def model_forward(model, batch: Dict[str, Any]) -> torch.Tensor:
    kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
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

    train_ds = LlavaHiddenStateDataset(stage1_train, max_len=args.max_len, transform=aug)
    test_ds = LlavaHiddenStateDataset(stage1_test, max_len=args.max_len, transform=None)

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

    # prepare once, and reuse optimizer state for stage2
    model_p, head_p, optimizer_p, train_loader_p, test_loader_p, scheduler_p = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )

    if args.resume_stage1_state is not None:
        accelerator.load_state(args.resume_stage1_state)
        import re
        match = re.search(r'state_(\d+)', args.resume_stage1_state)
        if match:
            completed_epochs = int(match.group(1))
            accelerator.print(f"Resuming from epoch {completed_epochs}")
            remaining_epochs = args.stage1_epochs - completed_epochs
            if remaining_epochs <= 0:
                accelerator.print("No epochs left to train.")
                return
            # 重新计算步数
            steps_per_epoch = len(train_loader) // accelerator.gradient_accumulation_steps
            start_step = completed_epochs * steps_per_epoch
            total_steps = args.stage1_epochs * steps_per_epoch

            # 修改点：使用 optimizer_p 而不是 optimizer
            scheduler = get_linear_schedule_with_warmup(
                optimizer_p,  # 改为已包装的优化器
                num_warmup_steps=int(args.stage1_warmup_ratio * total_steps),
                num_training_steps=total_steps,
            )
            # 将调度器步进到当前步数
            for _ in range(start_step):
                scheduler.step()
            # 重新 prepare 调度器
            scheduler_p = accelerator.prepare(scheduler)

    for epoch in range(args.stage1_epochs):
        model_p.train()

        correct = 0.0
        total = 0.0
        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader_p, disable=not accelerator.is_local_main_process):
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
                wandb.log(
                    {
                        "stage1/lr": optimizer_p.optimizer.param_groups[0]["lr"],
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
                f"[Stage1] Epoch {epoch+1}/{args.stage1_epochs} | loss={epoch_loss:.4f} | "
                f"acc={100*correct_t/max(1,total_t):.2f}%"
            )
            if wandb is not None:
                wandb.log({"stage1/epoch_loss": float(epoch_loss), "stage1/epoch_acc": float(correct_t / max(1, total_t))})

        if (epoch + 1) % args.save_freq == 0 and accelerator.is_local_main_process:
            accelerator.save_state(output_dir=os.path.join(args.stage1_outdir, f"state_{epoch+1}"))

    if accelerator.is_local_main_process:
        accelerator.save_state(output_dir=os.path.join(args.stage1_outdir, "state_final"))

    # return prepared objects for stage2 reuse
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

    # ---- data2 always from stage2_datadir ----
    data2_files = file_lister(args.stage2_datadir)
    if len(data2_files) == 0:
        raise RuntimeError(f"[Stage2] No ckpt files found under: {args.stage2_datadir}")

    data2_train = data2_files[: int(0.95 * len(data2_files))]
    data2_test = data2_files[int(0.95 * len(data2_files)) :]

    # ---- data1_train must be identical to Stage1 train split whenever possible ----
    if stage1_train_list is not None:
        data1_train = stage1_train_list
    else:
        # only_stage2 mode fallback: scan data1, but for strict reproducibility use --ordered_files
        data1_files = file_lister(args.stage1_datadir)
        if len(data1_files) == 0:
            raise RuntimeError(f"[Stage2] No ckpt files found under: {args.stage1_datadir} (Stage2 replay needs data1)")
        data1_train = data1_files[: int(0.95 * len(data1_files))]

    # 这里只取和stage2等量的stage1的数据
    if len(data1_train) > len(data2_train):
        accelerator.print(f"[Stage2] Truncating Stage1 replay data from {len(data1_train)} to {len(data2_train)}")
        data1_train = data1_train[:len(data2_train)]

    combined_train = data1_train + data2_train
    distill_boundary_idx = len(data1_train)
    train_ds = LlavaHiddenStateDataset(combined_train, max_len=args.max_len, transform=None, cache_loss_mask=False)
    test_ds = LlavaHiddenStateDataset(data2_test, max_len=args.max_len, transform=None, cache_loss_mask=False)

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

    # IMPORTANT: model/head/optimizer are already prepared in stage1, do NOT prepare them again
    train_loader_p, test_loader_p, scheduler2_p = accelerator.prepare(train_loader, test_loader, scheduler2)

    # resume / init must be after stage2 objects are ready
    if args.resume_stage2_state is not None:
        accelerator.print(f"[Stage2] Resuming from: {args.resume_stage2_state}")
        accelerator.load_state(args.resume_stage2_state)
    else:
        if args.init_stage2_from_stage1_state is not None:
            accelerator.print(f"[Stage2] Initializing from stage1 state: {args.init_stage2_from_stage1_state}")
            accelerator.load_state(args.init_stage2_from_stage1_state)
    # A. 强制重置优化器内部的 step 计数（防止它告诉 scheduler 训练已结束）
    for group in optimizer_p.param_groups:
        # 重置某些优化器（如 Adam/AdamW）内部维护的 step 状态
        for p in group['params']:
            state = optimizer_p.state[p]
            if len(state) > 0:
                state['step'] = torch.tensor(0.0, device=p.device)

    # B. 重新创建全新的 Scheduler 对象
    # 这里的 total_steps 已经在前面计算好了
    scheduler2 = get_linear_schedule_with_warmup(
        optimizer_p,
        num_warmup_steps=min(args.stage2_warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    # C. 重新交给 accelerator 准备
    # 注意：这里只 prepare scheduler 即可，loader 和 model 之前已经 prepare 过了
    scheduler2_p = accelerator.prepare(scheduler2)
    
    accelerator.print(f"[Stage2] Scheduler has been reset. Warmup steps: {args.stage2_warmup_steps}, Total steps: {total_steps}")
    for pg in optimizer_p.param_groups:
            pg["lr"] = args.stage2_lr
    # ---- EndoSpec online dynamic mask ----
    global_sr_mean = 1.0
    total_token_list: List[float] = []
    selected_token_list: List[float] = []
    epoch_time_list: List[float] = []

    for epoch in range(args.stage2_epochs):
        model_p.train()
        start_time = time.time()

        correct = 0.0
        total = 0.0
        epoch_loss = 0.0
        num_batches = 0
        epoch_threshold_sum = 0.0

        for batch in tqdm(train_loader_p, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model_p):
                optimizer_p.zero_grad(set_to_none=True)
                predict = model_forward(model_p, batch)

                with torch.no_grad():
                    target_head = head_p(batch["target"])
                    target_p = torch.softmax(target_head / args.teacher_temp, dim=2).detach()

                    out_head_for_sr = head_p(predict)
                    probs_student = torch.softmax(out_head_for_sr, dim=-1)

                    true_token_ids = torch.argmax(target_p, dim=-1, keepdim=True)
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

                    current_threshold = global_sr_mean * args.difficulty_scale
                    epoch_threshold_sum += current_threshold

                    if args.disable_dynamic_mask:
                        dynamic_mask = raw_loss_mask.clone()
                    else:
                        is_hard_token = surprise_ratio < current_threshold

                        batch_indices = batch["index"]  # index in train_ds
                        is_replay_sample = torch.tensor(
                            [idx < distill_boundary_idx for idx in batch_indices],
                            device=raw_loss_mask.device,
                            dtype=torch.bool,
                        ).unsqueeze(1)

                        final_select_mask = is_hard_token | is_replay_sample
                        dynamic_mask = raw_loss_mask * final_select_mask.long()
                        with torch.no_grad():
                            # 计算原始需要计算的 token 总数
                            total_active_tokens = raw_loss_mask.sum().item()
                            # 计算经过动态掩码过滤后剩下的 token 总数
                            selected_active_tokens = dynamic_mask.sum().item()
                            # 计算选中比例
                            selection_ratio = selected_active_tokens / (total_active_tokens + 1e-8)
                            
                            # 区分 Stage1 回放和 Stage2 新数据的选中情况
                            replay_tokens = (raw_loss_mask * is_replay_sample.long()).sum().item()
                            hard_new_tokens = (raw_loss_mask * (~is_replay_sample).long() * is_hard_token.long()).sum().item()

                    for i in range(dynamic_mask.shape[0]):
                        selected_token_list.append(float(dynamic_mask[i].sum().item()))

                    loss_mask_3d = dynamic_mask[:, :, None].to(predict.dtype)

                vloss, ploss, out_head = compute_v_p_loss(
                    head=head_p,
                    criterion=criterion,
                    target_hidden=batch["target"],
                    target_p=target_p,
                    predict_hidden=predict,
                    loss_mask_3d=loss_mask_3d,
                )
                loss = args.v_w * vloss + args.p_w * ploss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # 获取模型所有参数的梯度范数
                    grad_norm = 0.0
                    for p in model_p.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                accelerator.clip_grad_value_(model_p.parameters(), args.grad_clip)
                optimizer_p.step()
                # optimizer_p.step()
                # 1. 强行获取当前真实的 Learning Rate
                current_lr = optimizer_p.param_groups[0]['lr']

                # 2. 计算这个 Batch 的掩码情况
                with torch.no_grad():
                    # 总共需要计算的 token 数
                    total_tokens = raw_loss_mask.sum().item()
                    # 最终选中的 token 数
                    selected_tokens = dynamic_mask.sum().item()
                    # 选中率
                    s_ratio = selected_tokens / (total_tokens + 1e-8)
                    
                    # 分辨一下：选中样本里有多少是旧数据（Replay）
                    replay_count = (raw_loss_mask * is_replay_sample.long()).sum().item()
                    # 真正的新数据难样本
                    hard_new_count = (dynamic_mask - (raw_loss_mask * is_replay_sample.long())).sum().clamp(min=0).sum().item()

                # 3. 每 20 个 Step 直接在屏幕打印（不需要 WandB）
                # if num_batches % 20 == 0:
                #     print(f"\n>>> DEBUG STEP {num_batches} <<<")
                #     print(f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
                #     print(f"Selection Ratio: {s_ratio:.2%} (Total: {selected_tokens:.0f}/{total_tokens:.0f})")
                #     print(f"Replay Tokens: {replay_count:.0f} | Hard New Tokens: {hard_new_count:.0f}")
                #     print(f"Threshold: {current_threshold:.4f} | SR Mean: {global_sr_mean:.4f}")
                #     print("-" * 30)
                if num_batches % 20 == 0:
                    print(f"\n>>> DEBUG STEP {num_batches} <<<")
                    print(f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
                    print(f"Selection Ratio: {s_ratio:.2%} (Total: {selected_tokens:.0f}/{total_tokens:.0f})")
                    print(f"Replay Tokens: {replay_count:.0f} | Hard New Tokens: {hard_new_count:.0f}")
                    print(f"Threshold: {current_threshold:.4f} | SR Mean: {global_sr_mean:.4f}")
                    print("-" * 30)

                    # -----------------------------
                    # Token 级别详细信息
                    # -----------------------------
                    batch_token_count = raw_loss_mask.shape[1]  # L
                    input_ids = batch["input_ids"][0]           # shape: (L,)

                    # 1️⃣ 先找出所有参与 loss 的 Hard token
                    hard_positions = []
                    for i in range(batch_token_count):
                        # 只考虑最终被选中参与 loss 的 token
                        if dynamic_mask[0, i].item() == 0:
                            continue
                        # 只考虑 Hard token
                        if not is_hard_token[0, i].item():
                            continue
                        hard_positions.append(i)

                    # 2️⃣ 打印 Hard token 详细信息
                    print("\n>>> Hard tokens info <<<")
                    for i in hard_positions:
                        sr_value = float(surprise_ratio[0, i].item())
                        is_replay = is_replay_sample[0].item() if not args.disable_dynamic_mask else False
                        token_id = input_ids[i].item()
                        token_str = tokenizer.decode([token_id]).strip()
                        print(
                            f"Token {i:03d} | ID={token_id} | Word='{token_str}' | "
                            f"SR={sr_value:.4f} | Replay={is_replay} | Selected=True"
                        )

                    # 3️⃣ 打印完整对话文本
                    full_input_ids_list = input_ids.tolist()
                    full_text = tokenizer.decode(full_input_ids_list, skip_special_tokens=False).replace("\n", " ")
                    print("\nFull sequence / dialogue:")
                    print(full_text)

                    # 4️⃣ 打印对话中标记 Hard token
                    highlighted_text = []
                    for i, tid in enumerate(full_input_ids_list):
                        word = tokenizer.decode([tid]).strip()
                        if i in hard_positions:
                            word = f"[{word}]"   # 用 [] 标记 Hard token
                        highlighted_text.append(word)

                    print("\nDialogue with Hard tokens highlighted:")
                    print(" ".join(highlighted_text))
                    print("=" * 60)
                scheduler2_p.step()

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
                        "stage2/lr": optimizer_p.param_groups[0]["lr"], # 确保拿到的是真实 LR
                        "stage2/vloss": float(vloss.item()),
                        "stage2/ploss": float(ploss.item()),
                        "stage2/loss": float(loss.item()),
                        "stage2/acc_step": float(cc / ct),
                        "stage2/grad_norm": float(grad_norm), # 观察梯度强度
                        "stage2/selection_ratio": float(selection_ratio), # 观察掩码是否过严
                        "stage2/hard_new_tokens_count": float(hard_new_tokens), # 观察新数据贡献
                        "stage2/replay_tokens_count": float(replay_tokens), # 观察旧数据贡献
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
                f"[Stage2] Epoch {epoch+1}/{args.stage2_epochs} | loss={epoch_loss:.4f} | "
                f"acc={100*correct_t/max(1,total_t):.2f}% | thr={avg_threshold:.4f} | "
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
            accelerator.save_state(output_dir=os.path.join(args.stage2_outdir, f"state_{epoch+1}"))

    if accelerator.is_local_main_process:
        accelerator.save_state(output_dir=os.path.join(args.stage2_outdir, "state_final"))
        with open(os.path.join(args.result_dir, "total_token_result.txt"), "w", encoding="utf-8") as f:
            f.write(str(total_token_list))
        with open(os.path.join(args.result_dir, "selected_token_result.txt"), "w", encoding="utf-8") as f:
            f.write(str(selected_token_list))
        with open(os.path.join(args.result_dir, "usage_time_result.txt"), "w", encoding="utf-8") as f:
            f.write(str(epoch_time_list))


# -----------------------------
# Main
# -----------------------------
def main():
    os.environ.setdefault("WANDB_MODE", "offline")

    parser = argparse.ArgumentParser("train_llava_dndoSpec_two_stage_with_resume")

    # base/model
    parser.add_argument("--basepath", type=str, default="/media/user/d/xyk/model/llava-1.5-7b-hf")
    parser.add_argument("--configpath", type=str, default="/media/user/e/xyk/eagle-eye2/eagle-eye2/train/llava_7B_config.json")
    parser.add_argument("--seed", type=int, default=0)
    
    # runtime
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=2048)

    # data
    parser.add_argument("--stage1_datadir", type=str, default="/media/user/d/xyk/data1/llava_instruct/get_data_llava/llava_instruct_150k_0_67999/0", help="data1: distill/general ckpt dir")
    parser.add_argument("--stage2_datadir", type=str, default="/media/user/d/xyk/data1/textVQA/0", help="data2: domain/selection ckpt dir")
    parser.add_argument("--ordered_files", action="store_true",
                        help="If set, list files as data_{id}.ckpt ordered; otherwise random walk.")

    # noise
    parser.add_argument("--data_noise", action="store_true")
    parser.add_argument("--noise_type", type=str, default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--noise_mean", type=float, default=0.0)

    # stage1
    parser.add_argument("--stage1_epochs", type=int, default=25)
    parser.add_argument("--stage1_bs", type=int, default=1)
    parser.add_argument("--stage1_lr", type=float, default=3e-5)
    parser.add_argument("--stage1_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--stage1_outdir", type=str, default="/media/user/d/xyk/model/llava_baseline")

    # stage2
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_bs", type=int, default=1)
    parser.add_argument("--stage2_lr", type=float, default=8e-6)
    parser.add_argument("--stage2_warmup_steps", type=int, default=500)
    parser.add_argument("--stage2_outdir", type=str, default="/media/user/d/xyk/model/Eagle_llava_draft/endoSpec_distill/TextVQA/Stage2")

    # loss weights / clip
    parser.add_argument("--p_w", type=float, default=0.1)
    parser.add_argument("--v_w", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=0.5)

    # EndoSpec dynamic mask
    parser.add_argument("--teacher_temp", type=float, default=0.5)
    parser.add_argument("--ema_momentum", type=float, default=0.99)
    parser.add_argument("--difficulty_scale", type=float, default=0.8)
    parser.add_argument("--disable_dynamic_mask", action="store_true")

    # save/log
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--result_dir", type=str, default="./result")
    parser.add_argument("--log_project", type=str, default="eagle-eye")
    parser.add_argument("--log_entity", type=str, default=None)

    # stage control
    parser.add_argument("--run_stage1", action="store_true")
    parser.add_argument("--run_stage2", action="store_true")
    parser.add_argument("--only_stage1", action="store_true")
    parser.add_argument("--only_stage2", action="store_true")

    # resume
    parser.add_argument("--resume_stage1_state", type=str, default=None)
    parser.add_argument("--resume_stage2_state", type=str, default=None)

    # init stage2
    parser.add_argument("--init_stage2_from_stage1_state", type=str, default=None)

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

    # wandb
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

    # import your model + config
    try:
        from eagle_eye.model.cnets import Model
        from eagle_eye.model.configs import EConfig
    except Exception:
        from model.cnets import Model
        from model.configs import EConfig

    # head
    head, _ = build_llava_lm_head(args.basepath)

    # model
    config = EConfig.from_pretrained(args.configpath)
    model = Model(config, load_emb=True, path=args.basepath)

    criterion = nn.SmoothL1Loss(reduction="none")

    # augmentation
    aug = None
    if args.data_noise:
        aug = AddUniformNoise(std=args.noise_std) if args.noise_type == "uniform" else AddGaussianNoise(
            mean=args.noise_mean, std=args.noise_std
        )

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
        # only_stage2 path: we still need prepared model/head/optimizer for stage2.
        # create optimizer with stage1_lr (or stage2_lr; state will be overwritten if init_stage2_from_stage1_state is used)
        optimizer = optim.AdamW(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.95))
        # prepare model/head/optimizer without loaders (safe)
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