import argparse
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

# 项目根目录自动加入 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------
# 基础工具函数
# -----------------------------
def normal_list_files(path: str) -> List[str]:
    datapath = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".ckpt"):
                datapath.append(os.path.join(root, file))
    return datapath

def ordered_list_files(path: str, pattern: str = r"data_(\d+)\.ckpt$") -> List[str]:
    rgx = re.compile(pattern)
    files_with_numbers = []
    for root, _, files in os.walk(path):
        for file in files:
            m = rgx.match(file)
            if not m:
                continue
            files_with_numbers.append((int(m.group(1)), os.path.join(root, file)))
    files_with_numbers.sort(key=lambda x: x[0])
    return [fp for _, fp in files_with_numbers]

# -----------------------------
# 数据集与 Collator
# -----------------------------
class LlavaHiddenStateDataset(Dataset):
    def __init__(self, file_list: List[str], max_len: int, transform=None):
        self.files = file_list
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = torch.load(self.files[index], map_location="cpu")
        new = {}

        hidden_state = data["hidden_state"][: self.max_len][None, :]  
        input_ids = data["input_ids"][: self.max_len][None, :]        

        lm = data["loss_mask"][: self.max_len][None, :]
        loss_mask = lm[0].tolist()
        if len(loss_mask) > 0:
            loss_mask[-1] = 0  # eos 不参与损失

        inputs_embeds = data.get("inputs_embeds", None)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[: self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length

        # Target 构造
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

        if self.transform:
            new = self.transform(new)
        return new

class DataCollatorWithPadding:
    def _pad3d(self, x: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = x.shape
        if n >= N: return x[:, :N, :]
        pad = torch.zeros(B, N - n, S, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)

    def _pad2d(self, x: torch.Tensor, N: int) -> torch.Tensor:
        B, n = x.shape
        if n >= N: return x[:, :N]
        pad = torch.zeros(B, N - n, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(item["hidden_state_big"].shape[1] for item in features)
        
        batch_input_ids = torch.cat([self._pad2d(item["input_ids"], max_len) for item in features], dim=0)
        batch_hidden_states = torch.cat([self._pad3d(item["hidden_state_big"], max_len) for item in features], dim=0)
        batch_target = torch.cat([self._pad3d(item["target"], max_len) for item in features], dim=0)

        embeds_list = []
        has_embeds = False
        for item in features:
            emb = item.get("inputs_embeds")
            if emb is not None:
                embeds_list.append(self._pad3d(emb, max_len))
                has_embeds = True
            else:
                embeds_list.append(torch.zeros_like(item["hidden_state_big"]))
        
        batch_loss_mask = torch.tensor(
            [item["loss_mask"] + [0] * (max_len - len(item["loss_mask"])) for item in features],
            dtype=torch.long,
        )
        batch_attention_mask = torch.tensor(
            [item["attention_mask"] + [0] * (max_len - len(item["attention_mask"])) for item in features],
            dtype=torch.long,
        )

        out = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        if has_embeds:
            out["inputs_embeds"] = torch.cat(embeds_list, dim=0)
        return out

# -----------------------------
# 损失与模型构建
# -----------------------------
def compute_v_p_loss(*, head, criterion, target_hidden, target_p, predict_hidden, loss_mask_3d):
    out_head = head(predict_hidden)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask_3d * plogp, dim=2)) / (loss_mask_3d.sum() + 1e-5)

    vloss_raw = criterion(predict_hidden, target_hidden)
    vloss = torch.sum(torch.mean(loss_mask_3d * vloss_raw, dim=2)) / (loss_mask_3d.sum() + 1e-5)
    return vloss, ploss, out_head

def build_llava_lm_head(basepath: str) -> Tuple[nn.Linear, AutoConfig]:
    baseconfig = AutoConfig.from_pretrained(basepath)
    
    if hasattr(baseconfig, "text_config"):
        hidden_size = getattr(baseconfig.text_config, "hidden_size", None)
    else:
        hidden_size = getattr(baseconfig, "hidden_size", None)

    if hidden_size is None:
        hidden_size = getattr(baseconfig, "vision_config", {}).get("hidden_size", 4096) 
        if not isinstance(hidden_size, int):
             hidden_size = 4096 
             print("Warning: Could not find hidden_size, using 4096")

    vocab_size = getattr(baseconfig, "vocab_size")
    head = nn.Linear(hidden_size, vocab_size, bias=False)

    weight_key = "language_model.lm_head.weight"
    idx_path = os.path.join(basepath, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            index_json = json.load(f)
        head_path = index_json["weight_map"][weight_key]
        with safe_open(os.path.join(basepath, head_path), framework="pt", device="cpu") as f:
            tensor = f.get_slice(weight_key)[:, :hidden_size].float()
    else:
        weights = torch.load(os.path.join(basepath, "pytorch_model.bin"), map_location="cpu")
        tensor = weights[weight_key].float()

    head.weight.data = tensor
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head, baseconfig

def model_forward(model, batch: Dict[str, Any]) -> torch.Tensor:
    kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    if "inputs_embeds" in batch:
        kwargs["inputs_embeds"] = batch["inputs_embeds"]
    return model(batch["hidden_states"], **kwargs)

# -----------------------------
# 训练主逻辑
# -----------------------------
def run_sft(args, accelerator, model, head, criterion, file_lister, wandb):
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    accelerator.print(f"Trainable params count: {len(trainable_params)}")

    all_files = file_lister(args.data_dir)
    split_idx = int(0.95 * len(all_files))
    train_ds = LlavaHiddenStateDataset(all_files[:split_idx], max_len=args.max_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=DataCollatorWithPadding(), num_workers=args.num_workers, pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    
    if args.init_from_stage1:
        accelerator.print(f"Loading Stage 1: {args.init_from_stage1}")
        accelerator.load_state(args.init_from_stage1)

    total_steps = args.epochs * (len(train_loader) // args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    model, head, optimizer, train_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, scheduler
    )

    for epoch in range(args.epochs):
        model.train()
        # 用于记录整个 Epoch 的平均值
        epoch_correct = 0
        epoch_total_tokens = 0
        epoch_loss = 0
        steps = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
        for batch in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                predict = model_forward(model, batch)

                with torch.no_grad():
                    t_head = head(batch["target"])
                    target_p = torch.softmax(t_head / args.teacher_temp, dim=-1)

                loss_mask_3d = batch["loss_mask"][:, :, None].to(predict.dtype)
                vloss, ploss, out_head = compute_v_p_loss(
                    head=head, criterion=criterion, target_hidden=batch["target"],
                    target_p=target_p, predict_hidden=predict, loss_mask_3d=loss_mask_3d
                )
                loss = args.v_w * vloss + args.p_w * ploss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                scheduler.step()

            # --- 计算 Accuracy ---
            with torch.no_grad():
                pred_ids = out_head.argmax(dim=-1)
                tgt_ids = t_head.argmax(dim=-1)
                valid_mask = batch["loss_mask"] > 0
                
                # 当前 Batch 统计
                num_correct = (pred_ids[valid_mask] == tgt_ids[valid_mask]).sum().item()
                num_total = valid_mask.sum().item()
                batch_acc = num_correct / num_total if num_total > 0 else 0
                
                # 累加到 Epoch 统计
                epoch_correct += num_correct
                epoch_total_tokens += num_total
                epoch_loss += loss.item()
                steps += 1
                
                if accelerator.is_main_process and wandb:
                    wandb.log({
                        "sft/loss": loss.item(),
                        "sft/lr": optimizer.param_groups[0]["lr"],
                        "sft/acc": batch_acc
                    })
                
                # 进度条显示当前 Batch 的 Accuracy
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "acc": f"{batch_acc:.2%}", 
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        # --- Epoch 结束打印 ---
        epoch_avg_loss = epoch_loss / steps
        epoch_avg_acc = epoch_correct / epoch_total_tokens if epoch_total_tokens > 0 else 0
        accelerator.print(f"\n[Epoch {epoch}] Final Loss: {epoch_avg_loss:.4f}, Final Acc: {epoch_avg_acc:.2%}")

        if (epoch + 1) % args.save_freq == 0:
            accelerator.wait_for_everyone()
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
            accelerator.save_state(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=str, default="/media/user/d/xyk/model/llava-1.5-7b-hf")
    parser.add_argument("--configpath", type=str, default="/media/user/e/xyk/eagle-eye2/eagle-eye2/train/llava_7B_config.json")
    parser.add_argument("--data_dir", type=str, default="/media/user/d/xyk/data1/A-OKVQA/llava_distillation/0")
    parser.add_argument("--output_dir", type=str, default="/media/user/d/xyk/model/Eagle_llava_draft/endoSpec_distill/OKVQA/SFT")
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--v_w", type=float, default=1.0)
    parser.add_argument("--p_w", type=float, default=0.1)
    parser.add_argument("--teacher_temp", type=float, default=1.0)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--init_from_stage1", type=str, default="/media/user/d/xyk/model/Eagle_llava_draft/endoSpec_distill/ScienceQA/Stage1/epoch_20")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ordered_files", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    wandb = None
    if accelerator.is_main_process:
        try:
            import wandb
            wandb.init(project="eagle-sft", config=args)
        except: pass

    try:
        from eagle_eye.model.cnets import Model
        from eagle_eye.model.configs import EConfig
    except ImportError:
        from model.cnets import Model
        from model.configs import EConfig

    head, _ = build_llava_lm_head(args.basepath)
    config = EConfig.from_pretrained(args.configpath)
    model = Model(config, load_emb=True, path=args.basepath)
    model.train()
    
    criterion = nn.SmoothL1Loss(reduction="none")
    file_lister = ordered_list_files if args.ordered_files else normal_list_files

    run_sft(args, accelerator, model, head, criterion, file_lister, wandb)

if __name__ == "__main__":
    main()