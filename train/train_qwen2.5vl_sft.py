import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

# 项目根目录自动加入 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

os.environ.setdefault("WANDB_MODE", "offline")
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
class QwenHiddenStateDataset(Dataset):
    def __init__(self, file_list: List[str], max_len: int, transform=None):
        self.files = file_list
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        data = torch.load(path, map_location="cpu")

        if not torch.isfinite(data["hidden_state"]).all():
            bad = (~torch.isfinite(data["hidden_state"])).sum().item()
            print(f"[Skip bad data] hidden_state has {bad} NaN/Inf values: {path}")
            return None
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

        input_ids_target = torch.cat([input_ids[:, 1:], torch.tensor([[0]], dtype=input_ids.dtype)], dim=1)
        target = torch.cat(
            [hidden_state[:, 1:, :], torch.zeros(1, 1, hidden_state.shape[2], dtype=hidden_state.dtype)],
            dim=1,
        )

        if inputs_embeds is not None:
            inputs_embeds_target = torch.cat(
                [
                    inputs_embeds[:, 1:, :],
                    torch.zeros(1, 1, inputs_embeds.shape[2], dtype=inputs_embeds.dtype),
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

        if self.transform:
            new = self.transform(new)
        return new


class DataCollatorWithPadding:
    def _pad3d(self, x: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = x.shape
        if n >= N:
            return x[:, :N, :]
        pad = torch.zeros(B, N - n, S, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def _pad2d(self, x: torch.Tensor, N: int) -> torch.Tensor:
        B, n = x.shape
        if n >= N:
            return x[:, :N]
        pad = torch.zeros(B, N - n, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [f for f in features if f is not None]

        if len(features) == 0:
            return None

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
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
        "language_model.model.lm_head.weight",
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "embed_tokens.weight",
        "language_model.embed_tokens.weight",
    ]

    tensor = None
    st_index = os.path.join(basepath, "model.safetensors.index.json")
    pt_index = os.path.join(basepath, "pytorch_model.bin.index.json")
    pt_single = os.path.join(basepath, "pytorch_model.bin")

    if os.path.exists(st_index):
        with open(st_index, "r", encoding="utf-8") as f:
            index_json = json.load(f)

        weight_map = index_json["weight_map"]
        weight_key = next((k for k in candidate_keys if k in weight_map), None)
        if weight_key is None:
            raise KeyError(f"None of candidate keys found: {candidate_keys}")

        head_path = weight_map[weight_key]
        print(f"[LMHead] use key={weight_key}, file={head_path}")

        with safe_open(os.path.join(basepath, head_path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(weight_key).float()

    elif os.path.exists(pt_index):
        with open(pt_index, "r", encoding="utf-8") as f:
            index_json = json.load(f)

        weight_map = index_json["weight_map"]
        weight_key = next((k for k in candidate_keys if k in weight_map), None)
        if weight_key is None:
            raise KeyError(f"None of candidate keys found: {candidate_keys}")

        head_path = weight_map[weight_key]
        print(f"[LMHead] use key={weight_key}, file={head_path}")

        weights = torch.load(os.path.join(basepath, head_path), map_location="cpu")
        tensor = weights[weight_key].float()

    elif os.path.exists(pt_single):
        weights = torch.load(pt_single, map_location="cpu")
        weight_key = next((k for k in candidate_keys if k in weights), None)
        if weight_key is None:
            raise KeyError(f"None of candidate keys found in pytorch_model.bin: {candidate_keys}")

        print(f"[LMHead] use key={weight_key}, file=pytorch_model.bin")
        tensor = weights[weight_key].float()

    else:
        raise FileNotFoundError(f"Cannot find model weights under: {basepath}")

    if tensor.shape != head.weight.shape:
        raise ValueError(f"LM head shape mismatch: tensor={tensor.shape}, expected={head.weight.shape}")

    head.weight.data = tensor
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    return head, baseconfig



def model_forward(model, batch: Dict[str, Any]) -> torch.Tensor:
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    if "inputs_embeds" in batch:
        kwargs["inputs_embeds"] = batch["inputs_embeds"]
    return model(batch["hidden_states"], **kwargs)


# -----------------------------
# 训练主逻辑
# -----------------------------
def run_sft(args, accelerator, model, head, criterion, file_lister, wandb_run):
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    accelerator.print(f"Trainable params count: {len(trainable_params)}")

    all_files = file_lister(args.data_dir)
    split_idx = int(0.95 * len(all_files))
    train_ds = QwenHiddenStateDataset(all_files[:split_idx], max_len=args.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    total_steps = args.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    model, head, optimizer, train_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, scheduler
    )

    if args.init_from_stage1:
        accelerator.print(f"Loading Stage 1 state from: {args.init_from_stage1}")
        accelerator.load_state(args.init_from_stage1)

    for epoch in range(args.epochs):
        model.train()
        epoch_correct = 0
        epoch_total_tokens = 0
        epoch_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
        for batch in pbar:
            if batch is None:
                continue

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                predict = model_forward(model, batch)

                with torch.no_grad():
                    t_head = head(batch["target"].float())
                    t_head = torch.nan_to_num(t_head, nan=0.0, posinf=1e4, neginf=-1e4)
                    target_p = torch.softmax(t_head / args.teacher_temp, dim=-1)
                    target_p = torch.nan_to_num(target_p, nan=0.0, posinf=0.0, neginf=0.0)

                loss_mask_3d = batch["loss_mask"][:, :, None].to(predict.dtype)
                vloss, ploss, out_head = compute_v_p_loss(
                    head=head,
                    criterion=criterion,
                    target_hidden=batch["target"],
                    target_p=target_p,
                    predict_hidden=predict,
                    loss_mask_3d=loss_mask_3d,
                )
                loss = args.v_w * vloss + args.p_w * ploss

                if not torch.isfinite(loss):
                    accelerator.print(
                        f"[NaN DEBUG] loss={loss.item()} "
                        f"vloss={vloss.item()} ploss={ploss.item()} "
                        f"mask_sum={loss_mask_3d.sum().item()} "
                        f"predict_finite={torch.isfinite(predict).all().item()} "
                        f"target_finite={torch.isfinite(batch['target']).all().item()} "
                        f"target_p_finite={torch.isfinite(target_p).all().item()} "
                        f"out_head_finite={torch.isfinite(out_head).all().item()}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                pred_ids = out_head.argmax(dim=-1)
                tgt_ids = t_head.argmax(dim=-1)
                valid_mask = batch["loss_mask"] > 0

                num_correct = (pred_ids[valid_mask] == tgt_ids[valid_mask]).sum().item()
                num_total = valid_mask.sum().item()
                batch_acc = num_correct / num_total if num_total > 0 else 0.0

                epoch_correct += num_correct
                epoch_total_tokens += num_total
                epoch_loss += loss.item()
                steps += 1

                if accelerator.is_main_process and wandb_run is not None:
                    wandb_run.log(
                        {
                            "sft/loss": loss.item(),
                            "sft/lr": optimizer.param_groups[0]["lr"],
                            "sft/acc": batch_acc,
                        }
                    )

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{batch_acc:.2%}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

        epoch_avg_loss = epoch_loss / max(steps, 1)
        epoch_avg_acc = epoch_correct / epoch_total_tokens if epoch_total_tokens > 0 else 0.0
        accelerator.print(
            f"\n[Epoch {epoch}] Final Loss: {epoch_avg_loss:.4f}, Final Acc: {epoch_avg_acc:.2%}"
        )

        if (epoch + 1) % args.save_freq == 0:
            accelerator.wait_for_everyone()
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}")
            accelerator.save_state(save_path)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=str, default="/media/user/d/xyk/model/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--configpath",
        type=str,
        default="/media/user/e/xyk/eagle-eye2/eagle-eye2/train/qwenvl2_5_3B_config.json",
    )
    parser.add_argument("--data_dir", type=str, default="/media/user/e/xyk/data/TextVQA/qwen2.5vl_3B_distill/0")
    parser.add_argument("--output_dir", type=str, default="/media/user/d/xyk/model/Eagle_qwen3B_draft/distill/TextVQA/sft")
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--v_w", type=float, default=1.0)
    parser.add_argument("--p_w", type=float, default=0.1)
    parser.add_argument("--teacher_temp", type=float, default=1.0)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--init_from_stage1", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ordered_files", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    wandb_run = None
    if accelerator.is_main_process:
        try:
            import wandb

            wandb.init(project="qwen-sft", config=vars(args))
            wandb_run = wandb
        except Exception as e:
            accelerator.print(f"WandB init failed: {e}")

    try:
        from eagle_eye.model.cnets import Model
        from eagle_eye.model.configs import EConfig
    except ImportError:
        from model.cnets import Model
        from model.configs import EConfig

    head, _ = build_qwen_lm_head(args.basepath)
    config = EConfig.from_pretrained(args.configpath)
    model = Model(config, load_emb=True, path=args.basepath)
    model.train()

    criterion = nn.SmoothL1Loss(reduction="none")
    file_lister = ordered_list_files if args.ordered_files else normal_list_files

    run_sft(args, accelerator, model, head, criterion, file_lister, wandb_run)


if __name__ == "__main__":
    main()
