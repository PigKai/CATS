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
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

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

      - visual_static_mask: (seq,)

      - visual_dep_score: (seq,)

      - custom hidden-state fields for augmentation

    """



    def __init__(

        self,

        file_list: List[str],

        max_len: int,

        transform=None,

        cache_loss_mask: bool = False,

        input_hidden_key: str = "hidden_state",

        target_hidden_key: str = "hidden_state",

    ):

        self.files = file_list

        self.max_len = max_len

        self.transform = transform

        self.cache_loss_mask = cache_loss_mask

        self.mask_cache: Dict[int, List[int]] = {}

        self.input_hidden_key = input_hidden_key

        self.target_hidden_key = target_hidden_key



    def __len__(self) -> int:

        return len(self.files)



    def __getitem__(self, index: int) -> Dict[str, Any]:

        data = torch.load(self.files[index], map_location="cpu")

        new: Dict[str, Any] = {}



        input_hidden_key = self.input_hidden_key

        target_hidden_key = self.target_hidden_key



        if input_hidden_key in data:

            input_hidden_state = data[input_hidden_key][: self.max_len][None, :]

        elif "hidden_state" in data:

            input_hidden_state = data["hidden_state"][: self.max_len][None, :]

        else:

            raise KeyError(

                f"{self.files[index]} missing input hidden key: {input_hidden_key} and fallback hidden_state"

            )



        if target_hidden_key in data:

            target_hidden_state = data[target_hidden_key][: self.max_len][None, :]

        elif target_hidden_key == "vision_aug_hidden_state" and "hidden_state" in data:

            target_hidden_state = data["hidden_state"][: self.max_len][None, :]

        else:

            raise KeyError(f"{self.files[index]} missing target hidden key: {target_hidden_key}")



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



        if "visual_static_mask" in data:

            visual_static_mask = data["visual_static_mask"][: self.max_len]

            if isinstance(visual_static_mask, torch.Tensor):

                visual_static_mask = visual_static_mask.long().view(-1)

            else:

                visual_static_mask = torch.tensor(visual_static_mask, dtype=torch.long).view(-1)

        else:

            visual_static_mask = torch.zeros(input_hidden_state.shape[1], dtype=torch.long)



        if visual_static_mask.shape[0] > 0:

            visual_static_mask[-1] = 0



        # visual_dep_score is pre-computed from the frozen target model.
        # It is expected to be aligned with loss_mask / visual_static_mask.
        if "visual_dep_score" in data:

            visual_dep_score = data["visual_dep_score"][: self.max_len]

            if isinstance(visual_dep_score, torch.Tensor):

                visual_dep_score = visual_dep_score.float().view(-1)

            else:

                visual_dep_score = torch.tensor(visual_dep_score, dtype=torch.float32).view(-1)

        else:

            # Stage-I replay samples may not have visual dependency scores.
            visual_dep_score = torch.zeros(input_hidden_state.shape[1], dtype=torch.float32)



        # Keep all token-level auxiliary arrays aligned with the model sequence.
        seq_len = input_hidden_state.shape[1]

        if visual_static_mask.shape[0] < seq_len:

            visual_static_mask = torch.cat(

                [visual_static_mask, torch.zeros(seq_len - visual_static_mask.shape[0], dtype=torch.long)], dim=0

            )

        else:

            visual_static_mask = visual_static_mask[:seq_len]



        if visual_dep_score.shape[0] < seq_len:

            visual_dep_score = torch.cat(

                [visual_dep_score, torch.zeros(seq_len - visual_dep_score.shape[0], dtype=torch.float32)], dim=0

            )

        else:

            visual_dep_score = visual_dep_score[:seq_len]



        if seq_len > 0:

            visual_static_mask[-1] = 0

            visual_dep_score[-1] = 0.0



        length = input_hidden_state.shape[1]

        attention_mask = [1] * length



        input_ids_target = torch.cat(

            [input_ids[:, 1:], torch.zeros((1, 1), dtype=input_ids.dtype)], dim=1

        )

        target = torch.cat(

            [

                target_hidden_state[:, 1:, :],

                torch.zeros((1, 1, target_hidden_state.shape[2]), dtype=target_hidden_state.dtype),

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

        new["hidden_state_big"] = input_hidden_state

        new["input_ids"] = input_ids_target

        new["inputs_embeds"] = inputs_embeds_target

        new["index"] = index

        new["visual_static_mask"] = visual_static_mask

        new["visual_dep_score"] = visual_dep_score



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

        batch_visual_static_mask = torch.stack(

            [

                torch.cat(

                    [

                        item["visual_static_mask"],

                        torch.zeros(max_len - item["visual_static_mask"].shape[0], dtype=torch.long),

                    ],

                    dim=0,

                )

                for item in features

            ],

            dim=0,

        )

        batch_visual_dep_score = torch.stack(

            [

                torch.cat(

                    [

                        item["visual_dep_score"],

                        torch.zeros(max_len - item["visual_dep_score"].shape[0], dtype=torch.float32),

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

            "visual_static_mask": batch_visual_static_mask,

            "visual_dep_score": batch_visual_dep_score,

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

    out_head = head(predict_hidden.float())

    out_logp = torch.log_softmax(out_head.float(), dim=2)



    target_p = torch.nan_to_num(target_p.float(), nan=0.0, posinf=0.0, neginf=0.0)

    out_logp = torch.nan_to_num(out_logp, nan=0.0, posinf=0.0, neginf=-100.0)



    plogp = target_p * out_logp

    denom = loss_mask_3d.sum().clamp_min(1.0)



    ploss = -torch.sum(torch.sum(loss_mask_3d.float() * plogp, dim=2)) / denom



    vloss = criterion(predict_hidden.float(), target_hidden.float())

    vloss = torch.nan_to_num(vloss, nan=0.0, posinf=1e4, neginf=1e4)

    vloss = torch.sum(torch.mean(loss_mask_3d.float() * vloss, dim=2)) / denom



    return vloss, ploss, out_head



@torch.no_grad()
def compute_acceptance_overlap(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute beta_t = sum_x min(p_t(x), q_t(x)) = 1 - TVD(p_t, q_t).

    Only valid response positions are materialized as full vocabulary probabilities,
    which keeps the temporary memory footprint much smaller than applying softmax
    over every prompt position.  Target and draft use the SAME temperature.

    Returns:
        beta: shape (B, S). Non-response positions are set to 1.0 and are ignored
              by the gain computation.
    """
    if temperature <= 0:
        raise ValueError(f"gain temperature must be > 0, got {temperature}")
    if target_logits.shape != draft_logits.shape:
        raise ValueError(
            f"target_logits and draft_logits must have identical shape, got "
            f"{target_logits.shape} vs {draft_logits.shape}"
        )
    if valid_mask.shape != target_logits.shape[:2]:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} does not match logits prefix {target_logits.shape[:2]}"
        )

    B, S, _ = target_logits.shape
    beta = torch.ones((B, S), dtype=torch.float32, device=target_logits.device)

    for i in range(B):
        idx = torch.nonzero(valid_mask[i].bool(), as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue

        t_logits = target_logits[i, idx].float() / temperature
        d_logits = draft_logits[i, idx].float() / temperature

        p = torch.softmax(t_logits, dim=-1)
        q = torch.softmax(d_logits, dim=-1)
        b = torch.minimum(p, q).sum(dim=-1)
        b = torch.nan_to_num(b, nan=eps, posinf=1.0, neginf=eps).clamp(min=eps, max=1.0)
        beta[i, idx] = b

    return beta



@torch.no_grad()
def normalize_visual_dependency(
    visual_dep_score: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-sample normalization of the pre-computed target-side visual dependency.

    CATS defines positive visual dependency as evidence that the image raises the
    target probability of the token. Negative values therefore do not receive a
    visual bonus. Among valid response tokens, positive scores are normalized by
    the sample maximum so the strongest visual token has value 1.
    """
    if visual_dep_score.shape != valid_mask.shape:
        raise ValueError(
            f"visual_dep_score shape {visual_dep_score.shape} != valid_mask shape {valid_mask.shape}"
        )

    out = torch.zeros_like(visual_dep_score, dtype=torch.float32)
    B = visual_dep_score.shape[0]
    for i in range(B):
        idx = torch.nonzero(valid_mask[i].bool(), as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        v = visual_dep_score[i, idx].float()
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = torch.clamp(v, min=0.0)
        vmax = v.max()
        if vmax > eps:
            v = v / vmax
        else:
            v = torch.zeros_like(v)
        out[i, idx] = v
    return out



@torch.no_grad()
def compute_unified_gain_score(
    beta: torch.Tensor,
    visual_dep_score: torch.Tensor,
    valid_mask: torch.Tensor,
    visual_lambda: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a single vision-aware marginal acceptance-gain score G_t.

    For a linear speculative trajectory with per-position acceptance probabilities
    beta_1...beta_L, the expected accepted length is

        E[A] = sum_{k=1}^L prod_{i=1}^k beta_i.

    If only beta_t were perfectly repaired to 1 while all other beta values were
    held fixed, the exact counterfactual gain under this linear approximation is

        G_accept_t = R_t * (1 - beta_t) * C_t,
        R_t = prod_{i<t} beta_i,
        C_t = 1 + beta_{t+1} C_{t+1}.

    We fold the pre-computed multimodal dependency into the SAME final ranking
    indicator rather than OR-ing a separate visual mask:

        G_t = G_accept_t * (1 + visual_lambda * v_hat_t).

    Ranking is performed in log-space for numerical stability.

    Returns:
        log_gain:      (B, S), -inf outside valid response positions.
        beta:          sanitized beta tensor.
        reachability:  R_t at valid positions.
        continuation:  C_t at valid positions.
        visual_norm:   v_hat_t in [0, 1].
    """
    if visual_lambda < 0:
        raise ValueError(f"visual_lambda must be >= 0, got {visual_lambda}")
    if beta.shape != valid_mask.shape:
        raise ValueError(f"beta shape {beta.shape} != valid_mask shape {valid_mask.shape}")

    visual_norm = normalize_visual_dependency(visual_dep_score, valid_mask, eps=eps)

    log_gain = torch.full_like(beta, float("-inf"), dtype=torch.float32)
    reachability = torch.zeros_like(beta, dtype=torch.float32)
    continuation = torch.zeros_like(beta, dtype=torch.float32)

    B = beta.shape[0]
    for i in range(B):
        idx = torch.nonzero(valid_mask[i].bool(), as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n == 0:
            continue

        b = beta[i, idx].float()
        b = torch.nan_to_num(b, nan=eps, posinf=1.0, neginf=eps).clamp(min=eps, max=1.0)

        # log R_t = sum_{i<t} log beta_i.
        log_r = torch.zeros_like(b)
        if n > 1:
            log_r[1:] = torch.cumsum(torch.log(b[:-1].clamp_min(eps)), dim=0)

        # C_t = 1 + beta_{t+1} * C_{t+1}.
        c = torch.ones_like(b)
        for j in range(n - 2, -1, -1):
            c[j] = 1.0 + b[j + 1] * c[j + 1]

        gap = (1.0 - b).clamp_min(eps)
        v = visual_norm[i, idx]
        visual_factor = 1.0 + visual_lambda * v

        # log(G_t) gives exactly the same ordering as G_t but avoids underflow.
        lg = log_r + torch.log(gap) + torch.log(c.clamp_min(eps)) + torch.log(visual_factor.clamp_min(eps))

        log_gain[i, idx] = lg
        reachability[i, idx] = torch.exp(log_r.clamp(min=-80.0, max=0.0))
        continuation[i, idx] = c

    return log_gain, beta, reachability, continuation, visual_norm



@torch.no_grad()
def build_gain_topk_mask(
    log_gain: torch.Tensor,
    raw_loss_mask: torch.Tensor,
    is_replay_sample: torch.Tensor,
    keep_ratio: float,
    min_tokens: int = 1,
    select_on_replay: bool = False,
) -> torch.Tensor:
    """
    Build a per-sample fixed-budget Top-K mask from the single G_t score.

    By default Stage-I replay samples remain dense, matching the original Stage-II
    replay behavior. Set select_on_replay=True only for a dedicated ablation.
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"gain_keep_ratio must be in (0, 1], got {keep_ratio}")
    if min_tokens < 1:
        raise ValueError(f"gain_min_tokens must be >= 1, got {min_tokens}")

    valid_mask = raw_loss_mask.bool()
    dynamic_mask = torch.zeros_like(raw_loss_mask, dtype=torch.long)
    B = raw_loss_mask.shape[0]

    replay_flags = is_replay_sample.squeeze(1).bool()
    for i in range(B):
        idx = torch.nonzero(valid_mask[i], as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n == 0:
            continue

        if replay_flags[i] and not select_on_replay:
            dynamic_mask[i, idx] = 1
            continue

        k = int(np.ceil(n * keep_ratio))
        k = max(min_tokens, k)
        k = min(n, k)

        scores = log_gain[i, idx]
        scores = torch.nan_to_num(scores, nan=-1e30, posinf=1e30, neginf=-1e30)
        top_local = torch.topk(scores, k=k, largest=True, sorted=False).indices
        chosen = idx[top_local]
        dynamic_mask[i, chosen] = 1

    return dynamic_mask



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

            raise KeyError(f"None of {candidate_keys} found in safetensors index.")

        head_path = weight_map[key]

        with safe_open(os.path.join(basepath, head_path), framework="pt", device="cpu") as f:

            sl = f.get_slice(key)

            _, hidden_dim = sl.get_shape()

            tensor = sl[:, :hidden_dim].float()

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



    combined_train = data1_train + data2_train

    distill_boundary_idx = len(data1_train)



    train_ds = QwenHiddenStateDataset(

        combined_train,

        max_len=args.max_len,

        transform=None,

        cache_loss_mask=False,

        input_hidden_key=args.stage2_input_hidden_key,

        target_hidden_key=args.stage2_target_hidden_key,

    )

    test_ds = QwenHiddenStateDataset(

        data2_test,

        max_len=args.max_len,

        transform=None,

        cache_loss_mask=False,

        input_hidden_key=args.stage2_input_hidden_key,

        target_hidden_key=args.stage2_target_hidden_key,

    )



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
        epoch_threshold_count = 0
        epoch_gain_beta_sum = 0.0
        epoch_gain_gap_sum = 0.0
        epoch_gain_log_sum = 0.0
        epoch_gain_visual_sum = 0.0
        epoch_gain_batches = 0



        for step_idx, batch in enumerate(

            tqdm(train_loader_p, disable=not accelerator.is_local_main_process), start=0

        ):

            with accelerator.accumulate(model_p):

                optimizer_p.zero_grad(set_to_none=True)

                predict = model_forward(model_p, batch)



                if not torch.isfinite(predict).all():

                    accelerator.print(

                        f"[SkipBadPredict] step={step_idx} "

                        f"hidden_finite={torch.isfinite(batch['hidden_states']).all().item()} "

                        f"target_finite={torch.isfinite(batch['target']).all().item()} "

                        f"input_ids_min={batch['input_ids'].min().item()} "

                        f"input_ids_max={batch['input_ids'].max().item()} "

                        f"loss_tokens={batch['loss_mask'].sum().item()}"

                    )

                    optimizer_p.zero_grad(set_to_none=True)

                    continue



                with torch.no_grad():
                    # Frozen target distribution used by the original Stage-II distillation loss.
                    target_head = head_p(batch["target"])
                    target_p = torch.softmax(target_head.float() / args.teacher_temp, dim=2).detach()

                    # Current draft logits. These are used for token selection only; gradients
                    # still flow through `predict` in compute_v_p_loss below.
                    draft_head_for_select = head_p(predict).detach()

                    raw_loss_mask = batch["loss_mask"]
                    visual_static_mask = batch["visual_static_mask"]
                    visual_dep_score = batch["visual_dep_score"]
                    batch_indices = batch["index"]

                    is_replay_sample = torch.tensor(
                        [idx < distill_boundary_idx for idx in batch_indices],
                        device=raw_loss_mask.device,
                        dtype=torch.bool,
                    ).unsqueeze(1)

                    for i in range(raw_loss_mask.shape[0]):
                        total_token_list.append(float(raw_loss_mask[i].sum().item()))

                    # Common defaults used by logging/debugging for all selection modes.
                    surprise_ratio = torch.ones_like(raw_loss_mask, dtype=torch.float32)
                    is_hard_token = torch.zeros_like(raw_loss_mask, dtype=torch.bool)
                    current_threshold = float("nan")

                    beta = torch.ones_like(raw_loss_mask, dtype=torch.float32)
                    log_gain = torch.full_like(raw_loss_mask, float("-inf"), dtype=torch.float32)
                    reachability = torch.zeros_like(raw_loss_mask, dtype=torch.float32)
                    continuation = torch.zeros_like(raw_loss_mask, dtype=torch.float32)
                    visual_norm = torch.zeros_like(raw_loss_mask, dtype=torch.float32)

                    gain_beta_mean = float("nan")
                    gain_gap_mean = float("nan")
                    gain_log_mean = float("nan")
                    gain_visual_mean = float("nan")

                    if args.disable_dynamic_mask:
                        # Explicit ablation: dense Stage-II updates on all valid answer tokens.
                        dynamic_mask = raw_loss_mask.clone()

                    elif args.selection_mode == "gain":
                        # 1) Per-token expected acceptance beta_t = 1 - TVD(p_t, q_t).
                        #    Target and draft MUST use the same temperature here.
                        beta = compute_acceptance_overlap(
                            target_logits=target_head.detach(),
                            draft_logits=draft_head_for_select,
                            valid_mask=raw_loss_mask,
                            temperature=args.gain_temp,
                            eps=args.gain_eps,
                        )

                        # 2) One unified score:
                        #    G_t = R_t * (1-beta_t) * C_t * (1 + lambda_v * v_hat_t).
                        #    We rank by log(G_t), which is numerically stable and order-equivalent.
                        log_gain, beta, reachability, continuation, visual_norm = compute_unified_gain_score(
                            beta=beta,
                            visual_dep_score=visual_dep_score,
                            valid_mask=raw_loss_mask,
                            visual_lambda=args.gain_visual_lambda,
                            eps=args.gain_eps,
                        )

                        # 3) Fixed per-sample budget. Replay remains dense by default to preserve
                        #    the original anti-forgetting setup.
                        dynamic_mask = build_gain_topk_mask(
                            log_gain=log_gain,
                            raw_loss_mask=raw_loss_mask,
                            is_replay_sample=is_replay_sample,
                            keep_ratio=args.gain_keep_ratio,
                            min_tokens=args.gain_min_tokens,
                            select_on_replay=args.gain_select_on_replay,
                        )

                        valid_gain_mask = raw_loss_mask.bool() & (~is_replay_sample)
                        if valid_gain_mask.any():
                            finite_gain = torch.isfinite(log_gain) & valid_gain_mask
                            gain_beta_mean = float(beta[valid_gain_mask].mean().item())
                            gain_gap_mean = float((1.0 - beta[valid_gain_mask]).mean().item())
                            gain_visual_mean = float(visual_norm[valid_gain_mask].mean().item())
                            if finite_gain.any():
                                gain_log_mean = float(log_gain[finite_gain].mean().item())

                    else:
                        # Original CATS-compatible selectors retained for fair ablations.
                        probs_student = torch.softmax(draft_head_for_select.float(), dim=-1)
                        true_token_ids = torch.argmax(target_p, dim=-1, keepdim=True)
                        true_token_probs = torch.gather(probs_student, 2, true_token_ids).squeeze(-1)
                        student_max_probs, _ = probs_student.max(dim=-1)
                        surprise_ratio = true_token_probs / (student_max_probs + 1e-8)

                        valid_sr = surprise_ratio[raw_loss_mask.bool()]
                        valid_sr = valid_sr[torch.isfinite(valid_sr)]
                        if valid_sr.numel() > 0:
                            batch_mean = float(valid_sr.mean().item())
                            global_sr_mean = (
                                args.ema_momentum * global_sr_mean
                                + (1 - args.ema_momentum) * batch_mean
                            )
                        else:
                            accelerator.print(f"[SRSkip] step={step_idx} no finite valid_sr")

                        current_threshold = global_sr_mean * args.difficulty_scale
                        epoch_threshold_sum += current_threshold
                        epoch_threshold_count += 1
                        is_hard_token = surprise_ratio < current_threshold

                        if args.visual_mask_on_replay:
                            replay_keep_mask = visual_static_mask.bool() & is_replay_sample
                        else:
                            replay_keep_mask = is_replay_sample

                        if args.selection_mode == "sr":
                            final_select_mask = is_hard_token | replay_keep_mask
                        elif args.selection_mode == "visual":
                            final_select_mask = visual_static_mask.bool() | replay_keep_mask
                        elif args.selection_mode == "sr_visual_or":
                            final_select_mask = (is_hard_token | visual_static_mask.bool()) | replay_keep_mask
                        elif args.selection_mode == "sr_visual_and":
                            final_select_mask = (is_hard_token & visual_static_mask.bool()) | replay_keep_mask
                        else:
                            raise ValueError(f"Unknown selection_mode: {args.selection_mode}")

                        dynamic_mask = raw_loss_mask * final_select_mask.long()

                    # Safety: no selection mechanism may activate prompt/padding positions.
                    dynamic_mask = dynamic_mask.long() * raw_loss_mask.long()

                    total_active_tokens = raw_loss_mask.sum().item()
                    selected_active_tokens = dynamic_mask.sum().item()
                    selection_ratio = selected_active_tokens / (total_active_tokens + 1e-8)
                    replay_tokens = (dynamic_mask * is_replay_sample.long()).sum().item()
                    selected_new_tokens = (
                        dynamic_mask * (~is_replay_sample).long()
                    ).sum().item()

                    if args.selection_mode == "gain" and np.isfinite(gain_beta_mean):
                        epoch_gain_beta_sum += gain_beta_mean
                        epoch_gain_gap_sum += gain_gap_mean
                        epoch_gain_visual_sum += gain_visual_mean
                        if np.isfinite(gain_log_mean):
                            epoch_gain_log_sum += gain_log_mean
                        epoch_gain_batches += 1

                    for i in range(dynamic_mask.shape[0]):
                        selected_token_list.append(float(dynamic_mask[i].sum().item()))

                    if (
                        accelerator.is_local_main_process
                        and args.mask_debug_print_steps > 0
                        and (step_idx % args.mask_debug_print_steps == 0)
                    ):
                        b = 0
                        b_valid = raw_loss_mask[b].bool()
                        b_selected = dynamic_mask[b].bool()
                        b_input_ids = batch["input_ids"][b]
                        selected_pos = torch.nonzero(b_selected, as_tuple=False).squeeze(-1)

                        print("\n" + "=" * 140)
                        print(
                            f"[MaskDebug] epoch={epoch + 1} step={step_idx} "
                            f"mode={args.selection_mode} replay={bool(is_replay_sample[b].item())}"
                        )
                        print(
                            f"selected={int(b_selected.sum().item())}/"
                            f"{int(b_valid.sum().item())} "
                            f"({float(b_selected.sum().item()) / (float(b_valid.sum().item()) + 1e-8):.2%})"
                        )

                        show_n = min(args.mask_debug_topk, selected_pos.numel())
                        if args.selection_mode == "gain":
                            print(
                                f"gain_keep_ratio={args.gain_keep_ratio:.3f} | "
                                f"gain_temp={args.gain_temp:.3f} | "
                                f"visual_lambda={args.gain_visual_lambda:.3f}"
                            )
                            for pos in selected_pos[:show_n].tolist():
                                tid = int(b_input_ids[pos].item())
                                print(
                                    f"  [SELECTED] pos={pos:4d} token_id={tid:6d} "
                                    f"beta={float(beta[b, pos].item()):.6f} "
                                    f"gap={float(1.0 - beta[b, pos].item()):.6f} "
                                    f"R={float(reachability[b, pos].item()):.6e} "
                                    f"C={float(continuation[b, pos].item()):.6f} "
                                    f"v_norm={float(visual_norm[b, pos].item()):.6f} "
                                    f"logG={float(log_gain[b, pos].item()):.6f}"
                                )
                        else:
                            print(
                                f"threshold={current_threshold:.6f} | "
                                f"global_sr_mean={global_sr_mean:.6f}"
                            )
                            for pos in selected_pos[:show_n].tolist():
                                tid = int(b_input_ids[pos].item())
                                print(
                                    f"  [SELECTED] pos={pos:4d} token_id={tid:6d} "
                                    f"sr={float(surprise_ratio[b, pos].item()):.6f} "
                                    f"hard={int(is_hard_token[b, pos].item())} "
                                    f"visual_static={int(visual_static_mask[b, pos].item())}"
                                )
                        print("=" * 140)

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

                    replay_count = (dynamic_mask * is_replay_sample.long()).sum().item()

                    selected_new_count = max(0.0, selected_tokens - replay_count)



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

                        "stage2/selected_new_tokens_count": float(selected_new_tokens),

                        "stage2/replay_tokens_count": float(replay_tokens),

                        "stage2/global_sr_mean": float(global_sr_mean),

                        "stage2/gain_beta_mean": float(gain_beta_mean),

                        "stage2/gain_gap_mean": float(gain_gap_mean),

                        "stage2/gain_log_mean": float(gain_log_mean),

                        "stage2/gain_visual_mean": float(gain_visual_mean),

                    }

                )



        epoch_time = time.time() - start_time

        epoch_time_list.append(epoch_time)



        correct_t = accelerator.gather_for_metrics(torch.tensor(correct, device=accelerator.device)).sum().item()

        total_t = accelerator.gather_for_metrics(torch.tensor(total, device=accelerator.device)).sum().item()

        epoch_loss = epoch_loss / max(1, num_batches)

        avg_threshold = (
            epoch_threshold_sum / epoch_threshold_count
            if epoch_threshold_count > 0
            else float("nan")
        )
        avg_gain_beta = epoch_gain_beta_sum / max(1, epoch_gain_batches)
        avg_gain_gap = epoch_gain_gap_sum / max(1, epoch_gain_batches)
        avg_gain_log = epoch_gain_log_sum / max(1, epoch_gain_batches)
        avg_gain_visual = epoch_gain_visual_sum / max(1, epoch_gain_batches)



        if accelerator.is_local_main_process:

            if args.selection_mode == "gain":
                accelerator.print(
                    f"[Stage2] Epoch {epoch + 1}/{args.stage2_epochs} | "
                    f"mode=gain | input={args.stage2_input_hidden_key} | "
                    f"target={args.stage2_target_hidden_key} | loss={epoch_loss:.4f} | "
                    f"acc={100 * correct_t / max(1, total_t):.2f}% | "
                    f"beta={avg_gain_beta:.4f} | gap={avg_gain_gap:.4f} | "
                    f"logG={avg_gain_log:.4f} | visual={avg_gain_visual:.4f} | "
                    f"time={epoch_time:.1f}s"
                )
            else:
                accelerator.print(
                    f"[Stage2] Epoch {epoch + 1}/{args.stage2_epochs} | "
                    f"mode={args.selection_mode} | "
                    f"input={args.stage2_input_hidden_key} | "
                    f"target={args.stage2_target_hidden_key} | "
                    f"loss={epoch_loss:.4f} | "
                    f"acc={100 * correct_t / max(1, total_t):.2f}% | "
                    f"thr={avg_threshold:.4f} | sr_mean={global_sr_mean:.4f} | "
                    f"time={epoch_time:.1f}s"
                )

            if wandb is not None:

                wandb.log(

                    {

                        "stage2/epoch_loss": float(epoch_loss),

                        "stage2/epoch_acc": float(correct_t / max(1, total_t)),

                        "stage2/avg_threshold": float(avg_threshold),

                        "stage2/epoch_gain_beta": float(avg_gain_beta),

                        "stage2/epoch_gain_gap": float(avg_gain_gap),

                        "stage2/epoch_gain_log": float(avg_gain_log),

                        "stage2/epoch_gain_visual": float(avg_gain_visual),

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

    parser.add_argument("--stage1_datadir", type=str, default="/media/user/d/xyk/data1/llava_instruct/qwen2.5vl/qwen2.5vl_7B_0_67999/0")

    parser.add_argument("--stage2_datadir", type=str, default="/media/user/d/xyk/data1/ScienceQA/qwen2.5vl_distill/0")

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

    parser.add_argument("--stage1_outdir", type=str, default="/media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/ScienceQA/stage1")



    # stage2

    parser.add_argument("--stage2_epochs", type=int, default=5)

    parser.add_argument("--stage2_bs", type=int, default=1)

    parser.add_argument("--stage2_lr", type=float, default=8e-6)

    parser.add_argument("--stage2_warmup_steps", type=int, default=500)

    parser.add_argument("--stage2_outdir", type=str, default="/media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/ScienceQA/gain")

    parser.add_argument(

        "--stage2_input_hidden_key",

        type=str,

        default="hidden_state",

        help="Stage2 输入使用哪个 hidden 字段",

    )

    parser.add_argument(

        "--stage2_target_hidden_key",

        type=str,

        default="hidden_state",

        help="Stage2 target 使用哪个 hidden 字段",

    )

    parser.add_argument(

        "--selection_mode",

        type=str,

        default="gain",

        choices=["gain", "sr", "visual", "sr_visual_or", "sr_visual_and"],

        help="Token selection strategy in Stage2. 'gain' uses the unified G_t score.",

    )

    parser.add_argument(

        "--visual_mask_on_replay",

        action="store_true",

        help="If set, also apply visual_static_mask on replay samples.",

    )

    # Unified acceptance-gain selector
    parser.add_argument(
        "--gain_keep_ratio",
        type=float,
        default=0.50,
        help="Per-sample fraction of valid answer tokens retained by Top-K G_t. "
             "For fair comparison, set this to the measured CATS retention budget.",
    )
    parser.add_argument(
        "--gain_min_tokens",
        type=int,
        default=1,
        help="Minimum number of selected tokens for each non-empty sample.",
    )
    parser.add_argument(
        "--gain_visual_lambda",
        type=float,
        default=1.0,
        help="Visual bonus in G_t = G_accept * (1 + lambda_v * normalized_visual_dep). "
             "Set to 0 for pure acceptance-gain ablation.",
    )
    parser.add_argument(
        "--gain_temp",
        type=float,
        default=1.0,
        help="Common temperature used for target/draft distributions when computing beta_t.",
    )
    parser.add_argument(
        "--gain_eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon used in gain computation.",
    )
    parser.add_argument(
        "--gain_select_on_replay",
        action="store_true",
        help="Apply G_t Top-K selection to Stage-I replay samples too. Default keeps replay dense.",
    )



    parser.add_argument(

        "--mask_debug_print_steps",

        type=int,

        default=-1,

        help="每多少个step打印一次mask选择信息，设为-1表示不打印",

    )

    parser.add_argument(

        "--mask_debug_topk",

        type=int,

        default=10,

        help="每次debug打印前几个被选中的token",

    )



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

    if args.teacher_temp <= 0:
        raise ValueError(f"--teacher_temp must be > 0, got {args.teacher_temp}")
    if args.gain_temp <= 0:
        raise ValueError(f"--gain_temp must be > 0, got {args.gain_temp}")
    if not (0.0 < args.gain_keep_ratio <= 1.0):
        raise ValueError(f"--gain_keep_ratio must be in (0, 1], got {args.gain_keep_ratio}")
    if args.gain_min_tokens < 1:
        raise ValueError(f"--gain_min_tokens must be >= 1, got {args.gain_min_tokens}")
    if args.gain_visual_lambda < 0:
        raise ValueError(f"--gain_visual_lambda must be >= 0, got {args.gain_visual_lambda}")
    if args.gain_eps <= 0:
        raise ValueError(f"--gain_eps must be > 0, got {args.gain_eps}")



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
