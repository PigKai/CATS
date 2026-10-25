import os
import argparse
import torch
from transformers import AutoProcessor

def find_image_token_id(tokenizer):
    """
    找到 Qwen2.5-VL 的视觉 token <|image_pad|>
    """
    candidates = ["<|image_pad|>"]
    for tok in candidates:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            return tok, tok_id
    raise ValueError("Cannot find image token id from tokenizer.")

def build_masks(input_ids, loss_mask, image_token_id):
    vision_mask = (input_ids == image_token_id)
    answer_mask = (loss_mask == 1)
    prompt_text_mask = (~vision_mask) & (~answer_mask)
    return vision_mask, prompt_text_mask, answer_mask

def make_field_name(pool_type: str, alpha_a: float, alpha_q: float) -> str:
    return f"{pool_type}_hidden_aug_a{int(alpha_a*100)}_q{int(alpha_q*100)}"

def pool_visual_hidden(hidden_state, vision_mask, pool_type="mean"):
    vision_hidden = hidden_state[vision_mask]
    if pool_type == "mean":
        return vision_hidden.mean(dim=0, keepdim=True)
    elif pool_type == "max":
        return vision_hidden.max(dim=0, keepdim=True).values
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

def augment_hidden_state(hidden_state, input_ids, loss_mask, image_token_id,
                         alpha_q=0.02, alpha_a=0.10, pool_type="mean"):
    hidden_state = hidden_state.float()
    vision_mask, prompt_mask, answer_mask = build_masks(input_ids, loss_mask, image_token_id)
    if vision_mask.sum() == 0:
        return None
    global_visual = pool_visual_hidden(hidden_state, vision_mask, pool_type)
    aug_hidden = hidden_state.clone()
    if prompt_mask.sum() > 0:
        aug_hidden[prompt_mask] += alpha_q * global_visual
    if answer_mask.sum() > 0:
        aug_hidden[answer_mask] += alpha_a * global_visual
    return aug_hidden.half()

def print_norm_stats(name, values):
    if values.numel() == 0:
        print(f"{name}: empty")
        return
    print(f"{name}: min={float(values.min()):.4f}, max={float(values.max()):.4f}, mean={float(values.mean()):.4f}")

def preview_one_field(td, tokenizer, image_token_id, save_key, max_print_tokens=20):
    input_ids = td["input_ids"]
    loss_mask = td["loss_mask"]
    vision_mask, prompt_mask, answer_mask = build_masks(input_ids, loss_mask, image_token_id)

    print("="*100)
    print(f"Preview field: {save_key}")
    print("Shapes:")
    print(" input_ids      :", tuple(td["input_ids"].shape))
    print(" hidden_state   :", tuple(td["hidden_state"].shape))
    print(f" {save_key:<24}:", tuple(td[save_key].shape))
    print(" loss_mask      :", tuple(td["loss_mask"].shape))

    print("\nCounts:")
    print(" num_vision_tokens      :", int(vision_mask.sum()))
    print(" num_prompt_text_tokens :", int(prompt_mask.sum()))
    print(" num_answer_tokens      :", int(answer_mask.sum()))

    hs = td["hidden_state"].float()
    hs_aug = td[save_key].float()
    delta = hs_aug - hs
    eps = 1e-8

    # 视觉增强比例
    if vision_mask.sum() > 0:
        vision_delta = delta[vision_mask]
        vision_norm = hs[vision_mask].norm(dim=-1)
        vision_ratio = vision_delta.norm(dim=-1) / (vision_norm + eps)
        print_norm_stats("Vision token delta ratio", vision_ratio)

    # prompt token 增强比例
    if prompt_mask.sum() > 0:
        delta_prompt = delta[prompt_mask]
        norm_prompt = hs[prompt_mask].norm(dim=-1)
        ratio_prompt = delta_prompt.norm(dim=-1) / (norm_prompt + eps)
        print_norm_stats("Prompt token delta ratio", ratio_prompt)

    # answer token 增强比例
    if answer_mask.sum() > 0:
        delta_ans = delta[answer_mask]
        norm_ans = hs[answer_mask].norm(dim=-1)
        ratio_ans = delta_ans.norm(dim=-1) / (norm_ans + eps)
        print_norm_stats("Answer token delta ratio", ratio_ans)

    # 前几个视觉 token 索引
    vision_idx = vision_mask.nonzero(as_tuple=True)[0]
    if vision_idx.numel() > 0:
        print("First vision token indices:", vision_idx[:max_print_tokens].tolist())
        
    # 前几个 answer token 显示 token 内容
    ans_idx = answer_mask.nonzero(as_tuple=True)[0]
    if ans_idx.numel() > 0:
        print("Answer token-level preview:")
        for i in ans_idx[:max_print_tokens].tolist():
            tid = int(input_ids[i])
            tok = tokenizer.convert_ids_to_tokens([tid])[0]
            print(f"  idx={i:4d} | token={repr(tok)}")
    print()

def collect_ckpt_files(root_dir):
    files = []
    for root, _, fnames in os.walk(root_dir):
        for fname in fnames:
            if fname.endswith(".ckpt"):
                files.append(os.path.join(root, fname))
    files.sort()
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/media/user/d/xyk/data1/llava_instruct/qwen2.5vl/qwen2.5vl_7B_0_67999/0")
    parser.add_argument("--model_path", type=str, default="/media/user/d/xyk/model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--preview_n", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pool_type = "mean"
    alpha_q = 0.02
    alpha_a = 0.10
    save_key = make_field_name(pool_type, alpha_a, alpha_q)

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    image_token_str, image_token_id = find_image_token_id(tokenizer)
    print(f"Detected image token: {image_token_str}, id={image_token_id}")

    ckpt_files = collect_ckpt_files(args.ckpt_dir)
    print(f"Found {len(ckpt_files)} ckpt files.")

    preview_count = 0
    success_count = 0
    skip_count = 0

    for ckpt_path in ckpt_files:
        td = torch.load(ckpt_path, map_location="cpu")

        required_keys = ["input_ids", "hidden_state", "loss_mask"]
        missing_keys = [k for k in required_keys if k not in td]
        if len(missing_keys) > 0:
            print(f"[Warning] Skip {ckpt_path}, missing keys: {missing_keys}")
            skip_count += 1
            continue

        input_ids = td["input_ids"].long()
        hidden_state = td["hidden_state"].float()
        loss_mask = td["loss_mask"].long()

        vision_mask = (input_ids == image_token_id)
        if vision_mask.sum() == 0:
            print(f"[Warning] Skip {ckpt_path}, no image token found.")
            skip_count += 1
            continue

        aug_hidden = augment_hidden_state(
            hidden_state, input_ids, loss_mask, image_token_id,
            alpha_q=alpha_q, alpha_a=alpha_a, pool_type=pool_type
        )

        if aug_hidden is None:
            print(f"[Warning] Skip {ckpt_path}, augment_hidden_state returned None.")
            skip_count += 1
            continue

        td[save_key] = aug_hidden
        torch.save(td, ckpt_path)
        success_count += 1

        if preview_count < args.preview_n:
            preview_one_field(td, tokenizer, image_token_id, save_key)
            preview_count += 1

        print(f"[Done {success_count}] updated: {ckpt_path}")

    print("="*100)
    print("All done.")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Updated directory: {args.ckpt_dir}")
    print(f"Saved field: {save_key}")

if __name__ == "__main__":
    main()