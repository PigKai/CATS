import argparse
import os
import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description="A-OKVQA -> Qwen2.5-VL ckpt (Model Predicted Answer)")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=50000)
parser.add_argument("--index", type=str, default="0")
parser.add_argument("--gpu_index", type=str, default="2")
parser.add_argument(
    "--outdir",
    type=str,
    default="/media/user/e/xyk/data/OKVQA/qwen2.5vl_3B_distill",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/media/user/d/xyk/data1/A-OKVQA",
)
parser.add_argument("--split", type=str, default="train")
parser.add_argument(
    "--model_path",
    type=str,
    default="/media/user/d/xyk/model/Qwen2.5-VL-3B-Instruct",
)
parser.add_argument("--max_new_tokens", type=int, default=256)

# 新增：前几个样本做详细打印
parser.add_argument("--debug_first_n", type=int, default=3)
parser.add_argument("--print_full_text", action="store_true")

args = parser.parse_args()

# -----------------------------
# Env & Model
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
device = "cuda" if torch.cuda.is_available() else "cpu"

bigmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
).eval()

bigprocessor = AutoProcessor.from_pretrained(args.model_path)
tokenizer = bigprocessor.tokenizer


# -----------------------------
# Core Logic
# -----------------------------
def build_prompt_text(question: str) -> str:
    instruction = (
        "Perform an OCR task on the provided image. Please extract the text accurately "
        "and provide a detailed explanation of the process. Ensure the response is "
        "comprehensive and well-structured."
    )
    return f"{question}\n{instruction}"


def build_generation_conversation(user_text: str):
    conversation = [
        {
            "role": "system",
            "content": "A chat between a curious human and an artificial intelligence assistant. "
                       "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return conversation


def build_full_conversation(user_text: str, assistant_text: str):
    conversation = [
        {
            "role": "system",
            "content": "A chat between a curious human and an artificial intelligence assistant. "
                       "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
        {
            "role": "assistant",
            "content": assistant_text,
        },
    ]
    return conversation


def print_debug_info(
    idx,
    item,
    raw_question,
    user_text,
    generated_text,
    full_text,
    input_ids,
    inputs_embeds,
    hidden_state_big,
    loss_mask,
    actual_inst_len,
):
    print("\n" + "=" * 100)
    print(f"[DEBUG SAMPLE] idx = {idx}")
    print("=" * 100)

    print(f"question_id : {item.get('question_id', 'N/A')}")
    print(f"image_id    : {item.get('image_id', 'N/A')}")
    print(f"raw_question: {raw_question}")
    print("-" * 100)
    print("[USER TEXT]")
    print(user_text)
    print("-" * 100)
    print("[GENERATED TEXT]")
    print(generated_text)

    print("-" * 100)
    print("[SHAPES]")
    print(f"input_ids.shape      = {tuple(input_ids.shape)}")
    print(f"inputs_embeds.shape  = {tuple(inputs_embeds.shape)}")
    print(f"hidden_state.shape   = {tuple(hidden_state_big.shape)}")
    print(f"loss_mask.shape      = {tuple(loss_mask.shape)}")

    loss_mask_1d = loss_mask[0]
    selected_pos = torch.nonzero(loss_mask_1d, as_tuple=False).squeeze(-1)

    print("-" * 100)
    print("[MASK INFO]")
    print(f"actual_inst_len      = {actual_inst_len}")
    print(f"loss_mask.sum()      = {int(loss_mask_1d.sum().item())}")

    if selected_pos.numel() > 0:
        answer_start = int(selected_pos[0].item())
        answer_end = int(selected_pos[-1].item()) + 1
        print(f"answer_start         = {answer_start}")
        print(f"answer_end           = {answer_end}")
        print(f"answer_len           = {answer_end - answer_start}")
    else:
        answer_start = None
        answer_end = None
        print("answer_start         = None")
        print("answer_end           = None")
        print("answer_len           = 0")

    print("-" * 100)
    print("[TOKEN RANGE CHECK]")
    print(f"input_ids min/max    = {int(input_ids.min().item())} / {int(input_ids.max().item())}")

    try:
        decoded_full = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print("[FULL TEXT | skip_special_tokens=False]")
        if args.print_full_text:
            print(decoded_full)
        else:
            print(decoded_full[:2000])
    except Exception as e:
        print(f"[FULL TEXT DECODE ERROR] {e}")

    try:
        decoded_full_clean = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print("-" * 100)
        print("[FULL TEXT | skip_special_tokens=True]")
        if args.print_full_text:
            print(decoded_full_clean)
        else:
            print(decoded_full_clean[:2000])
    except Exception as e:
        print(f"[FULL TEXT CLEAN DECODE ERROR] {e}")

    if answer_start is not None:
        try:
            answer_ids = input_ids[0, answer_start:answer_end]
            decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
            print("-" * 100)
            print("[DECODED ANSWER SEGMENT FROM loss_mask]")
            print(decoded_answer)
        except Exception as e:
            print(f"[ANSWER SEGMENT DECODE ERROR] {e}")

    if args.print_full_text:
        print("-" * 100)
        print("[FULL CHAT TEMPLATE TEXT]")
        print(full_text)

    print("=" * 100 + "\n")


@torch.no_grad()
def process_single_item(item, idx, outdir, do_debug=False):
    image = item["image"]
    raw_question = re.sub(r"\n?<image>\n?", "", item["question"])
    user_text = build_prompt_text(raw_question)

    # ---------------------------------------
    # 1) 构建 generation prompt
    # ---------------------------------------
    gen_conversation = build_generation_conversation(user_text)

    prompt = bigprocessor.apply_chat_template(
        gen_conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = bigprocessor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    # ---------------------------------------
    # 2) 生成回答
    # ---------------------------------------
    output_ids = bigmodel.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    generated_ids = output_ids[0][prompt_len:]
    generated_text = bigprocessor.batch_decode(
        [generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # ---------------------------------------
    # 3) 构建完整对话
    # ---------------------------------------
    full_conversation = build_full_conversation(user_text, generated_text)

    full_text = bigprocessor.apply_chat_template(
        full_conversation,
        tokenize=False,
        add_generation_prompt=False
    )

    full_inputs = bigprocessor(
        text=[full_text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)

    input_ids = full_inputs["input_ids"]

    # instruction 的真实长度
    inst_inputs = bigprocessor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)
    actual_inst_len = inst_inputs["input_ids"].shape[1]

    # ---------------------------------------
    # 4) 提 hidden states
    # ---------------------------------------
    outputs = bigmodel(
        **full_inputs,
        output_hidden_states=True,
        return_dict=True
    )

    inputs_embeds = outputs.hidden_states[0]
    hidden_state_big = outputs.hidden_states[-1]

    # ---------------------------------------
    # 5) 构建 loss_mask
    # ---------------------------------------
    loss_mask = torch.zeros_like(input_ids)
    if input_ids.shape[1] > actual_inst_len:
        loss_mask[0, actual_inst_len:] = 1

    # ---------------------------------------
    # 6) 保存
    # ---------------------------------------
    td = {
        "input_ids": input_ids.cpu()[0],
        "inputs_embeds": inputs_embeds.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": loss_mask.cpu()[0],
    }
    if "pixel_values" in full_inputs and full_inputs["pixel_values"] is not None:
        td["pixel_values"] = full_inputs["pixel_values"].detach().cpu()

    if "image_grid_thw" in full_inputs and full_inputs["image_grid_thw"] is not None:
        td["image_grid_thw"] = full_inputs["image_grid_thw"].detach().cpu()

    save_path = os.path.join(outdir, f"data_{idx}.ckpt")
    torch.save(td, save_path)

    # ---------------------------------------
    # 7) debug print
    # ---------------------------------------
    if do_debug:
        print_debug_info(
            idx=idx,
            item=item,
            raw_question=raw_question,
            user_text=user_text,
            generated_text=generated_text,
            full_text=full_text,
            input_ids=input_ids.detach().cpu(),
            inputs_embeds=inputs_embeds.detach().cpu(),
            hidden_state_big=hidden_state_big.detach().cpu(),
            loss_mask=loss_mask.detach().cpu(),
            actual_inst_len=actual_inst_len,
        )


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    ds = load_dataset(args.dataset_path, split=args.split, trust_remote_code=True)

    total_n = len(ds)
    start = max(0, min(args.start, total_n))
    end = max(start, min(args.end, total_n))

    ds = ds.select(range(start, end))

    save_dir = os.path.join(args.outdir, args.index)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting processing {len(ds)} samples (from {start} to {end})...")
    print(f"Debug first n samples = {args.debug_first_n}")

    for i, item in enumerate(tqdm(ds)):
        global_idx = start + i
        do_debug = (i < args.debug_first_n)
        process_single_item(item, global_idx, save_dir, do_debug=do_debug)