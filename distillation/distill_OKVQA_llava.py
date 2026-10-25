import argparse
import os
import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description="A-OKVQA -> LLaVA ckpt (Model Predicted Answer)")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=50000) 
parser.add_argument("--index", type=str, default="0")
parser.add_argument("--gpu_index", type=str, default="3")
parser.add_argument(
    "--outdir",
    type=str,
    default="/media/user/d/xyk/data1/A-OKVQA/llava_distillation",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/media/user/d/xyk/data1/A-OKVQA", 
)
parser.add_argument("--split", type=str, default="train")
args = parser.parse_args()

# -----------------------------
# Env & Model
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
device = "cuda" if torch.cuda.is_available() else "cpu"

bigname = "/media/user/d/xyk/model/llava-1.5-7b-hf"

bigmodel = LlavaForConditionalGeneration.from_pretrained(
    bigname,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
).eval() 

bigprocessor = AutoProcessor.from_pretrained(bigname)
tokenizer = bigprocessor.tokenizer

# -----------------------------
# Core Logic
# -----------------------------

def build_prompt_text(question: str) -> str:
    # 保持与 TextVQA 脚本一致的 OCR 任务指令
    instruction = (
        "Perform an OCR task on the provided image. Please extract the text accurately "
        "and provide a detailed explanation of the process. Ensure the response is "
        "comprehensive and well-structured."
    )
    return f"{question}\n{instruction}"

@torch.no_grad()
def process_single_item(item, idx, outdir):
    image = item["image"]
    # 清理问题文本中的原生占位符
    raw_question = re.sub(r"\n?<image>\n?", "", item["question"])
    user_text = build_prompt_text(raw_question)

    # 1. 构建符合目标的 Conversation 结构 (含 System Prompt)
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image"}, # 图片放在最后
            ]
        }
    ]
    
    # 2. 生成模型回答 (Hard Label)
    prompt = bigprocessor.apply_chat_template(conversation, add_generation_prompt=True)
    # 注意：LLava-1.5 在使用 apply_chat_template 后，text 中已经包含 <s> 等 token，
    # 再次传入 processor 时要小心重复添加 bos。
    inputs = bigprocessor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    
    prompt_len = inputs["input_ids"].shape[1]

    # 生成 token
    output_ids = bigmodel.generate(
        **inputs,
        max_new_tokens=256, # 考虑到需要 "detailed explanation"，设为 256
        do_sample=False,
    )
    
    generated_ids = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 3. 拼接完整对话 (用于提取 Hidden States)
    # 确保生成的文本后面带上 EOS
    full_text = f"{prompt}{generated_text}{tokenizer.eos_token}"
    
    # 再次通过 processor 得到对齐后的 input_ids
    full_inputs = bigprocessor(images=image, text=full_text, return_tensors="pt").to(device)
    # 此时 full_inputs['input_ids'] 包含了被展开后的视觉 token
    input_ids = full_inputs["input_ids"]
    
    # 重新计算指令长度 (因为视觉 token 展开后的长度在 full_text 下可能与之前不同)
    # 这里的技巧是重新对 prompt 进行一次 processor 处理来获取准确的视觉展开后长度
    inst_inputs = bigprocessor(images=image, text=prompt, return_tensors="pt").to(device)
    actual_inst_len = inst_inputs["input_ids"].shape[1]

    # 4. 前向传播提取 Hidden States
    outputs = bigmodel(
        **full_inputs,
        output_hidden_states=True,
        return_dict=True
    )
    
    inputs_embeds = outputs.hidden_states[0]
    hidden_state_big = outputs.hidden_states[-1]

    # 5. 构建 Loss Mask
    loss_mask = torch.zeros_like(input_ids)
    if input_ids.shape[1] > actual_inst_len:
        loss_mask[0, actual_inst_len:] = 1

    # 6. 保存
    td = {
        "input_ids": input_ids.cpu()[0],
        "inputs_embeds": inputs_embeds.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": loss_mask.cpu()[0]
    }
    if "pixel_values" in full_inputs and full_inputs["pixel_values"] is not None:
        td["pixel_values"] = full_inputs["pixel_values"].cpu()[0]
    # 可选：保存对话文本便于 debug
    # td["conversation"] = full_text

    save_path = os.path.join(outdir, f"data_{idx}.ckpt")
    torch.save(td, save_path)

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # 加载并预选数据
    # 注意：确保你的本地 textvqa.py 或 a-okvqa 加载脚本路径正确
    ds = load_dataset(args.dataset_path, split=args.split, trust_remote_code=True)
    
    total_n = len(ds)
    start = max(0, min(args.start, total_n))
    end = max(start, min(args.end, total_n))
    
    ds = ds.select(range(start, end))
    
    save_dir = os.path.join(args.outdir, args.index)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting processing {len(ds)} samples (from {start} to {end})...")
    for i, item in enumerate(tqdm(ds)):
        # 这里的 idx 使用全局索引 start + i 确保文件名唯一
        process_single_item(item, start + i, save_dir)