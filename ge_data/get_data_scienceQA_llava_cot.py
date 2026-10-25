import argparse
import os
import torch
from io import BytesIO
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# -----------------------------
# 环境与参数 (保持与你的一致)
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=10000)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="/media/user/d/xyk/data1/ScienceQA/llava_gold_answer")
parser.add_argument("--parquet_path", type=str, default="/media/user/d/xyk/data1/ScienceQA/train-00000-of-00001-1028f23e353fbe3e.parquet")
parser.add_argument("--model_path", type=str, default="/media/user/d/xyk/model/llava-1.5-7b-hf")
parser.add_argument("--num_proc", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--use_lecture", action="store_true")
parser.add_argument("--use_hint", action="store_true")
parser.add_argument("--only_solution", action="store_true")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))

# -----------------------------
# 加载模型
# -----------------------------
bigmodel = LlavaForConditionalGeneration.from_pretrained(
    args.model_path, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager"
)
bigprocessor = AutoProcessor.from_pretrained(args.model_path)
tokenizer = bigprocessor.tokenizer

def to_pil_image(img_field):
    if img_field is None: return None
    if isinstance(img_field, Image.Image): return img_field.convert("RGB")
    if isinstance(img_field, dict) and img_field.get("bytes"):
        return Image.open(BytesIO(img_field["bytes"])).convert("RGB")
    return None

# -----------------------------
# 数据处理核心
# -----------------------------
def build_scienceqa_dataset(processor):
    ds = load_dataset("parquet", data_files={"train": args.parquet_path}, split="train")
    n = len(ds)
    ds = ds.select(range(max(0, min(args.start, n)), max(0, min(args.end, n))))
    original_columns = ds.column_names

    def preprocess_function(examples):
        new_examples = {"input_ids": [], "pixel_values": [], "loss_mask": [], "conversation": []}
        
        for i in range(len(examples["question"])):
            image = to_pil_image(examples.get("image", [None])[i])
            question = examples["question"][i]
            choices = examples["choices"][i]
            
            # 答案文本解析
            try: ans_idx = int(examples["answer"][i])
            except: ans_idx = -1
            answer_text = str(choices[ans_idx]) if 0 <= ans_idx < len(choices) else str(examples["answer"][i])

            # 拼接回答
            sol = (examples.get("solution", [""])[i] or "").strip()
            lec = (examples.get("lecture", [""])[i] or "").strip()
            hint = (examples.get("hint", [""])[i] or "").strip()
            parts = [sol] if args.only_solution else [p for p in [sol, f"Lecture: {lec}" if args.use_lecture else "", f"Hint: {hint}" if args.use_hint else ""] if p]
            reason_text = "\n".join(parts)

            prompt_text = f"USER: <image>\n{question}\nChoices: {', '.join([str(c) for c in choices])} ASSISTANT:"
            answer_part = f" Answer: {answer_text}. Reason: {reason_text}{tokenizer.eos_token}"
            full_text = f"<s>{prompt_text}{answer_part}"

            # 使用 processor 处理
            if image is not None:
                # 传入 images 且 text 中只含有一个 <image> 标签
                inputs = processor(text=full_text, images=image, return_tensors="pt")
                # 计算指令长度：同样的逻辑，只编码到 ASSISTANT:
                inputs_inst = processor(text=f"<s>{prompt_text}", images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"][0].tolist()
            else:
                inputs = processor(text=full_text, return_tensors="pt")
                inputs_inst = processor(text=f"<s>{prompt_text}", return_tensors="pt")
                pixel_values = None

            input_ids = inputs["input_ids"][0]
            inst_len = inputs_inst["input_ids"].shape[1]

            loss_mask = torch.zeros_like(input_ids)
            if len(input_ids) > inst_len:
                loss_mask[inst_len:] = 1

            new_examples["input_ids"].append(input_ids.tolist())
            new_examples["loss_mask"].append(loss_mask.tolist())
            new_examples["pixel_values"].append(pixel_values)
            new_examples["conversation"].append(full_text)

        return new_examples

    return ds.map(preprocess_function, batched=True, num_proc=args.num_proc, 
                  batch_size=args.batch_size, remove_columns=original_columns)

@torch.no_grad()
def process_and_save(one_item, outdir):
    input_ids = torch.as_tensor(one_item["input_ids"]).unsqueeze(0).cuda()
    loss_mask = torch.as_tensor(one_item["loss_mask"]).unsqueeze(0)
    pixel_values = one_item["pixel_values"]
    if pixel_values is not None:
        pixel_values = torch.as_tensor(pixel_values).unsqueeze(0).cuda().to(torch.float16)

    # 这里的 forward 必须产生真正的视觉注入
    outs = bigmodel(input_ids=input_ids, pixel_values=pixel_values, output_hidden_states=True)
    
    # hidden_states[0] 是注入后的 embedding
    inputs_embeds = outs.hidden_states[0]
    
    td = {
        "input_ids": input_ids.cpu()[0],
        "inputs_embeds": inputs_embeds.cpu()[0],
        "hidden_state": outs.hidden_states[-1].cpu()[0],
        "loss_mask": loss_mask.cpu()[0],
    }
    os.makedirs(outdir, exist_ok=True)
    torch.save(td, f"{outdir}/data_{len(os.listdir(outdir))}.ckpt")

if __name__ == "__main__":
    ds = build_scienceqa_dataset(bigprocessor)
    outdir = f"{args.outdir}/{args.index}"
    for item in ds:
        process_and_save(item, outdir)
    print(f"Finished: {outdir}")