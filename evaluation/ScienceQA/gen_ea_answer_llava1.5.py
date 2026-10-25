import argparse
import json
import os
import time
import shortuuid
from tqdm import tqdm
from PIL import Image
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utils import *
from model.ea_model import EaModel
from model.kv_cache import initialize_past_key_values
from model.choices import *

from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def format_question(sample):
    """将ScienceQA样本格式化为问题文本"""
    question = sample['question']
    choices = sample['choices']
    hint = sample['hint']
    
    # 构建选项文本
    options_text = ""
    for i, choice in enumerate(choices):
        options_text += f"{chr(65+i)}. {choice}\n"
    
    # 构建完整提示
    prompt = ""
    if hint:
        prompt += f"Context: {hint}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += f"Options:\n{options_text}\n"
    # prompt += "Please select the correct answer from the options above. Only respond with the letter (A, B, C, or D) of your choice."
    
    return prompt


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    # 加载ScienceQA数据集
    dataset = load_dataset('/media/user/d/xyk/data1/ScienceQA', cache_dir=args.cache_dir)
    
    # 根据split选择数据集
    data_split = dataset[args.split]
    
    # 只选择带图像的样本（如果指定）
    if args.image_only:
        data = [sample for sample in data_split if sample['image'] is not None]
    else:
        data = list(data_split)
    
    data = data[question_begin:question_end]
    
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(data) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(data), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                data[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args,
                question_begin + i  # 传递起始索引
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args,
        start_idx=0
):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=torch.float16,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager"
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('warmup ...')
    
    # warmup - 使用带图像的样本
    warmup_samples = [q for q in questions if q['image'] is not None][:10]
    if len(warmup_samples) < 10:
        warmup_samples = questions[:10]
    
    for j in range(min(10, len(warmup_samples))):
        sample = warmup_samples[j]
        prompt = format_question(sample)
        
        if sample['image'] is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            image = sample['image']
            if isinstance(image, dict):
                # 针对 datasets 库加载的字典格式进行转换
                if 'image' in image: # 某些版本的 datasets 结构
                    image = image['image']
                else:
                    import io
                    image = Image.open(io.BytesIO(image['bytes']))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            image = None
                
        text = model.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if image is not None:
            inputs = model.processor(images=image, text=text, return_tensors="pt")
            input_ids = inputs.input_ids
            pixel_values = inputs.pixel_values
            
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, idx = model.eagenerate(
                input_ids=torch.as_tensor(input_ids).cuda(),
                pixel_values=torch.as_tensor(pixel_values).cuda(),
                max_new_tokens=max_new_token,
                temperature=temperature,
                log=True
            )
            torch.cuda.synchronize()
        else:
            inputs = model.processor(text=text, return_tensors="pt")
            input_ids = inputs.input_ids
            
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, idx = model.eagenerate(
                input_ids=torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_token,
                temperature=temperature,
                log=True
            )
            torch.cuda.synchronize()
            
    print('Warmup done')

    torch.manual_seed(123)
    total_tokens = 0
    total_time = 0
    
    for q_idx, question in enumerate(tqdm(questions)):
        prompt = format_question(question)
        
        if question['image'] is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            image = question['image']
            if isinstance(image, dict):
                if 'image' in image:
                    image = image['image']
                else:
                    import io
                    image = Image.open(io.BytesIO(image['bytes']))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            image = None
                
        text = model.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if image is not None:
            inputs = model.processor(images=image, text=text, return_tensors="pt")
            input_ids = inputs.input_ids
            pixel_values = inputs.pixel_values
            
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, idx = model.eagenerate(
                input_ids=torch.as_tensor(input_ids).cuda(),
                pixel_values=torch.as_tensor(pixel_values).cuda(),
                max_new_tokens=max_new_token,
                temperature=temperature,
                log=True
            )
            torch.cuda.synchronize()
        else:
            inputs = model.processor(text=text, return_tensors="pt")
            input_ids = inputs.input_ids
            
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, idx = model.eagenerate(
                input_ids=torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_token,
                temperature=temperature,
                log=True
            )
            torch.cuda.synchronize()
            
        sample_time = time.time() - start_time
        output_ids = output_ids[0][len(input_ids[0]):]
        new_token = output_ids.shape[-1]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        if tokenizer.eos_token and output.find(tokenizer.eos_token) > 0:
            output = output[: output.find(tokenizer.eos_token)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        
        total_tokens += new_token
        total_time += sample_time
        
        # Dump answers
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": start_idx + q_idx,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "has_image": question['image'] is not None,
                "response": output,
                "idx": idx,
                "new_tokens": new_token,
                "wall_time": sample_time,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    
    # 打印速度统计
    print(f"\n{'='*50}")
    print(f"Speed Statistics:")
    print(f"{'='*50}")
    print(f"Total Samples: {len(questions)}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Tokens/Second: {total_tokens/total_time:.2f}")
    print(f"Avg Time/Sample: {total_time/len(questions):.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/media/user/d/xyk/model/Eagle_llava_draft/endoSpec_distill/ScienceQA/SFT/checkpoint_epoch_5",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/media/user/d/xyk/model/llava-1.5-7b-hf",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="llava-v1.5-7b-hf-fp16-ea")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="ScienceQA",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        default=0,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", 
        type=int, 
        default=50,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-dir", type=str, default="/media/user/e/xyk/eagle-eye2/eagle-eye2/outputs/ScienceQA", help="The output answer file.")
    parser.add_argument("--cache-dir", type=str, default="/media/user/d/xyk/data1/ScienceQA", help="Cache directory for the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train/validation/test).")
    parser.add_argument("--image-only", action="store_true", help="Only evaluate samples with images.")

    parser.add_argument(
        "--max-new-token",
        type=int,
        default=256,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=64,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=7,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    args.tree_choices = eval(args.tree_choices)
    
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    answer_file = os.path.join(args.answer_dir, args.bench_name, f"{args.model_id}.jsonl")
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    print(f"Output to {answer_file}")

    run_eval(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        model_id=args.model_id,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        temperature=args.temperature,
        args=args
    )
