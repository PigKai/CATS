import argparse
import json
import os
import time
import shortuuid
from tqdm import tqdm
from PIL import Image
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utils import *
from model.ea_model import EaModel
from model.kv_cache import initialize_past_key_values
from model.choices import *


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        datapath,
        max_new_token,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    with open(os.path.expanduser(question_file), "r") as f:
        js = json.load(f)

    if isinstance(js, list):
        questions = js
    elif "questions" in js:
        questions = js["questions"]
    elif "data" in js:
        questions = js["data"]
    else:
        questions = js.get("questions", [])

    data = questions[question_begin:question_end]

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
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
                datapath,
                max_new_token,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
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
        datapath,
        max_new_token,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager"
    )

    tokenizer = model.get_tokenizer()
    model.eval()

    print('warmup ...')
    for j in range(min(10, len(questions))):
        q = questions[j]
        image_path = q.get("image") or q.get("image_path") or q.get("file_name")
        question_text = q.get("question") or q.get("text", "")
        image = Image.open(os.path.join(datapath, image_path)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        text = model.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = model.processor(images=image, text=text, return_tensors="pt")
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw

        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx = model.naivegenerate(
            input_ids=torch.as_tensor(input_ids).cuda(),
            pixel_values=torch.as_tensor(pixel_values).cuda(),
            image_grid_thw=torch.as_tensor(image_grid_thw).cuda(),
            max_new_tokens=max_new_token,
            temperature=temperature,
            log=True,
        )
        torch.cuda.synchronize()
        _ = time.time() - start_time
    print('Warmup done')

    torch.manual_seed(123)
    for question in tqdm(questions):
        image_path = question.get("image") or question.get("image_path") or question.get("file_name")
        question_text = question.get("question") or question.get("text", "")
        image = Image.open(os.path.join(datapath, image_path)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        text = model.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = model.processor(images=image, text=text, return_tensors="pt")
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw

        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx = model.naivegenerate(
            input_ids=torch.as_tensor(input_ids).cuda(),
            pixel_values=torch.as_tensor(pixel_values).cuda(),
            image_grid_thw=torch.as_tensor(image_grid_thw).cuda(),
            max_new_tokens=max_new_token,
            temperature=temperature,
            log=True,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        output_ids = output_ids[0][len(input_ids[0]):]
        new_token = output_ids.shape[-1]

        output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        if tokenizer.eos_token and output.find(tokenizer.eos_token) > 0:
            output = output[: output.find(tokenizer.eos_token)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        question_id = question.get("question_id") or question.get("id") or shortuuid.uuid()
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question_id,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "response": output,
                "idx": idx,
                "new_tokens": new_token,
                "wall_time": total_time,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-model-path", type=str, default="/data1/dhz/Model/EAGLE-EYE-Qwen2.5vl-7B-40k-20epoch")
    parser.add_argument("--base-model-path", type=str, default="/data1/dhz/Model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--load-in-8bit", action="store_false")
    parser.add_argument("--model-id", type=str, default="Qwen2.5-VL-7B-Instruct-fp16-baseline")
    parser.add_argument("--bench-name", type=str, default="MME")
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=50)
    parser.add_argument("--answer-dir", type=str, default="/home/dhz/eagle-eye2/outputs")
    parser.add_argument("--datajson", type=str, default="/data1/dhz/Data/MME/llava_mme.json")
    parser.add_argument("--datapath", type=str, default="/data1/dhz/Data/MME")
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=8)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tree-choices", type=str, default="mc_sim_7b_63")

    args = parser.parse_args()
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    question_file = args.datajson
    datapath = args.datapath

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
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        datapath=datapath,
        max_new_token=args.max_new_token,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        temperature=args.temperature,
        args=args
    )
