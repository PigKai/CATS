from model.ea_model import EaModel

import torch
from PIL import Image
import os
import time
from qwen_vl_utils import process_vision_info

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

base_model_path = "/data1/dhz/Model/llava-v1.5-7b-hf"
ea_model_path = "/data1/dhz/Model/EAGLE-EYE-LLaVA-7B-68k-20epoch-SFDD"

# base_model_path = "/home/dhz/Qwen2.5-VL-7B-Instruct"
# ea_model_path = "/home/dhz/tmp_model/EAGLE-EYE-Qwen2.5vl-7B-10k"

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=ea_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="eager"
)
model.eval()

url = "/home/dhz/eagle-eye/EAGLE_EYE/eagle_eye/example.jpg"

image = Image.open(url)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": url,
                # "min_pixels": 1024*28*28,
                # "max_pixels": 1024*28*28,
            },
            {"type": "text", "text": "Describe the image in detail."},
        ],
    }
]


text = model.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
) 

inputs = model.processor(images=image, text=text , return_tensors='pt').to(model.base_model.device)

output_ids = model.eagenerate(**inputs, temperature=0.0, max_new_tokens=200)

output = model.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1]:],skip_special_tokens=True)