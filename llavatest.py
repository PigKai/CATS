from PIL import Image
import requests
from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration
import torch
import time

model_name_or_path = "/media/user/d/xyk/model/llava-1.5-7b-hf"
device = "cuda:0"


model = LlavaForConditionalGeneration.from_pretrained(
  model_name_or_path,
  device_map=device,
  torch_dtype=torch.float16,
  attn_implementation='eager'
)

processor = AutoProcessor.from_pretrained(model_name_or_path)

processor.patch_size=14

url= "/media/user/e/xyk/eagle-eye2/eagle-eye2/example.jpg"

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "image","image":url},
          {"type": "text", "text": "Describe the image in detail."}
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image = Image.open(url)
inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, torch.float16)

generate_ids = model.generate(**inputs, max_new_tokens=100)

response = processor.decode(generate_ids[0], skip_special_tokens=True)

print(response)


