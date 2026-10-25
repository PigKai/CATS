import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from PIL import Image


model_dir = "/media/user/d/xyk/model/Qwen2.5-VL-7B-Instruct" 

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

url ="/media/user/e/xyk/eagle-eye2/eagle-eye2/example.jpg"
processor = AutoProcessor.from_pretrained(model_dir,max_pixels = 1280*28*28)





messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": url,
            },
            {"type": "text", "text": "Describe the image in detail."},
        ],
    }
]
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
) 
image = Image.open(url)
inputs = processor(images=image, text=text , return_tensors='pt').to(model.base_model.device)


# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])


   


