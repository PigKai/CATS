# Eagle eye

## 介绍
目前Eagle-eye已经具备完成了训练以及推理部分的测试，表现出了一定的加速效果。目前兼容llava-v1.5-7b和Qwen2.5-VL-7B-Instruct两个模型。

## 安装 `eagle_eye` 

**新：为了适配qwen2.5vl，transformers>=4.49.0,这里建议选择4.51.1**

```
cd EAGLE_EYE
pip install -e .
```
## 推理

我们提供的推理代码会自动分配模型权重（在多个 GPU 上加载模型），从而允许您运行超过单个 GPU 内存的模型。

**新：高版本transformers提供了chat_template，这里使用其来帮助进行推理。**

### 使用代码

您可以使用我们提供的“eagenerate” 来加速生成，就像使用 Hugging Face 的 “generate” 一样。下面是一个示例。

**新：推理过程中有一些要注意的问题**

**1.qwen2.5vl当以torch.bfloat16加载模型时无论是否使用eagle推理，输出有概率会出现乱码（猜测是transformers的问题)，所以只支持以torch.float16来加载模型。**

**2.Qwen2.5-VL-7B-Instruct在实际运行中显存占用会超过一张3090的显存，这里在使用qwen2.5vl推理时建议将device_map="auto"，保证有足够的显存，llava的话设置为"cuda:0"即可。**

**3.一定要设置为attn_implementation="eager"，由于transformers默认使用SdpaAttention来进行推理，eagle的实现是基于普通的attention，所以要这样设置。**

```python
from eagle_eye.model.ee_model import EeModel

import torch
from PIL import Image
import os
import time

# base_model_path = "/home/dhz/llava-v1.5-7b-hf"
# ee_model_path = "/home/dhz/tmp_model/EAGLE-EYE-LLaVA-7B-10k"

base_model_path = "/home/dhz/Qwen2.5-VL-7B-Instruct"
ee_model_path = "/home/dhz/tmp_model/EAGLE-EYE-Qwen2.5vl-7B-10k"

model = EeModel.from_pretrained(
    base_model_path=base_model_path,
    ee_model_path=ee_model_path,
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

output_ids = model.eagenerate(**inputs,temperature=0.0, max_new_tokens=200)

output = model.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1]:],skip_special_tokens=True)
print(output)

```

注意：LLaVA 和qwen2.5vl都是聊天模型。您需要使用正确的聊天模板，否则会导致模型输出异常，影响 EAGLE-EYE的性能。

## 训练

### 生成训练数据

您可以执行以下命令来生成训练数据。

```python
cd ge_data/
python get_data_all_llava.py -outdir [path of data]

python get_data_all_qwen2.5vl.py -outdir [path of data]
```

### 训练自回归头

```
cd train/
python train_llava.py --tmpdir [path of data]\
--cpdir [path of checkpoints] -- configpath [path of config file]

python train_qwenvl2.5.py --tmpdir [path of data]\
--cpdir [path of checkpoints] -- configpath [path of config file]
```

## 评估

您可以使用以下命令在COCO-caption上测试EAGLE-EYE的速度。

```
cd evaluation/
python gen_ee_answer_llava.py  --ee-model-path [path of EAGLE-EYE weight]\ --base-model-path [path of the original model]\

python gen_ee_answer_qwen2.5vl.py  --ee-model-path [path of EAGLE-EYE weight]\ --base-model-path [path of the original model]\
```

如果你需要特定的加速比，你还需要运行以下命令来获取原版自动回归的速度。

```
python  gen_baseline_answer_llava.py -ee-model-path [path of EAGLE-EYE weight]\ --base-model-path [path of the original model]\


python  gen_baseline_answer_qwen2.5vl.py -ee-model-path [path of EAGLE-EYE weight]\ --base-model-path [path of the original model]\
```

以上两个命令都会生成一个 .jsonl 文件，记录生成结果和实际时间。然后，您可以使用 evaluation/speed.py 来计算速度比率。