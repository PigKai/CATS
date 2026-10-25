import json
from transformers import AutoTokenizer
import numpy as np

jsonl_file_base = "/home/dhz/eagle-eye2/outputs/COCO-caption/llava-v1.5-7b-hf-fp16-baseline-temperature-0.0.jsonl"
jsonl_file = "/home/dhz/eagle-eye2/outputs/COCO-caption/llava-v1.5-7b-hf-fp16-ea2-SFDD-temperature-0.0.jsonl"
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
total_idxs=0
speeds=[]
for datapoint in data:
    answer=datapoint['response']
    idx=datapoint['idx']
    tokens=datapoint['new_tokens']
    times = datapoint['wall_time']
    
    speeds.append(tokens/times)

    total_time+=times
    total_token+=tokens
    total_idxs+=(idx+1)

print(total_token)
print(total_idxs)

print("accept_ratio",total_token/total_idxs)

data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)
        
total_time=0
total_token=0
speeds0=[]
total_idxs=0
for datapoint in data:
    answer=datapoint['response']
    tokens=datapoint['new_tokens']
    times = datapoint['wall_time']
    idx=datapoint['idx']

    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens
    total_idxs+=(idx+1)
print("accept_ratio0",total_token/total_idxs)

print('speed',np.array(speeds).mean())
print('speed0',np.array(speeds0).mean())
print("speedup_ratio",np.array(speeds).mean()/np.array(speeds0).mean())


