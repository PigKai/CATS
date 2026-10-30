import argparse
import re
import random
import sys
from io import BytesIO
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.cats import vision_critical_mask


def load_data(path, split):
    root = Path(path)
    if root.is_dir() and ((root / 'dataset_dict.json').exists() or (root / 'state.json').exists()):
        data = load_from_disk(path)
        if hasattr(data, 'keys'):
            return data[split]
        return data
    if root.is_file():
        kind = 'parquet' if root.suffix == '.parquet' else 'json'
        return load_dataset(kind, data_files={split: path}, split=split)
    return load_dataset(path, split=split)


def image_from_example(example, image_root=None):
    image = example.get('image')
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, dict):
        if image.get('bytes') is not None:
            return Image.open(BytesIO(image['bytes'])).convert('RGB')
        image = image.get('path')
    if image is None:
        return None
    path = Path(image)
    if image_root and not path.is_absolute():
        path = Path(image_root) / path
    return Image.open(path).convert('RGB')


def question_from_example(example, benchmark):
    if benchmark == 'llava_instruct':
        turns = example.get('conversations', [])
        question = next(turn['value'] for turn in turns if turn['from'] == 'human')
    else:
        question = example['question']
    question = re.sub(r'<image>\s*', '', question).strip()
    if benchmark == 'scienceqa':
        choices = example.get('choices', [])
        options = '\n'.join(f'({chr(65+i)}) {choice}' for i, choice in enumerate(choices))
        question = f'{question}\n{options}\nChoose the best answer and explain your reasoning.'
    elif benchmark in ('aokvqa', 'textvqa'):
        question += ('\nPerform an OCR task on the provided image. Please extract the text accurately '
                     'and provide a detailed explanation of the process. Ensure the response is '
                     'comprehensive and well-structured.')
    return question


def image_token_ids(processor, config):
    ids = set()
    for key in ('image_token_id', 'image_token_index', 'vision_start_token_id', 'vision_end_token_id'):
        value = getattr(config, key, None)
        if value is not None:
            ids.add(int(value))
    vocabulary = processor.tokenizer.get_vocab()
    for token in ('<image>', '<|image_pad|>', '<|vision_start|>', '<|vision_end|>'):
        if token in vocabulary:
            ids.add(vocabulary[token])
    return ids


def find_question_mask(ids, prompt_length, text, tokenizer):
    mask = torch.zeros_like(ids, dtype=torch.bool)
    candidates = [text, '\n' + text, ' ' + text]
    for candidate in candidates:
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        if not tokens:
            continue
        needle = ids.new_tensor(tokens)
        for start in range(prompt_length - len(tokens), -1, -1):
            if torch.equal(ids[start:start+len(tokens)], needle):
                mask[start:start+len(tokens)] = True
                return mask
    decoded = tokenizer.decode(ids[:prompt_length].tolist(), skip_special_tokens=False)
    start = decoded.rfind(text)
    if start < 0:
        raise ValueError('Cannot locate question tokens in the chat template')
    end = start + len(text)
    previous = 0
    for index in range(prompt_length):
        current = len(tokenizer.decode(ids[:index+1].tolist(), skip_special_tokens=False))
        if current > start and previous < end:
            mask[index] = True
        previous = current
    return mask


def prompt_inputs(processor, question, image):
    content = []
    if image is not None:
        content.append({'type': 'image'})
    content.append({'type': 'text', 'text': question})
    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                         tokenize=False, add_generation_prompt=True)
    kwargs = dict(text=[text], return_tensors='pt')
    if image is not None:
        kwargs['images'] = [image]
    return processor(**kwargs)


@torch.no_grad()
def cache_trajectory(model, processor, inputs, question, gamma=0.1, max_new_tokens=256,
                     target_sampling=False):
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) if torch.is_tensor(value) else value
              for key, value in inputs.items()}
    prompt_length = inputs['input_ids'].shape[1]
    generation = dict(do_sample=target_sampling, num_beams=1, max_new_tokens=max_new_tokens,
                      repetition_penalty=1.0, return_dict_in_generate=False)
    if target_sampling:
        generation.update(temperature=1.0, top_p=1.0, top_k=0)
    sequence = model.generate(**inputs, **generation)
    eos_id = processor.tokenizer.eos_token_id
    if not target_sampling and eos_id is not None and sequence[0, -1].item() != eos_id:
        sequence = torch.cat((sequence, sequence.new_tensor([[eos_id]])), dim=1)
    full_inputs = dict(inputs)
    full_inputs['input_ids'] = sequence
    full_inputs['attention_mask'] = torch.ones_like(sequence)
    full_inputs.pop('position_ids', None)
    outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)
    ids = sequence[0]
    answer = torch.arange(len(ids), device=ids.device) >= prompt_length
    image_ids = image_token_ids(processor, model.config)
    image_mask = torch.zeros_like(answer)
    for token in image_ids:
        image_mask |= ids == token
    image_mask[prompt_length:] = False
    feature_id = getattr(model.config, 'image_token_id', getattr(model.config, 'image_token_index', None))
    if feature_id is None:
        feature_id = processor.tokenizer.get_vocab().get('<|image_pad|>',
                     processor.tokenizer.get_vocab().get('<image>'))
    feature_mask = (ids == feature_id) & ~answer if feature_id is not None else torch.zeros_like(answer)
    keep = ~image_mask
    without_ids = sequence[:, keep]
    without = model(input_ids=without_ids, attention_mask=torch.ones_like(without_ids), use_cache=False)
    answer_positions = answer.nonzero(as_tuple=True)[0]
    without_positions = keep.long().cumsum(0)[answer_positions] - 1
    logp_with = outputs.logits[0, answer_positions - 1].float().log_softmax(-1).gather(
        -1, ids[answer_positions, None]).squeeze(-1)
    logp_without = without.logits[0, without_positions - 1].float().log_softmax(-1).gather(
        -1, ids[answer_positions, None]).squeeze(-1)
    dependence = torch.zeros(len(ids), device=ids.device, dtype=torch.float32)
    dependence[answer_positions] = logp_with - logp_without
    question_mask = find_question_mask(ids, prompt_length, question, processor.tokenizer)
    question_mask &= ~image_mask & ~answer
    embeddings = outputs.hidden_states[0][0]
    data = dict(input_ids=ids.cpu(), hidden_state=outputs.hidden_states[-1][0].cpu(),
                inputs_embeds=embeddings.cpu(), loss_mask=answer.long().cpu(),
                question_mask=question_mask.cpu(), image_mask=feature_mask.cpu(),
                image_embeddings=embeddings[feature_mask].cpu(),
                visual_dep_score=dependence.cpu(),
                visual_static_mask=vision_critical_mask(dependence, answer, gamma).long().cpu(),
                teacher_generated=True, teacher_decoding='sampling' if target_sampling else 'greedy',
                target_sampled=bool(target_sampling), prompt_length=prompt_length)
    for key in ('pixel_values', 'image_grid_thw'):
        if key in inputs:
            data[key] = inputs[key].cpu()
    return data


def encode_conversation(processor, messages, image, add_generation_prompt=False):
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=add_generation_prompt)
    kwargs = dict(text=[text], return_tensors='pt')
    if image is not None:
        kwargs['images'] = [image]
    return processor(**kwargs)


def common_prefix_length(first, second):
    limit = min(len(first), len(second))
    different = (first[:limit] != second[:limit]).nonzero(as_tuple=True)[0]
    return int(different[0]) if different.numel() else limit


@torch.no_grad()
def cache_instruct_conversation(model, processor, example, image):
    messages = []
    image_added = False
    for turn in example['conversations']:
        role = {'human': 'user', 'gpt': 'assistant'}.get(turn['from'], turn['from'])
        content = []
        if role == 'user' and image is not None and not image_added:
            content.append({'type': 'image'})
            image_added = True
        text = re.sub(r'<image>\s*', '', turn['value']).strip()
        content.append({'type': 'text', 'text': text})
        messages.append({'role': role, 'content': content})
    if not messages or messages[0]['role'] not in ('system', 'user'):
        raise ValueError('LLaVA-Instruct conversations must begin with a user or system turn')
    full = encode_conversation(processor, messages, image)
    ids = full['input_ids'][0]
    answer = torch.zeros_like(ids, dtype=torch.bool)
    question_mask = torch.zeros_like(answer)
    for index, message in enumerate(messages):
        if message['role'] == 'assistant':
            if index == 0:
                raise ValueError('An assistant response requires a preceding conversation')
            prompt = encode_conversation(processor, messages[:index], image, True)['input_ids'][0]
            through_answer = encode_conversation(processor, messages[:index+1], image)['input_ids'][0]
            start = common_prefix_length(prompt, ids)
            stop = common_prefix_length(through_answer, ids)
            if stop <= start:
                raise ValueError('Could not align assistant supervision in the chat template')
            answer[start:stop] = True
        elif message['role'] == 'user':
            through_question = encode_conversation(processor, messages[:index+1], image)['input_ids'][0]
            text = next(item['text'] for item in message['content'] if item['type'] == 'text')
            question_mask |= find_question_mask(ids, len(through_question), text, processor.tokenizer)
    if not answer.any():
        raise ValueError('LLaVA-Instruct sample contains no assistant response tokens')
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) if torch.is_tensor(value) else value
              for key, value in full.items()}
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    image_mask = torch.zeros_like(answer)
    feature_id = getattr(model.config, 'image_token_id', getattr(model.config, 'image_token_index', None))
    if feature_id is None:
        vocabulary = processor.tokenizer.get_vocab()
        feature_id = vocabulary.get('<|image_pad|>', vocabulary.get('<image>'))
    if feature_id is not None:
        image_mask = (ids == feature_id) & ~answer
    embeddings = outputs.hidden_states[0][0].cpu()
    data = dict(input_ids=ids.cpu(), hidden_state=outputs.hidden_states[-1][0].cpu(),
                inputs_embeds=embeddings, loss_mask=answer.long().cpu(),
                question_mask=(question_mask & ~answer & ~image_mask).cpu(),
                image_mask=image_mask.cpu(), image_embeddings=embeddings[image_mask.cpu()],
                teacher_generated=False, teacher_decoding='reference', target_sampled=False,
                visual_dep_score=torch.zeros_like(ids, dtype=torch.float32),
                visual_static_mask=torch.zeros_like(ids, dtype=torch.long))
    for key in ('pixel_values', 'image_grid_thw'):
        if key in full:
            data[key] = full[key].cpu()
    return data


def main(family=None, benchmark=None):
    parser = argparse.ArgumentParser('CATS distillation data construction')
    parser.add_argument('--model_path', '--basepath', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--benchmark', choices=['scienceqa', 'aokvqa', 'textvqa', 'llava_instruct'],
                        default=benchmark, required=benchmark is None)
    parser.add_argument('--split', default='train')
    parser.add_argument('--image_root')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--index', default='0')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target_sampling', action='store_true')
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = load_data(args.dataset_path, args.split)
    counts = dict(llava_instruct=68000, scienceqa=12726, aokvqa=17056, textvqa=34602)
    end = args.end if args.end is not None else counts[args.benchmark]
    if args.start < 0 or end <= args.start or end > len(dataset):
        raise ValueError(f'Requested examples [{args.start}, {end}) from a split with {len(dataset)} examples')
    config = AutoConfig.from_pretrained(args.model_path)
    cls = LlavaForConditionalGeneration if config.model_type == 'llava' else Qwen2_5_VLForConditionalGeneration
    model = cls.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                device_map='auto', attn_implementation='eager').eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_path)
    output = Path(args.outdir) / args.index
    output.mkdir(parents=True, exist_ok=True)
    for index in tqdm(range(args.start, end)):
        example = dataset[index]
        image = image_from_example(example, args.image_root)
        if args.benchmark == 'llava_instruct':
            data = cache_instruct_conversation(model, processor, example, image)
        else:
            question = question_from_example(example, args.benchmark)
            inputs = prompt_inputs(processor, question, image)
            data = cache_trajectory(model, processor, inputs, question, args.gamma,
                                    args.max_new_tokens, args.target_sampling)
        data['sample_id'] = str(example.get('question_id', example.get('id', index)))
        data['benchmark'] = args.benchmark
        data['split'] = args.split
        torch.save(data, output / f'data_{index}.ckpt')


if __name__ == '__main__':
    main()
