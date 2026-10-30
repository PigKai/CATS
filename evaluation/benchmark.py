import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distillation.generate import load_data, image_from_example, question_from_example, prompt_inputs
from model.ea_model import EaModel


def synchronize():
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


@torch.inference_mode()
def measure(model, inputs, speculative, temperature, max_new_tokens, max_length):
    generate = model.eagenerate if speculative else model.naivegenerate
    synchronize()
    started = time.perf_counter()
    result = generate(**inputs, temperature=temperature, max_new_tokens=max_new_tokens,
                      max_length=max_length, log=True)
    synchronize()
    elapsed = time.perf_counter() - started
    generated = result[0][0, inputs['input_ids'].shape[1]:]
    accepted = [int(value) for value in result[3]] if speculative else []
    return dict(response=model.processor.tokenizer.decode(generated, skip_special_tokens=True),
                new_tokens=len(generated), wall_time=elapsed,
                verification_cycles=len(accepted), accepted_draft_tokens=sum(accepted),
                accept_lens=accepted)


def summarize(records):
    target = sum(row['target']['wall_time'] for row in records if 'target' in row)
    draft = sum(row['draft']['wall_time'] for row in records if 'draft' in row)
    cycles = sum(row['draft']['verification_cycles'] for row in records if 'draft' in row)
    accepted = sum(row['draft']['accepted_draft_tokens'] for row in records if 'draft' in row)
    return dict(samples=len(records), target_wall_time=target, speculative_wall_time=draft,
                speedup=target / draft if target > 0 and draft > 0 else None,
                average_acceptance_length=accepted / cycles if cycles else None)


def main(family=None, benchmark=None, mode='both'):
    parser = argparse.ArgumentParser('CATS benchmark')
    parser.add_argument('--base-model-path', '--base_model_path', required=True)
    parser.add_argument('--ea-model-path', '--ee-model-path', '--ea_model_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--benchmark', choices=['scienceqa', 'aokvqa', 'textvqa'],
                        default=benchmark, required=benchmark is None)
    parser.add_argument('--split', default='test')
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--image_root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--temperature', type=float, nargs='+', choices=[0.0, 1.0], default=[0.0, 1.0])
    parser.add_argument('--mode', choices=['both', 'draft', 'target'], default=mode)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--total_token', type=int, default=60)
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    split = args.split
    dataset = load_data(args.dataset_path, split)
    end = args.end if args.end is not None else len(dataset)
    if args.start < 0 or end > len(dataset) or end <= args.start:
        raise ValueError('Invalid evaluation index range')
    if args.sample_size <= 0 or end - args.start < args.sample_size:
        raise ValueError(f'Uniform evaluation requires {args.sample_size} examples; only {end - args.start} are available')
    indices = random.Random(args.seed).sample(range(args.start, end), args.sample_size)
    model = EaModel.from_pretrained(base_model_path=args.base_model_path,
                                    ea_model_path=args.ea_model_path, torch_dtype=torch.float16,
                                    device_map='auto', attn_implementation='eager',
                                    total_token=args.total_token, depth=args.depth, top_k=args.top_k).eval()
    model.requires_grad_(False)
    device = model.base_model.get_input_embeddings().weight.device
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with output.open('w', encoding='utf-8') as stream:
        for temperature in args.temperature:
            records = []
            warmed = False
            for index in indices:
                example = dataset[index]
                question = question_from_example(example, args.benchmark)
                image = image_from_example(example, args.image_root)
                inputs = prompt_inputs(model.processor, question, image)
                inputs = {key: value.to(device) if torch.is_tensor(value) else value
                          for key, value in inputs.items()}
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
                row = dict(sample_id=str(example.get('question_id', example.get('id', index))),
                           temperature=temperature, benchmark=args.benchmark, split=split,
                           dataset_index=index, seed=args.seed)
                modes = [('target', False), ('draft', True)] if args.mode == 'both' else [
                    ('draft', True) if args.mode == 'draft' else ('target', False)]
                if not warmed:
                    for _, speculative in modes:
                        measure(model, inputs, speculative, temperature, 8, args.max_length)
                    warmed = True
                for label, speculative in modes:
                    random.seed(args.seed + index)
                    torch.manual_seed(args.seed + index)
                    row[label] = measure(model, inputs, speculative, temperature,
                                         args.max_new_tokens, args.max_length)
                records.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
            if not records:
                raise ValueError('The selected evaluation split contains no samples')
            results[str(temperature)] = summarize(records)
    output.with_suffix('.summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
