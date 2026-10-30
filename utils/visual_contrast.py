import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distillation.generate import image_token_ids
from model.cats import vision_critical_mask


def build_visual_static_mask(visual_dep_score, loss_mask, topk_ratio=0.1):
    return vision_critical_mask(visual_dep_score, loss_mask, topk_ratio).long()


@torch.no_grad()
def process_one_ckpt(data, model, processor, topk_ratio=0.1):
    device = model.get_input_embeddings().weight.device
    ids = data['input_ids'].long().flatten().to(device)
    answer = data['loss_mask'].bool().flatten().to(device)
    image_mask = torch.zeros_like(answer)
    for token in image_token_ids(processor, model.config):
        image_mask |= ids == token
    image_mask &= ~answer
    if image_mask.any() and 'pixel_values' not in data:
        raise KeyError('pixel_values is required for target-side visual contrast')
    kwargs = dict(input_ids=ids.unsqueeze(0), attention_mask=torch.ones_like(ids).unsqueeze(0),
                  use_cache=False)
    for key in ('pixel_values', 'image_grid_thw'):
        if key in data:
            value = data[key].to(device)
            kwargs[key] = value.to(model.dtype) if key == 'pixel_values' else value
    if 'pixel_values' in kwargs and kwargs['pixel_values'].ndim == 3 and model.config.model_type == 'llava':
        kwargs['pixel_values'] = kwargs['pixel_values'].unsqueeze(0)
    with_image = model(**kwargs).logits[0]
    keep = ~image_mask
    text_ids = ids[keep]
    without_image = model(input_ids=text_ids.unsqueeze(0),
                          attention_mask=torch.ones_like(text_ids).unsqueeze(0), use_cache=False).logits[0]
    positions = answer.nonzero(as_tuple=True)[0]
    text_positions = keep.long().cumsum(0)[positions] - 1
    if (positions == 0).any() or (text_positions == 0).any():
        raise ValueError('Every answer token must have a conditioning prefix')
    tokens = ids[positions, None]
    full_logp = with_image[positions-1].float().log_softmax(-1).gather(-1, tokens).squeeze(-1)
    text_logp = without_image[text_positions-1].float().log_softmax(-1).gather(-1, tokens).squeeze(-1)
    dependence = torch.zeros_like(ids, dtype=torch.float32)
    dependence[positions] = full_logp - text_logp
    data['visual_dep_score'] = dependence.cpu()
    data['visual_static_mask'] = build_visual_static_mask(dependence, answer, topk_ratio).cpu()
    return data


def main():
    parser = argparse.ArgumentParser('CATS vision-critical token selection')
    parser.add_argument('--src_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--topk_ratio', type=float, default=0.1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--output_dir')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not args.output_dir and not args.overwrite:
        parser.error('Choose --output_dir or --overwrite')
    config = AutoConfig.from_pretrained(args.model_path)
    cls = LlavaForConditionalGeneration if config.model_type == 'llava' else Qwen2_5_VLForConditionalGeneration
    model = cls.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map='auto',
                                attn_implementation='eager').eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_path)
    source = Path(args.src_dir)
    files = sorted(source.rglob('*.ckpt'))[args.start:args.end]
    for path in files:
        data = torch.load(path, map_location='cpu', weights_only=True)
        data = process_one_ckpt(data, model, processor, args.topk_ratio)
        target = Path(args.output_dir) / path.relative_to(source) if args.output_dir else path
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, target)


if __name__ == '__main__':
    main()
