import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.cats import enhance_hidden_states


def pool_visual_embeddings(image_embeddings, pool_type='mean'):
    if image_embeddings.ndim != 2 or not len(image_embeddings):
        raise ValueError('Image embeddings must have shape [N, hidden_size] with N > 0')
    values = image_embeddings.float()
    if pool_type == 'mean':
        return values.mean(dim=0, keepdim=True)
    if pool_type == 'max':
        return values.max(dim=0, keepdim=True).values
    if pool_type == 'mean_max':
        return (values.mean(dim=0, keepdim=True) + values.max(dim=0, keepdim=True).values) / 2
    raise ValueError(pool_type)


def augment_hidden_state(hidden_state, image_embeddings, question_mask, answer_mask,
                         alpha_q=0.02, alpha_a=0.1, pool_type='mean'):
    if image_embeddings.shape[0] == 0:
        return hidden_state.clone()
    pooled = pool_visual_embeddings(image_embeddings, pool_type)
    return enhance_hidden_states(hidden_state, pooled, question_mask, answer_mask, alpha_q, alpha_a)


def main():
    parser = argparse.ArgumentParser('CATS visual hidden-state enhancement')
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--output_dir')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--alpha_q', type=float, default=0.02)
    parser.add_argument('--alpha_a', type=float, default=0.1)
    parser.add_argument('--pool_type', choices=['mean', 'max', 'mean_max'], default='mean')
    args = parser.parse_args()
    if not args.output_dir and not args.overwrite:
        parser.error('Choose --output_dir or --overwrite')
    root = Path(args.ckpt_dir)
    for path in sorted(root.rglob('*.ckpt')):
        data = torch.load(path, map_location='cpu', weights_only=True)
        data['vision_aug_hidden_state'] = augment_hidden_state(
            data['hidden_state'], data['image_embeddings'], data['question_mask'].bool(),
            data['loss_mask'].bool(), args.alpha_q, args.alpha_a, args.pool_type)
        target = Path(args.output_dir) / path.relative_to(root) if args.output_dir else path
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, target)


if __name__ == '__main__':
    main()
