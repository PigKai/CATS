import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.cats import enhance_hidden_states, pool_visual_embeddings


def augment_hidden_state(hidden_state, image_embeddings, question_mask, answer_mask,
                         alpha_q=0.02, alpha_a=0.1, pool_type='mean'):
    if image_embeddings.shape[0] == 0:
        return hidden_state.clone()
    return enhance_hidden_states(hidden_state, image_embeddings, question_mask, answer_mask,
                                 alpha_q, alpha_a, pool_type)


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
        data['vision_aug_config'] = dict(pool_type=args.pool_type, eta_q=args.alpha_q, eta_a=args.alpha_a)
        target = Path(args.output_dir) / path.relative_to(root) if args.output_dir else path
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, target)


if __name__ == '__main__':
    main()
