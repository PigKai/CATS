import argparse
import copy
import hashlib
import re
import json
import math
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.cats import HardTokenSelector, critical_mask, distillation_loss
from model.cats import enhance_hidden_states, nll_loss, token_logprobs, vision_critical_mask
from model.cnets import Model
from model.configs import EConfig


def load_weight(basepath, keys):
    root = Path(basepath)
    for index_name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index = root / index_name
        if not index.exists():
            continue
        mapping = json.loads(index.read_text())['weight_map']
        for key in keys:
            if key not in mapping:
                continue
            path = root / mapping[key]
            if path.suffix == '.safetensors':
                with safe_open(str(path), framework='pt', device='cpu') as weights:
                    return weights.get_tensor(key)
            return torch.load(path, map_location='cpu', weights_only=True)[key]
    for name in ('model.safetensors', 'pytorch_model.bin'):
        path = root / name
        if not path.exists():
            continue
        if path.suffix == '.safetensors':
            with safe_open(str(path), framework='pt', device='cpu') as weights:
                for key in keys:
                    if key in weights.keys():
                        return weights.get_tensor(key)
        else:
            weights = torch.load(path, map_location='cpu', weights_only=True)
            for key in keys:
                if key in weights:
                    return weights[key]
    raise KeyError(f'No matching weight in {basepath}: {keys}')


def build_lm_head(basepath):
    config = AutoConfig.from_pretrained(basepath)
    text = getattr(config, 'text_config', config)
    keys = ['lm_head.weight', 'language_model.lm_head.weight', 'model.language_model.lm_head.weight']
    if getattr(text, 'tie_word_embeddings', False):
        keys += ['model.embed_tokens.weight', 'model.language_model.embed_tokens.weight',
                 'language_model.model.embed_tokens.weight']
    weight = load_weight(basepath, keys)
    head = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    head.weight = nn.Parameter(weight.float(), requires_grad=False)
    return head.eval()


class CachedTrajectoryDataset(Dataset):
    def __init__(self, directory, max_len=2048, stage=1, gamma=0.1, vhe=False,
                 eta_q=0.02, eta_a=0.1, require_vision=True, require_sampled=False,
                 benchmark=None):
        self.files = sorted(Path(directory).rglob('*.ckpt'))
        if not self.files:
            raise ValueError(f'No cached trajectories in {directory}')
        self.max_len = max_len
        self.stage = stage
        self.gamma = gamma
        self.vhe = vhe and stage == 2
        self.eta_q = eta_q
        self.eta_a = eta_a
        self.require_vision = require_vision
        self.require_sampled = require_sampled
        self.benchmark = benchmark

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        data = torch.load(path, map_location='cpu', weights_only=True)
        if self.benchmark and data.get('benchmark', self.benchmark) != self.benchmark:
            raise ValueError(f'The cached sample belongs to a different benchmark: {path}')
        if self.stage == 1 and data.get('benchmark', 'llava_instruct') != 'llava_instruct':
            raise ValueError(f'Stage I must use LLaVA-Instruct data: {path}')
        if self.stage == 2:
            if 'teacher_generated' in data and not data['teacher_generated']:
                raise ValueError(f'Stage II requires teacher-generated responses: {path}')
            if 'split' in data and data['split'] != 'train':
                raise ValueError(f'Stage II must use the training split: {path}')
            if data.get('benchmark') == 'llava_instruct':
                raise ValueError(f'Stage-I data cannot be used for downstream refinement: {path}')
        ids = data['input_ids'].long().flatten()[:self.max_len]
        hidden = data['hidden_state'][:len(ids)]
        answer = data['loss_mask'].bool().flatten()[:len(ids)]
        if len(ids) < 3 or len(hidden) != len(ids) or len(answer) != len(ids):
            raise ValueError(f'Invalid cached trajectory: {path}')
        if self.stage == 2 and self.require_sampled and data.get('target_sampled') is not True:
            raise ValueError(f'Stage II requires a target-sampled trajectory: {path}')
        visual = torch.zeros_like(answer)
        if self.stage == 2 and self.require_vision:
            if 'visual_dep_score' not in data:
                raise KeyError(f'visual_dep_score missing from {path}')
            visual = vision_critical_mask(data['visual_dep_score'][:len(ids)], answer, self.gamma)
        enhanced = hidden
        if self.vhe and len(data['image_embeddings']):
            question = data['question_mask'][:len(ids)].bool()
            enhanced = enhance_hidden_states(hidden, data['image_embeddings'], question, answer,
                                             self.eta_q, self.eta_a)
        item = dict(hidden_states=hidden[:-2], enhanced_hidden_states=enhanced[:-2],
                    input_ids=ids[1:-1], target=hidden[1:-1], labels=ids[2:],
                    loss_mask=answer[2:], visual_static_mask=visual[2:],
                    attention_mask=torch.ones(len(ids) - 2, dtype=torch.long))
        if 'inputs_embeds' in data:
            item['inputs_embeds'] = data['inputs_embeds'][1:len(ids)-1]
        return item


def collate(features):
    keys = features[0].keys()
    if any(item.keys() != keys for item in features):
        raise ValueError('All cached samples must have the same tensor fields')
    return {key: pad_sequence([item[key] for item in features], batch_first=True)
            for key in keys}


def forward(model, batch, enhanced=False):
    hidden = batch['enhanced_hidden_states'] if enhanced else batch['hidden_states']
    return model(hidden, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                 inputs_embeds=batch.get('inputs_embeds'), use_cache=False)


def load_draft(model, checkpoint):
    path = Path(checkpoint)
    if path.is_dir():
        path = next((path / name for name in ('model.safetensors', 'pytorch_model.bin')
                     if (path / name).exists()), path / 'pytorch_model.bin')
    state = load_file(str(path)) if path.suffix == '.safetensors' else torch.load(
        path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)


def checkpoint_weights(checkpoint):
    path = Path(checkpoint).resolve()
    if path.is_file():
        return path
    for name in ('model.safetensors', 'pytorch_model.bin'):
        if (path / name).is_file():
            return path / name
    raise FileNotFoundError(f'No draft weights in {path}')


def checkpoint_identity(checkpoint, expected_epoch=20):
    weights = checkpoint_weights(checkpoint)
    progress_path = weights.parent / 'progress.json'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['stage'] != 1 or progress['epoch'] != expected_epoch:
            raise ValueError(f'Expected the shared Stage-I epoch-{expected_epoch} checkpoint')
    else:
        match = re.fullmatch(r'(?:state|epoch)_(\d+)', weights.parent.name)
        if not match or int(match[1]) != expected_epoch:
            raise ValueError('The Stage-I initialization epoch cannot be verified')
    digest = hashlib.sha256()
    with weights.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(weights.parent), epoch=expected_epoch, sha256=digest.hexdigest())


def save_checkpoint(accelerator, model, config, progress, directory):
    accelerator.wait_for_everyone()
    accelerator.save_state(str(directory))
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        directory.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
        save_file(state, str(directory / 'model.safetensors'))
        config.save_pretrained(directory)
        (directory / 'progress.json').write_text(json.dumps(progress, indent=2))
    accelerator.wait_for_everyone()


def run_stage(args, stage, model, head, config):
    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == 'no' else args.mixed_precision,
                              gradient_accumulation_steps=args.gradient_accumulation_steps)
    directory = args.stage1_datadir if stage == 1 else args.stage2_datadir
    epochs = args.stage1_epochs if stage == 1 else args.stage2_epochs
    lr = args.stage1_lr if stage == 1 else args.stage2_lr
    bs = args.stage1_bs if stage == 1 else args.stage2_bs
    outdir = Path(args.stage1_outdir if stage == 1 else args.stage2_outdir)
    dataset = CachedTrajectoryDataset(directory, args.max_len, stage, args.gamma,
                                      args.vhe and not args.dense, args.eta_q, args.eta_a,
                                      require_vision=not args.dense and args.selection_mode != 'hard',
                                      require_sampled=args.objective == 'nll',
                                      benchmark=getattr(args, 'benchmark', None) if stage == 2 else None)
    expected = args.stage1_samples if stage == 1 else args.stage2_samples
    if stage == 2 and expected is None:
        sizes = dict(scienceqa=12726, aokvqa=17056, textvqa=34602)
        metadata = torch.load(dataset.files[0], map_location='cpu', weights_only=True)
        benchmark = getattr(args, 'benchmark', None) or metadata.get('benchmark')
        if benchmark in sizes:
            dataset.benchmark = benchmark
            expected = sizes[benchmark]
        elif len(dataset) not in sizes.values():
            raise ValueError('The downstream sample count does not match any benchmark in the paper')
    if expected is not None and len(dataset) != expected:
        raise ValueError(f'Expected {expected} cached examples, found {len(dataset)}')
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate,
                        num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    head = head.to(accelerator.device).eval()
    if stage == 1:
        total_steps = args.stage1_total_steps
        warmup = args.stage1_warmup_steps
    else:
        total_steps = epochs * math.ceil(len(loader) / args.gradient_accumulation_steps)
        warmup = args.stage2_warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scheduler = accelerator.prepare(scheduler)
    selector = HardTokenSelector(args.ema_momentum, args.difficulty_scale)
    accelerator.register_for_checkpointing(selector)
    resume = args.resume_stage1_state if stage == 1 else args.resume_stage2_state
    start_epoch = 0
    if resume:
        progress = json.loads((Path(resume) / 'progress.json').read_text())
        if progress['stage'] != stage:
            raise ValueError('Cannot resume optimizer state from a different training stage')
        if progress['objective'] != args.objective:
            raise ValueError('Cannot resume optimizer state with a different objective')
        if stage == 2 and progress['method'] != args.method:
            raise ValueError('Cannot resume optimizer state from a different refinement method')
        if stage == 2 and progress['initialization']['sha256'] != args.initialization['sha256']:
            raise ValueError('The resumed run uses a different epoch-20 initialization')
        accelerator.load_state(resume)
        start_epoch = progress['epoch']
        if start_epoch > epochs:
            raise ValueError('The resumed epoch exceeds the training budget')
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch in loader:
            with accelerator.accumulate(model):
                with torch.no_grad(), accelerator.autocast():
                    target_logits = head(batch['target'])
                if stage == 1 and args.stage1_noise_std:
                    batch['hidden_states'] = batch['hidden_states'] + (
                        torch.rand_like(batch['hidden_states']) - 0.5
                    ) * args.stage1_noise_std * 512 / batch['hidden_states'].shape[-1]
                active = batch['loss_mask'].bool()
                use_vhe = stage == 2 and args.vhe and not args.dense
                if use_vhe:
                    with torch.no_grad(), accelerator.autocast():
                        selection_logits = head(forward(model, batch))
                    refinement_hidden = forward(model, batch, enhanced=True)
                    refinement_logits = head(refinement_hidden)
                else:
                    refinement_hidden = forward(model, batch)
                    refinement_logits = head(refinement_hidden)
                    selection_logits = refinement_logits.detach()
                if stage == 2 and not args.dense:
                    with torch.no_grad():
                        hard = selector(token_logprobs(target_logits, batch['labels']),
                                        token_logprobs(selection_logits, batch['labels']), active,
                                        update=accelerator.sync_gradients)
                        visual = batch['visual_static_mask'].bool()
                        if args.selection_mode == 'hard':
                            active = hard
                        elif args.selection_mode == 'visual':
                            active = visual & active
                        else:
                            active = critical_mask(hard, visual, active)
                if args.objective == 'distillation':
                    loss = distillation_loss(refinement_hidden, batch['target'], refinement_logits,
                                             target_logits, active, args.v_w, args.p_w)
                else:
                    loss = nll_loss(refinement_logits, batch['labels'], active, normalize=stage == 2)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        progress = dict(epoch=epoch + 1, stage=stage, objective=args.objective,
                        method=args.method, initialization=args.initialization,
                        training=dict(lr=lr, batch_size=bs,
                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                      warmup_steps=warmup, total_steps=total_steps,
                                      mixed_precision=args.mixed_precision,
                                      betas=[0.9, 0.95], grad_clip=args.grad_clip,
                                      v_w=args.v_w, p_w=args.p_w,
                                      vhe=stage == 2 and args.vhe and not args.dense,
                                      eta_q=args.eta_q, eta_a=args.eta_a,
                                      ema_momentum=args.ema_momentum,
                                      difficulty_scale=args.difficulty_scale,
                                      gamma=args.gamma, selection_mode=args.selection_mode))
        save_checkpoint(accelerator, model, config, progress, outdir / f'state_{epoch + 1}')
    accelerator.wait_for_everyone()
    result = accelerator.unwrap_model(model)
    accelerator.free_memory()
    return result


def main(family='llava', baseline=False, stage1_only=False):
    parser = argparse.ArgumentParser('CATS')
    parser.add_argument('--basepath', required=True)
    parser.add_argument('--configpath')
    parser.add_argument('--method', choices=['cats', 'sft', 'eagle2', 'all'],
                        default='sft' if baseline else 'cats')
    parser.add_argument('--benchmark', choices=['scienceqa', 'aokvqa', 'textvqa'])
    parser.add_argument('--stage1_datadir', '--tmpdir')
    parser.add_argument('--stage2_datadir', '--data_dir')
    parser.add_argument('--stage1_outdir', '--cpdir', default='checkpoints/shared/stage1')
    parser.add_argument('--stage2_outdir', '--output_dir')
    parser.add_argument('--sft_outdir', default='checkpoints/sft')
    parser.add_argument('--eagle2_outdir', default='checkpoints/eagle2')
    parser.add_argument('--stage1_epochs', type=int, default=20)
    parser.add_argument('--stage2_epochs', '--epochs', type=int, default=5)
    parser.add_argument('--stage1_samples', type=int, default=68000)
    parser.add_argument('--stage2_samples', type=int)
    parser.add_argument('--stage1_lr', type=float, default=3e-5)
    parser.add_argument('--stage2_lr', '--lr', type=float, default=8e-6)
    parser.add_argument('--stage1_bs', type=int, default=5)
    parser.add_argument('--stage2_bs', '--batch_size', type=int, default=1)
    parser.add_argument('--stage1_warmup_steps', type=int, default=2000)
    parser.add_argument('--stage1_total_steps', type=int, default=800000)
    parser.add_argument('--stage1_noise_std', type=float, default=0.2)
    parser.add_argument('--stage2_warmup_steps', '--warmup_steps', type=int)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--gradient-accumulation-steps', '--gradient_accumulation_steps',
                        dest='gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--mixed_precision', choices=['no', 'fp16', 'bf16'], default='bf16')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--grad_clip', type=float)
    parser.add_argument('--ema_momentum', type=float, default=0.99)
    parser.add_argument('--difficulty_scale', type=float, default=0.8)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--selection_mode', choices=['union', 'hard', 'visual'], default='union')
    parser.add_argument('--objective', choices=['nll', 'distillation'], default='distillation')
    parser.add_argument('--v_w', type=float, default=1.0)
    parser.add_argument('--p_w', type=float, default=0.1)
    parser.add_argument('--vhe', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eta_q', type=float, default=0.02)
    parser.add_argument('--eta_a', type=float, default=0.1)
    parser.add_argument('--only_stage1', action='store_true', default=stage1_only)
    parser.add_argument('--only_stage2', action='store_true', default=baseline)
    parser.add_argument('--resume_stage1_state')
    parser.add_argument('--resume_stage2_state')
    parser.add_argument('--init_stage2_from_stage1_state', '--init_from_stage1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eagle2', action='store_true')
    args = parser.parse_args()
    if args.eagle2:
        args.method = 'eagle2'
        args.only_stage1 = False
    if not 0 <= args.ema_momentum < 1 or not 0 < args.gamma <= 1 or args.difficulty_scale <= 0:
        parser.error('Invalid critical-token selector parameters')
    if args.only_stage1 and args.only_stage2:
        parser.error('Choose one training stage')
    needs_stage2 = not args.only_stage1 and args.method != 'eagle2'
    if needs_stage2 and not args.stage2_datadir:
        parser.error('--stage2_datadir is required for downstream refinement')
    if args.method == 'all' and args.resume_stage2_state:
        parser.error('Resume each method independently')
    if args.benchmark:
        sizes = dict(scienceqa=12726, aokvqa=17056, textvqa=34602)
        if args.stage2_samples is None:
            args.stage2_samples = sizes[args.benchmark]
    set_seed(args.seed)
    base = AutoConfig.from_pretrained(args.basepath)
    text = getattr(base, 'text_config', base)
    if args.configpath:
        config = EConfig.from_pretrained(args.configpath)
    else:
        values = text.to_dict()
        values['num_hidden_layers'] = 1
        values['architectures'] = base.architectures
        config = EConfig(**values)
    model = Model(config, load_emb=False)
    model.embed_tokens.weight.data.copy_(load_weight(args.basepath,
        ['language_model.model.embed_tokens.weight', 'model.embed_tokens.weight',
         'model.language_model.embed_tokens.weight']))
    head = build_lm_head(args.basepath)
    args.initialization = None
    args.dense = False
    if args.only_stage1:
        if not args.stage1_datadir:
            parser.error('--stage1_datadir is required for Stage I')
        args.method = 'stage1'
        args.grad_clip = args.grad_clip if args.grad_clip is not None else 0.5
        run_stage(args, 1, model, head, config)
        return
    foundation = Path(args.init_stage2_from_stage1_state) if args.init_stage2_from_stage1_state else (
        Path(args.stage1_outdir) / 'state_20')
    if not foundation.exists():
        if args.only_stage2 or not args.stage1_datadir:
            parser.error('Supply the shared epoch-20 Stage-I checkpoint')
        if args.stage1_epochs != 20:
            parser.error('The shared initialization must be trained for exactly 20 epochs')
        foundation_args = copy.copy(args)
        foundation_args.method = 'stage1'
        foundation_args.grad_clip = 0.5
        model = run_stage(foundation_args, 1, model, head, config)
    initialization = checkpoint_identity(foundation)
    if args.resume_stage2_state:
        previous = json.loads((Path(args.resume_stage2_state) / 'progress.json').read_text())
        if previous['initialization']['sha256'] != initialization['sha256']:
            raise ValueError('The resumed run uses a different shared checkpoint')
    del model
    methods = ['eagle2', 'sft', 'cats'] if args.method == 'all' else [args.method]
    for method in methods:
        branch = copy.copy(args)
        branch.method = method
        branch.initialization = initialization
        branch.dense = method == 'sft'
        branch.vhe = args.vhe and method == 'cats'
        branch.stage2_warmup_steps = args.stage2_warmup_steps if args.stage2_warmup_steps is not None else (
            100 if branch.dense else 500)
        branch.grad_clip = args.grad_clip if args.grad_clip is not None else (1.0 if branch.dense else 0.5)
        draft = Model(config, load_emb=False)
        load_draft(draft, foundation)
        if method == 'eagle2':
            if not args.stage1_datadir:
                parser.error('--stage1_datadir is required for EAGLE-2 continuation')
            branch.stage1_epochs = 25
            branch.stage1_outdir = args.eagle2_outdir
            branch.resume_stage1_state = str(foundation)
            run_stage(branch, 1, draft, head, config)
        else:
            if method == 'sft':
                branch.stage2_outdir = args.stage2_outdir if args.method != 'all' and args.stage2_outdir else args.sft_outdir
            else:
                branch.stage2_outdir = args.stage2_outdir or 'checkpoints/cats/stage2'
            run_stage(branch, 2, draft, head, config)


if __name__ == '__main__':
    main()
