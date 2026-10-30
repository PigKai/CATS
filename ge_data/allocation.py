import argparse
import os
import subprocess
import sys
from pathlib import Path


def split_range(start, end, count):
    size, extra = divmod(end - start, count)
    ranges = []
    for index in range(count):
        stop = start + size + (index < extra)
        ranges.append((start, stop))
        start = stop
    return ranges


def main():
    parser = argparse.ArgumentParser('CATS Stage-I feature generation')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--image_root')
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=68000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    script = Path(__file__).resolve().parents[1] / 'distillation/generate.py'
    processes = []
    for index, (gpu, (start, end)) in enumerate(zip(args.gpus, split_range(args.start, args.end, len(args.gpus)))):
        if start == end:
            continue
        command = [sys.executable, str(script), '--model_path', args.model_path,
                   '--dataset_path', args.dataset_path, '--outdir', args.outdir,
                   '--benchmark', 'llava_instruct', '--start', str(start), '--end', str(end),
                   '--index', str(index), '--seed', str(args.seed)]
        if args.image_root:
            command += ['--image_root', args.image_root]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        processes.append(subprocess.Popen(command, env=environment))
    outcomes = [process.wait() for process in processes]
    if any(outcomes):
        raise SystemExit(next(code for code in outcomes if code))


if __name__ == '__main__':
    main()
