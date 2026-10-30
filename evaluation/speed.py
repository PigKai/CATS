import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.benchmark import summarize


def main():
    parser = argparse.ArgumentParser('CATS wall-clock speedup and acceptance length')
    parser.add_argument('results')
    args = parser.parse_args()
    groups = defaultdict(list)
    for line in Path(args.results).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            groups[str(row['temperature'])].append(row)
    if not groups:
        raise ValueError('No evaluation records')
    print(json.dumps({key: summarize(rows) for key, rows in groups.items()}, indent=2))


if __name__ == '__main__':
    main()
