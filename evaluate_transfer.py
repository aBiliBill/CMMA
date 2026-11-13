"""Cross-dataset retrieval with fixed source checkpoints and no target training."""
import argparse
import json
import math
from pathlib import Path

import torch

from checkpoints import load_checkpoint
from evaluate import evaluate


HEIGHTS = (150, 200, 250, 300)
DIRECTIONS = {
    'D2S': ('query_drone', 'gallery_satellite'),
    'S2D': ('query_satellite', 'gallery_drone'),
}


def evaluate_both(net, test_directory, *, batch_size=16, amp=False):
    """Recompute each direction; local matching is directional, not a transpose."""
    base = Path(test_directory)
    results = {direction: evaluate(net, base / query, base / gallery,
                                   batch_size=batch_size, amp=amp)
               for direction, (query, gallery) in DIRECTIONS.items()}
    for metrics in results.values():
        for metric in ('R@1', 'AP'):
            value = float(metrics[metric])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError('Metrics must be finite percentages in [0, 100]')
    return results


def mean_metrics(rows):
    """Equal mean of four independent models' metrics, before display rounding."""
    if len(rows) != 4 or {row['source_height_m'] for row in rows} != set(HEIGHTS):
        raise ValueError('Exactly one result for each of the four source heights is required')
    result = {}
    for direction in DIRECTIONS:
        result[direction] = {}
        for metric in ('R@1', 'AP'):
            values = [float(row[direction][metric]) for row in rows]
            if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
                raise ValueError('Metrics must be finite percentages in [0, 100]')
            result[direction][metric] = math.fsum(values) / len(values)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='transfer', required=True)
    sues = modes.add_parser('sues-to-university', help='Four SUES source models on one University test set')
    sues.add_argument('--university-test', required=True, help='University-1652/test directory')
    for height in HEIGHTS:
        sues.add_argument(f'--checkpoint-{height}', required=True,
                          help=f'Final public-format model trained on SUES at {height} m')
    university = modes.add_parser('university-to-sues', help='One University source model on four SUES test sets')
    university.add_argument('--sues-test', required=True, help='SUES-200/Testing directory containing all four heights')
    university.add_argument('--checkpoint', required=True, help='Final public-format University-trained model')
    for command in (sues, university):
        command.add_argument('--device', choices=('cpu', 'cuda'),
                             default='cuda' if torch.cuda.is_available() else 'cpu')
        command.add_argument('--precision', choices=('fp32', 'amp'), default=None,
                             help='Feature precision; default AMP on CUDA, FP32 on CPU. Scores use FP32.')
        command.add_argument('--batch-size', type=int, default=16)
    return parser


def run(args):
    precision = args.precision or ('amp' if args.device == 'cuda' else 'fp32')
    if args.batch_size < 1:
        raise ValueError('batch_size must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    if precision == 'amp' and args.device != 'cuda':
        raise ValueError('AMP requires --device cuda; use --precision fp32 on CPU')
    amp = precision == 'amp'
    if args.transfer == 'sues-to-university':
        checkpoints = [Path(getattr(args, f'checkpoint_{height}')).resolve() for height in HEIGHTS]
        if len(set(checkpoints)) != len(HEIGHTS):
            raise ValueError('Supply a separate final source checkpoint for each height')
        bases = [Path(args.university_test)] * len(HEIGHTS)
    else:
        checkpoints = [Path(args.checkpoint).resolve()]
        bases = [Path(args.sues_test) / str(height) for height in HEIGHTS]
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
    for base in bases:
        for query, gallery in DIRECTIONS.values():
            for role in (query, gallery):
                if not (base / role).is_dir():
                    raise FileNotFoundError(base / role)

    torch.backends.cuda.matmul.allow_tf32 = False
    output = {
        'transfer': args.transfer, 'target_training': False,
        'feature_precision': precision, 'score_precision': 'fp32',
        'AP_definition': 'original trapezoidal', 'gallery_policy': 'query_locations',
        'per_height': [],
    }
    options = dict(batch_size=args.batch_size, amp=amp)
    if args.transfer == 'sues-to-university':
        for height, checkpoint, base in zip(HEIGHTS, checkpoints, bases):
            net = load_checkpoint(checkpoint, args.device)
            metrics = evaluate_both(net, base, **options)
            output['per_height'].append({'source_height_m': height, **metrics})
            del net
        output['aggregation'] = 'arithmetic mean of four independent source-model metrics; no ensemble'
        output['mean'] = mean_metrics(output['per_height'])
    else:
        net = load_checkpoint(checkpoints[0], args.device)
        for height, base in zip(HEIGHTS, bases):
            metrics = evaluate_both(net, base, **options)
            output['per_height'].append({'target_height_m': height, **metrics})
        del net
    return output


def main():
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
