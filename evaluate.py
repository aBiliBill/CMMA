"""Evaluate one retrieval direction using a model in memory or a local checkpoint."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from data import get_transforms
from metrics import gallery_valid_mask, metric_from_score_row
from retrieval import directed_scores


class EvaluationImages(Dataset):
    def __init__(self, directory):
        self.files = [(image, int(place.name)) for place in sorted(Path(directory).iterdir())
                      if place.is_dir() for image in sorted(place.iterdir())
                      if image.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
        if not self.files:
            raise ValueError(f'No evaluation images in {directory}')
        self.transform = get_transforms((384, 384))[0]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label = self.files[index]
        pixels = cv2.imread(str(image))
        if pixels is None:
            raise ValueError(f'Cannot read {image}')
        rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        return self.transform(image=rgb)['image'], label


@torch.no_grad()
def extract(net, directory, batch_size=16, *, amp=False):
    net.eval()
    device = next(net.parameters()).device
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if amp and device.type != 'cuda':
        raise ValueError('AMP feature extraction requires a CUDA device')
    batches = []
    for images, labels in DataLoader(EvaluationImages(directory), batch_size=batch_size):
        with torch.autocast(device.type, enabled=amp):
            global_desc, local_desc, _ = net(images.to(device))
        batches.append((global_desc.float().cpu(), local_desc.float().cpu(), labels))
    return tuple(torch.cat(group) for group in zip(*batches))


@torch.no_grad()
def evaluate(net, query_directory, gallery_directory, *, batch_size=16, amp=False):
    qg, qt, qids = extract(net, query_directory, batch_size, amp=amp)
    gg, gt, gids = extract(net, gallery_directory, batch_size, amp=amp)
    mask = gallery_valid_mask(qids.numpy(), gids.numpy())
    device = next(net.parameters()).device
    hits, aps = [], []
    for begin in range(0, len(qids), 8):
        q, t = qg[begin:begin+8].to(device), qt[begin:begin+8].to(device)
        parts = [directed_scores(q, t, gg[j:j+128].to(device), gt[j:j+128].to(device)).cpu()
                 for j in range(0, len(gids), 128)]
        for i, row in enumerate(torch.cat(parts, 1).numpy()):
            m = metric_from_score_row(row, int(qids[begin+i]), gids.numpy(), mask)
            hits.append(m['first_positive_rank']==1)
            aps.append(m['AP_trapezoid'])
    return {'R@1': float(100*np.mean(hits)), 'AP': float(100*np.mean(aps))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--gallery', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--amp', action='store_true', help='CUDA AMP feature extraction; scoring remains FP32')
    parser.add_argument('--device', choices=('cpu', 'cuda'),
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    from checkpoints import load_checkpoint
    torch.backends.cuda.matmul.allow_tf32 = False
    net = load_checkpoint(args.checkpoint, args.device)
    print(json.dumps(evaluate(net, args.query, args.gallery,
                              batch_size=args.batch_size, amp=args.amp), indent=2))
