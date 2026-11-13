"""Local initialization and checkpoint I/O for the public CMMA implementation."""
import hashlib
from pathlib import Path

import torch

from model import CMMA


PRETRAINED_FILENAME = 'convnext_base_22k_1k_224.pth'
PRETRAINED_SHA256 = '8afbd1fa254629e6db80df9822908a83c741bfe98778f58851a15c1639a80453'
FORMAT = 'cmma-code-v1'
# Accept earlier public checkpoints; state-dictionary keys are unchanged.
SUPPORTED_FORMATS = {FORMAT, 'mccm-code-v1'}


def initialize_encoder(encoder, path):
    """Load the original ImageNet-22K -> 1K ConvNeXt-Base checkpoint."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != PRETRAINED_SHA256:
        raise ValueError('Pretrained checkpoint SHA256 mismatch. See README.md for the exact model.')
    state = torch.load(path, map_location='cpu', weights_only=True)
    encoder.load_state_dict(state['model'], strict=True)


def save_checkpoint(net, path, training):
    """Save a new local result; never overwrite an existing checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': FORMAT,
               'model': {key: value.detach().cpu() for key, value in net.state_dict().items()},
               'training': dict(training)}
    with path.open('xb') as stream:
        torch.save(payload, stream)


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') not in SUPPORTED_FORMATS:
        raise ValueError('Expected a checkpoint produced by this release of train.py.')
    net = CMMA()
    net.load_state_dict(payload['model'], strict=True)
    return net.to(device).eval()
