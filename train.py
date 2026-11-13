"""Train one epoch on location-organized image directories."""
import argparse
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import U1652DatasetTrain, get_transforms
from model import CMMA
from sampling import ExtraDroneSampler, auxiliary_seed, isolated_aux_rng
from checkpoints import initialize_encoder, save_checkpoint, PRETRAINED_SHA256


def train(satellite, drone, seed=1, encoder=None, pretrained=None, output=None):
    if encoder is None and pretrained is None:
        raise ValueError('Supply the ImageNet checkpoint via pretrained, or an initialized encoder.')
    if encoder is not None and pretrained is not None:
        raise ValueError('Supply either encoder or pretrained, not both.')
    if output is not None and Path(output).exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not Path(satellite).is_dir() or not Path(drone).is_dir():
        raise ValueError('Both training image directories must exist.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = CMMA(encoder)
    if pretrained is not None:
        initialize_encoder(net.encoder, pretrained)
    net = net.to(device).train()
    _, ts, td = get_transforms((384, 384))
    ds = U1652DatasetTrain(satellite, drone, ts, td, prob_flip=.5, shuffle_batch_size=24)
    if len(ds.ids) < 24:
        raise ValueError('Need at least 24 distinct locations for a training batch.')
    if any(len(ds.gallery_dict[idx]['files']) < 2 for idx in ds.ids):
        raise ValueError('Each training location needs at least two drone images.')
    nominal = math.ceil(len(ds) / 24)
    ds.shuffle()
    if not len(ds):
        raise ValueError('Need at least 24 distinct locations for a training batch.')
    extra_sampler = ExtraDroneSampler(ds, 24, seed, extra_count=24)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    warm = nominal * .1

    def multiplier(step):
        if step < warm:
            return step / max(1., warm)
        return .5 * (1 + math.cos(math.pi * (step - warm) / max(1., nominal - warm)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, multiplier)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=1024)
    for step, (sat, drone_batch, ids, labels) in enumerate(DataLoader(ds, batch_size=24)):
        aux_seed = auxiliary_seed(seed, step)
        with isolated_aux_rng(aux_seed):
            extra, extra_ids, _ = extra_sampler.sample(ids, labels, step)
        if not torch.equal(labels, extra_ids):
            raise ValueError('Auxiliary image locations must match the main batch.')
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            s = net(sat.to(device))
            d = net(drone_batch.to(device))
            with isolated_aux_rng(aux_seed, use_cuda=amp):
                e = net(extra.to(device))
            loss = net.objective(s, d, e, labels.to(device), step+1)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_value_(net.parameters(), 100.)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        print(f'{step+1}/{len(ds)//24} loss={loss.item():.4f}', flush=True)
    net.eval()
    if output is not None:
        save_checkpoint(net, output, {'seed': seed, 'epochs': 1, 'lr': .001,
                                    'batch_size': 24, 'resolution': 384,
                                    'pretrained_sha256': PRETRAINED_SHA256 if pretrained else None})
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--satellite', required=True)
    parser.add_argument('--drone', required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pretrained', required=True, help='Local official ConvNeXt-Base ImageNet checkpoint')
    parser.add_argument('--output', required=True, help='New local checkpoint path; existing files are protected')
    args = parser.parse_args()
    train(args.satellite, args.drone, args.seed, pretrained=args.pretrained, output=args.output)
