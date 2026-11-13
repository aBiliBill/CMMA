"""CMMA: shared ConvNeXt-Base, training-time MCFA and directional CLM."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

from backbone import ConvNeXt
from context import Mutil_module
from retrieval import directed_scores, per_positive_loss


def describe(global_vector, feature_map, norm):
    g = F.normalize(global_vector, dim=-1).float()
    tokens = F.adaptive_avg_pool2d(feature_map.float(), (4, 4)).flatten(2).transpose(1, 2)
    with torch.autocast(g.device.type, enabled=False):
        tokens = F.layer_norm(tokens, norm.normalized_shape, norm.weight.float(),
                              norm.bias.float(), norm.eps)
        t = F.normalize(tokens, dim=-1)
        direction = F.normalize(g, dim=-1)
        t = F.normalize(t, dim=-1)
        residual = t - (t * direction[:, None]).sum(-1, keepdim=True) * direction[:, None]
        return g, F.normalize(residual, dim=-1)


class CMMA(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = (encoder if encoder is not None else
                        ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]))
        self.context = Mutil_module(1024)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))

    def forward(self, images):
        if self.training and images.is_cuda and torch.is_grad_enabled():
            chunks = [checkpoint(self.encoder, x, use_reentrant=False)
                      for x in images.split(4)]
            pooled, feature_map = (torch.cat(group) for group in zip(*chunks))
        else:
            pooled, feature_map = self.encoder(images)
        g, t = describe(pooled, feature_map, self.encoder.norm)
        aligned = None
        if self.training:
            context = self.context(feature_map).flatten(2)
            context = F.normalize(context, dim=1).softmax(dim=2)
            aligned = torch.cat((feature_map.flatten(2), context), dim=1).flatten(1)
        return g, t, aligned

    def objective(self, sat, drone, extra, labels, step):
        sg, st, sa = sat
        dg, dt, da = drone
        eg, et, _ = extra
        targets = torch.arange(len(labels), device=labels.device)
        logits = (self.logit_scale.exp() * sg) @ dg.T
        global_loss = (F.cross_entropy(logits, targets, label_smoothing=.1)
                       + F.cross_entropy(logits.T, targets, label_smoothing=.1)) / 2
        align_loss = 1 - (F.normalize(sa, dim=1) * F.normalize(da, dim=1)).sum() / len(labels)
        with torch.autocast(sg.device.type, enabled=False):
            gd, td = torch.cat((dg, eg)), torch.cat((dt, et))
            ids = labels.repeat(2)
            rank = (per_positive_loss(directed_scores(sg, st, gd, td), labels, ids)
                    + per_positive_loss(directed_scores(gd, td, sg, st), ids, labels)) / 2
        return global_loss + .5 * align_loss + .1 * min(step / 10, 1.) * rank
