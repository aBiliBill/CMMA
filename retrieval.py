import math
import torch

def directed_scores(qg, qt, gg, gt):
    """Existing global + directed local readout, fixed coefficients of one."""
    local = torch.einsum('qtc,gsc->qgts', qt, gt).max(-1).values.mean(-1)
    return qg @ gg.T + local

def per_positive_loss(scores, query_ids, gallery_ids, tau=.1):
    """Every positive competes only against other locations, then average queries."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('positive finite temperature required')
    mask = query_ids[:, None] == gallery_ids[None, :]
    if not bool((mask.any(1) & (~mask).any(1)).all()):
        raise ValueError('each query must have positive and negative candidates')
    rows = []
    for row, positive in zip(scores.float(), mask):
        diff = (row[~positive][None] - row[positive][:, None]) / tau
        rows.append(torch.logsumexp(torch.cat((torch.zeros_like(diff[:, :1]), diff), 1), 1).mean())
    return torch.stack(rows).mean()
