"""Small synthetic end-to-end method check. No external inputs or network access."""
import torch
from model import CMMA, directed_scores
from metrics import gallery_valid_mask, metric_from_score_row


def main():
    torch.manual_seed(1)
    torch.set_num_threads(2)
    net = CMMA()
    labels = torch.arange(2)
    outputs = [net(torch.randn(2, 3, 64, 64)) for _ in range(3)]
    loss = net.objective(*outputs, labels, 1)
    loss.backward()
    assert torch.isfinite(loss)
    assert net.context.FC11.weight.grad is not None
    net.eval()
    with torch.no_grad():
        g, t, a = net(torch.randn(2, 3, 64, 64))
        assert a is None and t.shape==(2,16,1024)
        assert float((t*g[:,None]).sum(-1).abs().max()) < 1e-4
        assert directed_scores(g,t,g,t).shape==(2,2)
    mask = gallery_valid_mask([1], [1, 2])
    m = metric_from_score_row([0.4, 0.8], 1, [1, 2], mask)
    assert m['first_positive_rank']==1 and m['AP_trapezoid']==1
    print('PASS: forward, backward, directional scoring and effective-gallery AP')


if __name__=='__main__':
    main()
