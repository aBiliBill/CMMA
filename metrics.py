import numpy as np

def gallery_valid_mask(query_labels,gallery_labels,junk_policy='query_locations'):
    query_labels=np.asarray(query_labels);gallery_labels=np.asarray(gallery_labels)
    if query_labels.ndim!=1 or gallery_labels.ndim!=1 or not len(query_labels) or not len(gallery_labels):
        raise ValueError('Nonempty label vectors required')
    if np.any(query_labels<0):
        raise ValueError('Query labels cannot be unknown/junk')
    if junk_policy=='query_locations':
        mask=(gallery_labels>=0)&np.isin(gallery_labels,np.unique(query_labels))
    elif junk_policy=='minus_one_only':
        mask=gallery_labels>=0
    else:
        raise ValueError('Unknown junk policy')
    for label in np.unique(query_labels):
        if not np.any(mask&(gallery_labels==label)):
            raise ValueError('Missing positive gallery images for query label '+str(label))
    return mask

def metric_from_score_row(scores,query_label,gallery_labels,valid_mask):
    """Full sort BEFORE junk, multi-positive original trapezoid and standard AP.

    A missing positive is an error, never silently omitted from the denominator.
    The original compute_mAP would flag this query and its outer loop still use
    total-query denominator; production prevalidation prevents that ambiguity.
    """
    scores=np.asarray(scores);gallery_labels=np.asarray(gallery_labels);valid_mask=np.asarray(valid_mask)
    if scores.ndim!=1 or scores.shape!=gallery_labels.shape or valid_mask.shape!=scores.shape or valid_mask.dtype!=np.bool_:
        raise ValueError('One physical gallery score/label/mask per entry required')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores')
    full_order=np.argsort(scores)[::-1]
    order=full_order[valid_mask[full_order]]
    ranks=np.flatnonzero(gallery_labels[order]==query_label)+1
    if not len(ranks):
        raise ValueError('Missing positive after junk filtering')
    # Match the original loop's operation order, including trapezoidal recall.
    ap=0.0;ngood=len(ranks)
    for i,rank in enumerate(ranks):
        row=int(rank)-1
        precision=(i+1)*1.0/(row+1)
        old_precision=i*1.0/row if row!=0 else 1.0
        ap=ap+(1.0/ngood)*(old_precision+precision)/2
    standard=float(np.mean(np.arange(1,ngood+1,dtype=np.float64)/ranks))
    return {'positive_count':ngood,'positive_ranks':ranks.tolist(),'first_positive_rank':int(ranks[0]),
        'top1_gallery_index':int(order[0]),'top1_gallery_label':int(gallery_labels[order[0]]),
        'AP_trapezoid':float(ap),'AP_standard':standard,'effective_gallery_count':int(valid_mask.sum())}
