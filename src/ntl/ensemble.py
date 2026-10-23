"""Probability alignment, conservative blending and bounded prompt grouping."""
import numpy as np


def validate_probabilities(probability, rows, classes):
    p = np.asarray(probability, dtype=np.float64)
    if (p.shape != (rows, classes) or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(1), 1.0, atol=1e-5, rtol=0)):
        raise ValueError('Invalid probability shape, values or normalization')
    return p


def softmax(values):
    z = np.asarray(values, dtype=np.float64)
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def align_probabilities(probability, actual_labels, expected_labels):
    if len(set(actual_labels)) != len(actual_labels) or set(actual_labels) != set(expected_labels):
        raise ValueError('Checkpoint label set differs from the competition')
    p = validate_probabilities(probability, len(probability), len(actual_labels))
    return p[:, [actual_labels.index(label) for label in expected_labels]]


def mix_candidates(baseline, candidates, weights):
    base = validate_probabilities(baseline, len(baseline), baseline.shape[1])
    if set(candidates) - set(weights) or any(w < 0 or not np.isfinite(w) for w in weights.values()):
        raise ValueError('Unknown candidate or invalid blend weight')
    if sum(weights.values()) > 1:
        raise ValueError('Candidate weights exceed the probability mass')
    used = {name: weights[name] for name in candidates}
    answer = (1.0 - sum(used.values())) * base
    for name, weight in used.items():
        answer += weight * validate_probabilities(candidates[name], len(base), base.shape[1])
    return validate_probabilities(answer, len(base), base.shape[1]), {'baseline':1-sum(used.values()), **used}


def sinkhorn(probability, iterations=100):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0)
    p /= p.sum(1, keepdims=True)
    if len(p) < 50:
        return p
    for _ in range(iterations):
        p *= (len(p) / p.shape[1]) / p.sum(0, keepdims=True)
        p /= p.sum(1, keepdims=True)
    return validate_probabilities(p, len(p), p.shape[1])


def content_groups(texts, threshold=0.85):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(texts) < 2:
        return np.arange(len(texts), dtype=np.int64)
    try:
        x = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts)
    except ValueError:
        return np.arange(len(texts), dtype=np.int64)
    distance = np.clip(1-(x @ x.T).toarray(), 0, 1)
    np.fill_diagonal(distance, 0)
    return fcluster(linkage(squareform(distance, checks=False), method='average'),
                    threshold, criterion='distance').astype(np.int64)


def checkpoint_groups_eligible(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return (counts[inverse] >= 2) & (counts[inverse] <= 18)


def selective_atlas_rows(probability, margin=0.3):
    p = validate_probabilities(probability, len(probability), probability.shape[1])
    if len(p) < 50:
        return np.arange(len(p))
    top = np.sort(p,axis=1)[:,-2:]
    return np.flatnonzero(top[:,1]-top[:,0] < margin)


def dense_group_hungarian(probability,groups):
    """Dhruv-style hard assignment only in detected groups of 12 through 18 rows."""
    from scipy.optimize import linear_sum_assignment
    p=validate_probabilities(probability,len(probability),probability.shape[1])
    groups=np.asarray(groups)
    if groups.shape!=(len(p),):raise ValueError('Prompt group row count differs')
    predictions=p.argmax(1).copy()
    decisions=[]
    for group in np.unique(groups):
        rows=np.flatnonzero(groups==group)
        if not 12<=len(rows)<=min(18,p.shape[1]):continue
        r,c=linear_sum_assignment(-np.log(np.clip(p[rows],1e-12,1)))
        changed=int((predictions[rows[r]]!=c).sum())
        predictions[rows[r]]=c
        decisions.append({'group':int(group),'size':len(rows),'changed_rows':changed})
    return predictions,{'method':'dhruv-style-dense-group-hungarian','minimum_group_size':12,
                        'maximum_group_size':18,'changed_rows':int((predictions!=p.argmax(1)).sum()),
                        'groups':decisions}


def merge_pools(frames, train, labels):
    import pandas as pd
    import re
    norm = lambda value: re.sub(r'\s+', ' ', str(value).strip())
    data = pd.concat([frame[['text','model']] for frame in frames],ignore_index=True)
    data = data.dropna(subset=['text','model'])
    data = data[data.model.isin(labels)].copy()
    data['text'] = data.text.map(norm)
    data = data[data.text.ne('')]
    conflict = data.groupby('text').model.nunique()
    conflicts = set(conflict[conflict>1].index)
    data = data[~data.text.isin(conflicts)].drop_duplicates(subset=['text'])
    data = data[~data.text.isin(set(train.text.map(norm)))].reset_index(drop=True)
    if data.empty or set(data.model) != set(labels):
        raise ValueError('External pool is empty or lacks competition classes')
    return data, {'rows':len(data), 'conflicting_texts_removed':len(conflicts), 'classes':len(labels)}
