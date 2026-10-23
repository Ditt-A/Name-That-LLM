"""Conservative final-label decoding; never modifies the model probabilities.

The gates below are experimental heuristics, not accuracy-calibrated thresholds.
Keep a whole group's baseline labels whenever any gate fails. In particular,
assignment stability is not evidence that the inferred prompt group is correct.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


SELECTIVE_HUNGARIAN_DEFAULTS = {
    'min_group_size': 12,
    'max_group_size': 18,
    'min_assigned_probability': 0.01,
    'min_relative_probability': 1.0 / 3.0,
    'max_mean_log_loss': 0.15,
    'min_assignment_gap': 0.15,
    'max_assigned_rank': 3,
}


def _checked_groups(values, rows, name):
    values = np.asarray(values)
    if values.shape != (rows,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f'{name} must contain one integer group ID per test row')
    return values


def _assignment_with_gap(cost):
    """Find the best and second-best *distinct* rectangular assignments.

Every alternative omits at least one edge of the optimum. Forbid each optimum
edge in turn and take the cheapest alternative. With n <= 18 this needs at most
19 small assignment solves, rather than enumerating 18! label permutations.
"""
    row_index, column_index = linear_sum_assignment(cost)
    if not np.array_equal(row_index, np.arange(len(cost))):
        raise ValueError('Assignment did not cover every group member')
    best_cost = float(cost[row_index, column_index].sum())
    second_cost = np.inf
    for row, column in zip(row_index, column_index):
        alternative = cost.copy()
        alternative[row, column] = np.inf
        try:
            alt_rows, alt_columns = linear_sum_assignment(alternative)
        except ValueError:
            continue  # no complete alternative, possible only in a degenerate case
        alt_cost = float(alternative[alt_rows, alt_columns].sum())
        second_cost = min(second_cost, alt_cost)
    gap = max(0.0, float(second_cost - best_cost))
    return column_index, gap


def selective_hungarian(probabilities, tfidf_groups, glm_groups, config=None):
    """Return final label indices and an audit of accepted/rejected groups.

Two groups agree only when both partitions give exactly the same member set.
Only dense groups with baseline label collisions are considered. A selected
group is applied atomically; reverting individual rows could break uniqueness.
The likelihood loss is relative to independent argmax, which cannot be improved
under that same independent probability objective by adding a constraint.
"""
    settings = dict(SELECTIVE_HUNGARIAN_DEFAULTS)
    if config:
        unknown = set(config) - set(settings)
        if unknown:
            raise ValueError(f'Unknown selective-Hungarian settings: {sorted(unknown)}')
        settings.update(config)
    minimum, maximum = settings['min_group_size'], settings['max_group_size']
    if (not isinstance(minimum, int) or not isinstance(maximum, int)
            or not 2 <= minimum <= maximum <= 18):
        raise ValueError('Group limits must be integers with 2 <= min <= max <= 18')
    if not 0 < settings['min_assigned_probability'] <= 1:
        raise ValueError('Assigned probability floor must be in (0, 1]')
    if not 0 < settings['min_relative_probability'] <= 1:
        raise ValueError('Relative probability floor must be in (0, 1]')
    if (not np.isfinite(settings['max_mean_log_loss']) or settings['max_mean_log_loss'] < 0
            or not np.isfinite(settings['min_assignment_gap']) or settings['min_assignment_gap'] <= 0):
        raise ValueError('Loss bound must be nonnegative and stability gap strictly positive')
    if not isinstance(settings['max_assigned_rank'], int) or settings['max_assigned_rank'] < 1:
        raise ValueError('Assigned rank limit must be a positive integer')

    probability = np.asarray(probabilities, dtype=np.float64)
    if (probability.ndim != 2 or probability.shape[0] == 0 or probability.shape[1] < 2
            or not np.isfinite(probability).all() or (probability < 0).any()
            or not np.allclose(probability.sum(1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError('Expected a finite, nonnegative, row-normalized probability matrix')
    rows, classes = probability.shape
    first = _checked_groups(tfidf_groups, rows, 'TF-IDF grouping')
    second = _checked_groups(glm_groups, rows, 'GLM grouping')
    baseline = probability.argmax(1)
    selected = baseline.copy()
    _, second_inverse, second_sizes = np.unique(second, return_inverse=True, return_counts=True)
    decisions = []
    for group_id in np.unique(first):
        members = np.flatnonzero(first == group_id)
        size = len(members)
        record = {'tfidf_group': int(group_id), 'rows': int(size), 'accepted': False,
                  'changed_rows': 0, 'reason': ''}
        if not minimum <= size <= min(maximum, classes):
            record['reason'] = 'group_size'
            decisions.append(record)
            continue
        second_labels = np.unique(second_inverse[members])
        if len(second_labels) != 1 or int(second_sizes[second_labels[0]]) != size:
            record['reason'] = 'grouping_disagreement'
            decisions.append(record)
            continue
        record['glm_group'] = int(second[members[0]])
        old = baseline[members]
        collisions = size - len(np.unique(old))
        record['baseline_duplicate_labels'] = int(collisions)
        if not collisions:
            record['reason'] = 'no_label_collision'
            decisions.append(record)
            continue
        group_probability = probability[members]
        cost = -np.log(np.clip(group_probability, 1e-12, 1.0))
        candidate, gap = _assignment_with_gap(cost)
        assigned = group_probability[np.arange(size), candidate]
        original = group_probability[np.arange(size), old]
        relative = assigned / original
        ranks = (group_probability > assigned[:, None]).sum(1) + 1
        mean_loss = float(np.log(np.maximum(original, 1e-300)
                                 / np.maximum(assigned, 1e-300)).mean())
        record.update({'assignment_gap': float(gap), 'mean_log_loss': mean_loss,
                       'min_assigned_probability': float(assigned.min()),
                       'min_relative_probability': float(relative.min()),
                       'max_assigned_rank': int(ranks.max()),
                       'candidate_changed_rows': int(np.count_nonzero(candidate != old))})
        if assigned.min() < settings['min_assigned_probability']:
            record['reason'] = 'assigned_probability_too_low'
        elif relative.min() < settings['min_relative_probability']:
            record['reason'] = 'individual_probability_loss'
        elif ranks.max() > settings['max_assigned_rank']:
            record['reason'] = 'assigned_label_rank'
        elif mean_loss > settings['max_mean_log_loss']:
            record['reason'] = 'group_probability_loss'
        elif gap < settings['min_assignment_gap']:
            record['reason'] = 'unstable_assignment'
        else:
            if len(np.unique(candidate)) != size:
                raise AssertionError('Accepted assignment repeated a label within its group')
            selected[members] = candidate
            record.update({'accepted': True, 'reason': 'accepted',
                           'changed_rows': int(np.count_nonzero(candidate != old))})
        decisions.append(record)

    reasons = {}
    for record in decisions:
        reasons[record['reason']] = reasons.get(record['reason'], 0) + 1
    report = {'method': 'exact-group-consensus-selective-assignment-v1',
              'settings': settings, 'rows': int(rows), 'classes': int(classes),
              'tfidf_groups': int(len(np.unique(first))),
              'glm_groups': int(len(np.unique(second))),
              'accepted_groups': sum(int(record['accepted']) for record in decisions),
              'changed_rows': int(np.count_nonzero(selected != baseline)),
              'group_reasons': reasons,
              'thresholds_accuracy_validated': False,
              'group_decisions': decisions}
    return selected, report
