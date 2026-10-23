"""MRF-inspired finite cannot-link penalties on the final ensemble probabilities.

This is an adaptation of Basu, Bilenko and Mooney (KDD 2004), not a reproduction
of HMRF-KMEANS. The graph is a set of disjoint weighted cliques formed by the
intersections of two existing prompt partitions. A uniform pair penalty inside
each clique permits exact MAP decoding with repeated label slots.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


SOFT_MRF_DEFAULTS = {
    'penalty_strength': 0.40,
    'min_group_size': 2,
    'max_parent_size': 18,
    'min_group_confidence': 0.50,
    'min_energy_improvement': 0.05,
    'min_configuration_gap': 0.05,
    'min_assigned_probability': 0.01,
    'min_relative_probability': 1.0 / 3.0,
    'max_mean_log_loss': 0.15,
    'max_assigned_rank': 3,
}


def _soft_energy(unary, labels, penalty):
    counts = np.bincount(labels, minlength=unary.shape[1])
    return float(unary[np.arange(len(labels)), labels].sum()
                 + penalty * (counts * (counts - 1) / 2).sum())


def _solve_soft_clique(unary, penalty):
    """Exact optimum and next distinct *label configuration*, not slot permutation.

Giving the j-th use of label c a cost penalty*j realizes the total pair penalty
penalty*count_c*(count_c-1)/2. Every row can use every label, so duplicates remain
possible. Forbid all slots of a selected label for each row to find the cheapest
distinct labeling. At most n+1 small assignment solves for n <= 18.
"""
    rows, classes = unary.shape
    slot_labels = np.repeat(np.arange(classes), rows)
    extra_cost = penalty * np.tile(np.arange(rows), classes)
    cost = unary[:, slot_labels] + extra_cost[None, :]
    row_index, slot_index = linear_sum_assignment(cost)
    if not np.array_equal(row_index, np.arange(rows)):
        raise AssertionError('Soft assignment failed to cover every group row')
    chosen = slot_labels[slot_index]
    optimum = _soft_energy(unary, chosen, penalty)
    if not np.isclose(float(cost[row_index, slot_index].sum()), optimum, atol=1e-9, rtol=0):
        raise AssertionError('Repeated label slots disagree with the pairwise objective')
    second = np.inf
    for row, label in enumerate(chosen):
        alternative = cost.copy()
        alternative[row, slot_labels == label] = np.inf
        alt_rows, alt_slots = linear_sum_assignment(alternative)
        second = min(second, float(alternative[alt_rows, alt_slots].sum()))
    return chosen, optimum, max(0.0, float(second - optimum))


def soft_constraint_decode(probabilities, tfidf_groups, glm_groups, config=None):
    """Return guarded MAP labels, retaining probabilities and rejected blocks intact."""
    settings = dict(SOFT_MRF_DEFAULTS)
    if config:
        unknown = set(config) - set(settings)
        if unknown:
            raise ValueError(f'Unknown soft-MRF settings: {sorted(unknown)}')
        settings.update(config)
    for name in ['penalty_strength', 'min_energy_improvement', 'min_configuration_gap',
                 'max_mean_log_loss']:
        if not np.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if settings['min_configuration_gap'] <= 0:
        raise ValueError('The configuration gap must be positive to reject tied labelings')
    for name in ['min_group_confidence', 'min_assigned_probability', 'min_relative_probability']:
        if not 0 < settings[name] <= 1:
            raise ValueError(f'{name} must be in (0,1]')
    if (not isinstance(settings['min_group_size'], int)
            or not isinstance(settings['max_parent_size'], int)
            or not 2 <= settings['min_group_size'] <= settings['max_parent_size'] <= 18):
        raise ValueError('Expected 2 <= min_group_size <= max_parent_size <= 18')
    if not isinstance(settings['max_assigned_rank'], int) or settings['max_assigned_rank'] < 1:
        raise ValueError('The rank limit must be a positive integer')
    probability = np.asarray(probabilities, dtype=np.float64)
    if (probability.ndim != 2 or len(probability) == 0 or probability.shape[1] < 2
            or not np.isfinite(probability).all() or (probability < 0).any()
            or not np.allclose(probability.sum(1), 1, atol=1e-6, rtol=0)):
        raise ValueError('Expected finite, nonnegative, normalized probabilities')
    rows, classes = probability.shape
    partitions = []
    sizes = []
    for name, values in [('tfidf', tfidf_groups), ('glm', glm_groups)]:
        values = np.asarray(values)
        if values.shape != (rows,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f'{name} grouping must contain one integer ID per test row')
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        partitions.append(inverse)
        sizes.append(counts[inverse])
    intersections, block_index = np.unique(np.column_stack(partitions), axis=0, return_inverse=True)
    baseline = probability.argmax(1)
    selected = baseline.copy()
    decisions = []
    for block in range(len(intersections)):
        members = np.flatnonzero(block_index == block)
        count = len(members)
        parent_a, parent_b = int(sizes[0][members[0]]), int(sizes[1][members[0]])
        confidence = count / (parent_a + parent_b - count)
        penalty = float(settings['penalty_strength'] * confidence)
        record = {'block':int(block), 'rows':int(count), 'tfidf_parent_rows':parent_a,
                  'glm_parent_rows':parent_b, 'confidence':float(confidence),
                  'pair_penalty':penalty, 'accepted':False, 'changed_rows':0}
        if count < settings['min_group_size']:
            record['reason'] = 'singleton_or_small_intersection'
        elif max(parent_a, parent_b) > settings['max_parent_size']:
            record['reason'] = 'oversized_parent_group'
        elif confidence < settings['min_group_confidence']:
            record['reason'] = 'low_grouping_confidence'
        elif penalty == 0:
            record['reason'] = 'zero_penalty'
        else:
            old = baseline[members]
            collisions = count - len(np.unique(old))
            record['baseline_duplicate_labels'] = int(collisions)
            if not collisions:
                record['reason'] = 'no_label_collision'
                decisions.append(record)
                continue
            group_probability = probability[members]
            unary = -np.log(np.clip(group_probability, 1e-12, 1))
            candidate, energy, gap = _solve_soft_clique(unary, penalty)
            baseline_energy = _soft_energy(unary, old, penalty)
            improvement = baseline_energy - energy
            if improvement < -1e-8:
                raise AssertionError('Exact soft assignment increased the objective')
            assigned = group_probability[np.arange(count), candidate]
            original = group_probability[np.arange(count), old]
            relative = assigned / original
            ranks = (group_probability > assigned[:, None]).sum(1) + 1
            mean_loss = float(np.log(np.maximum(original, 1e-300)
                                     / np.maximum(assigned, 1e-300)).mean())
            changed = int(np.count_nonzero(candidate != old))
            record.update({'baseline_energy':baseline_energy, 'candidate_energy':energy,
                           'energy_improvement':float(improvement), 'configuration_gap':float(gap),
                           'min_assigned_probability':float(assigned.min()),
                           'min_relative_probability':float(relative.min()),
                           'max_assigned_rank':int(ranks.max()), 'mean_log_loss':mean_loss,
                           'candidate_changed_rows':changed,
                           'candidate_duplicate_labels':int(count - len(np.unique(candidate)))})
            if not changed:
                record['reason'] = 'baseline_is_soft_optimum'
            elif improvement < settings['min_energy_improvement']:
                record['reason'] = 'insufficient_energy_improvement'
            elif gap < settings['min_configuration_gap']:
                record['reason'] = 'ambiguous_label_configuration'
            elif assigned.min() < settings['min_assigned_probability']:
                record['reason'] = 'assigned_probability_too_low'
            elif relative.min() < settings['min_relative_probability']:
                record['reason'] = 'individual_probability_loss'
            elif ranks.max() > settings['max_assigned_rank']:
                record['reason'] = 'assigned_label_rank'
            elif mean_loss > settings['max_mean_log_loss']:
                record['reason'] = 'group_probability_loss'
            else:
                selected[members] = candidate
                record.update({'accepted':True, 'reason':'accepted', 'changed_rows':changed})
        decisions.append(record)
    reasons = {}
    for item in decisions:
        reasons[item['reason']] = reasons.get(item['reason'], 0) + 1
    report = {'method':'weighted-cannot-link-mrf-exact-clique-map-v1',
              'paper':'https://www.cs.utexas.edu/~ml/papers/semi-kdd-04.pdf',
              'settings':settings, 'rows':int(rows), 'classes':int(classes),
              'intersection_blocks':int(len(intersections)),
              'accepted_blocks':sum(int(item['accepted']) for item in decisions),
              'changed_rows':int(np.count_nonzero(selected != baseline)),
              'block_reasons':reasons, 'thresholds_accuracy_validated':False,
              'full_hmrf_kmeans_reproduction':False, 'block_decisions':decisions}
    return selected, report
