"""Fit one SVC temperature from prompt-grouped out-of-fold decision scores.

Only classifier heads are fitted. This module never loads or trains an LLM.
Known prompt aliases are connected through exact normalized duplicate texts.
Unknown semantic duplicates can remain; no claim of fully independent ensemble
validation is made. Temperature-fit NLL is not a held-out final-ensemble score.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


def normalize_text(value):
    return re.sub(r'\s+', ' ', str(value).strip())


def connected_prompt_groups(raw_frames, pool, train):
    parent = {}

    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    text_owner = {}
    frames = [(f'external-{i}', frame, 'prompt_key') for i, frame in enumerate(raw_frames)]
    frames.append(('official', train, 'prompt_id'))
    for namespace, frame, column in frames:
        if column not in frame or frame[column].isna().any():
            raise ValueError('Prompt IDs are required for grouped SVC calibration')
        for text, prompt in zip(frame.text, frame[column]):
            key = namespace + ':' + str(prompt)
            find(key)
            normalized = normalize_text(text)
            if normalized in text_owner:
                union(key, text_owner[normalized])
            else:
                text_owner[normalized] = key
    texts = pool.text.tolist() + train.text.tolist()
    groups = np.asarray([find(text_owner[normalize_text(text)]) for text in texts], dtype=str)
    if len(np.unique(groups)) < 3:
        raise ValueError('At least three independent known prompt components are required')
    return groups


def fit_temperature(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=int)
    if (scores.ndim != 2 or len(scores) != len(targets) or not len(scores)
            or not np.isfinite(scores).all()
            or (targets < 0).any() or (targets >= scores.shape[1]).any()):
        raise ValueError('Invalid temperature calibration input')

    def loss(log_temperature):
        scaled = scores / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))

    result = minimize_scalar(loss, bounds=(np.log(0.25), np.log(8.0)), method='bounded',
                             options={'maxiter': 64, 'xatol': 1e-5})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Temperature optimization did not converge')
    original = loss(0.0)
    chosen = float(result.x) if result.fun <= original else 0.0
    return float(np.exp(chosen)), {'nll_before': original, 'nll_after_fit': loss(chosen),
                                  'fit_rows': len(targets), 'optimizer_iterations': int(result.nfev)}


def fit_svc_temperature(matrix, targets, pool, train, raw_frames, labels, output_dir=None):
    targets = np.asarray(targets, dtype=int)
    groups = connected_prompt_groups(raw_frames, pool, train)
    if matrix.shape[0] != len(groups) or len(targets) != len(groups):
        raise ValueError('SVC calibration row order differs')
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.full((len(targets), len(labels)), np.nan, dtype=np.float64)
    folds = np.full(len(targets), -1, dtype=int)
    reports = []
    base_weight = np.r_[np.full(len(pool), 0.5), np.ones(len(train))]
    for fold, (fitting, validation) in enumerate(splitter.split(matrix, targets, groups)):
        overlap = set(groups[fitting]) & set(groups[validation])
        if overlap:
            raise AssertionError('Prompt component crossed a calibration fold')
        weight = base_weight[fitting].copy()
        totals = np.bincount(targets[fitting], weights=weight, minlength=len(labels))
        if (totals <= 0).any():
            raise ValueError('A calibration fold is missing a model class')
        weight *= totals.mean() / totals[targets[fitting]]
        model = LinearSVC(C=0.5, max_iter=3000, random_state=42)
        model.fit(matrix[fitting], targets[fitting], sample_weight=weight)
        if model.classes_.tolist() != list(range(len(labels))):
            raise ValueError('Calibration class order differs')
        oof[validation] = model.decision_function(matrix[validation])
        folds[validation] = fold
        report = {'fold': fold, 'fit_rows': len(fitting), 'validation_rows': len(validation),
                  'known_prompt_overlap': len(overlap)}
        reports.append(report)
        print('SVC calibration fold', fold + 1, '/3 complete', report, flush=True)
        del model
    if not np.isfinite(oof).all() or (folds < 0).any():
        raise AssertionError('Out-of-fold calibration is incomplete')
    official = np.arange(len(pool), len(targets))
    temperature, loss_report = fit_temperature(oof[official], targets[official])
    report = {'method': 'single-temperature-on-official-rows-from-three-prompt-grouped-SVC-folds',
              'temperature': temperature, 'folds': reports, **loss_report,
              'known_prompt_components': int(len(np.unique(groups))),
              'known_prompt_overlap': 0, 'qwen_or_lora_training': False,
              'scope': 'new-SVC-only; baseline-Qwen-ensemble-is-not-calibrated',
              'nll_after_fit_is_independent_validation': False,
              'unidentified_semantic_prompt_duplicates_may_remain': True,
              'temperature_transferred_from_fold_estimators_to_full_SVC': True,
              'labels': list(labels),
              'group_order_sha256': hashlib.sha256(json.dumps(groups.tolist()).encode()).hexdigest()}
    if output_dir is not None:
        directory = Path(output_dir)
        (directory / 'svc-calibration-report.json').write_text(json.dumps(report, indent=2))
        np.savez_compressed(directory / 'svc-calibration-oof.npz',
                            official_scores=oof[official], targets=targets[official],
                            folds=folds[official], groups=groups[official],
                            ids=train.id.astype(str).to_numpy(dtype=str), labels=np.asarray(labels))
    return temperature, report
