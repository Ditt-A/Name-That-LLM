"""Checkpoint-bound train features and fitted Gaussian parameters; never cache test predictions."""
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

FORMAT = 1
_STORE = None
_IDENTITIES = {}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        return digest_json(value.tolist())
    result = hashlib.sha256()
    result.update(str(value.dtype).encode()); result.update(str(value.shape).encode())
    result.update(memoryview(value).cast('B'))
    return result.hexdigest()


def train_signature(frame):
    columns = ['id', 'prompt_id', 'model', 'text']
    if not all(column in frame for column in columns):
        raise ValueError('Official train columns are incomplete')
    return digest_json([[str(v) for v in row] for row in frame[columns].itertuples(index=False, name=None)])


def runtime_versions():
    result = {}
    for name in ['numpy', 'scipy', 'scikit-learn', 'torch', 'transformers']:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return result


def checkpoint_identity(base, adapter=None):
    """Bind inputs to metadata/tokenizers/heads/adapters and base shard names/sizes.

    Kaggle input references are recorded separately. Full multi-GB base shards are
    not rehashed on every inference; adapter/head tensors are SHA256 checked.
    """
    base = Path(base).resolve()
    adapter = Path(adapter).resolve() if adapter is not None else None
    key = (str(base), str(adapter))
    if key in _IDENTITIES:
        return _IDENTITIES[key]
    if not (base / 'config.json').is_file():
        raise FileNotFoundError('Base config missing')
    small = {}
    names = ['config.json', 'model.safetensors.index.json', 'pytorch_model.bin.index.json',
             'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'tokenizer.model', 'spm.model', 'vocab.json', 'merges.txt', 'added_tokens.json']
    for name in names:
        if (base / name).is_file():
            small[name] = file_sha(base / name)
    weights = {}
    for name in ['model.safetensors.index.json', 'pytorch_model.bin.index.json']:
        path = base / name
        if path.is_file():
            shards = sorted(set(json.loads(path.read_text())['weight_map'].values()))
            for shard in shards:
                path = base / shard
                if not path.is_file():
                    raise FileNotFoundError('A base weight shard is missing')
                weights[shard] = path.stat().st_size
    if not weights:
        for name in ['model.safetensors', 'pytorch_model.bin']:
            if (base / name).is_file():
                weights[name] = (base / name).stat().st_size
    if not weights:
        raise FileNotFoundError('Refusing a config-only model folder')
    ad = {}
    if adapter is not None:
        for name in ['meta.json', 'heads.pt', 'heads.safetensors', 'adapter_config.json',
                     'adapter_model.safetensors', 'adapter/adapter_config.json',
                     'adapter/adapter_model.safetensors']:
            if (adapter / name).is_file():
                ad[name] = file_sha(adapter / name)
        if not any(name.endswith('adapter_model.safetensors') for name in ad):
            raise FileNotFoundError('An actual LoRA adapter is required')
        if not any(name in ad for name in ['heads.pt', 'heads.safetensors']):
            raise FileNotFoundError('An actual pretrained classifier head is required')
    result = {'base_metadata_sha256': small, 'base_weight_bytes': weights, 'adapter_sha256': ad}
    _IDENTITIES[key] = result
    return result


def _jsonable(value):
    if isinstance(value, dict): return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


class TrainCache:
    def __init__(self, root, contract, train, write=False):
        self.root = Path(root)
        self.manifest_path = self.root / 'train-cache-manifest.json'
        self.write = write
        self.contract = contract
        self.train_sha = train_signature(train)
        self.events = []
        if write: self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            if (self.manifest['format'] != FORMAT or self.manifest['contract'] != contract or
                    self.manifest['train_sha256'] != self.train_sha):
                raise ValueError('Train cache source contract or official training data differs')
            if self.manifest['runtime_versions'] != runtime_versions():
                raise ValueError('Cache preparation/inference software versions differ')
            if not write and self.manifest['status'] != 'COMPLETE':
                raise ValueError('Incomplete train cache is not usable for inference')
        elif write:
            self.manifest = {'format': FORMAT, 'contract': contract, 'train_sha256': self.train_sha,
                             'train_rows': len(train), 'status': 'BUILDING', 'features': {}, 'fits': {},
                             'runtime_versions': runtime_versions(),
                             'parity': {}, 'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            self.flush()
        else:
            raise FileNotFoundError('Verified train-cache-manifest.json is required')

    def flush(self):
        if not self.write: return
        target = self.manifest_path.with_suffix('.tmp')
        target.write_text(json.dumps(_jsonable(self.manifest), indent=2), encoding='utf-8')
        target.replace(self.manifest_path)

    def save_features(self, key, arrays, recipe, identity, runtime=None):
        if not self.write: raise RuntimeError('Read-only cache')
        for value in arrays.values():
            if value.ndim != 2 or len(value) != self.manifest['train_rows'] or not np.isfinite(value).all():
                raise ValueError('Invalid train feature matrix')
        filename = 'features-' + key + '.npz'
        path = self.root / filename
        np.savez_compressed(path, **arrays)
        self.manifest['features'][key] = {'file': filename, 'sha256': file_sha(path),
            'recipe': _jsonable(recipe), 'checkpoint': identity,
            'runtime': runtime or {},
            'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': array_sha(v)} for k, v in arrays.items()}}
        self.flush()
        loaded = self.load_features(key, recipe, identity)
        if not all(np.array_equal(loaded[k], value) for k, value in arrays.items()):
            raise ValueError('Train feature serialization changed values')
        self.manifest['parity']['features-' + key] = {'passed': True, 'method': 'exact-array-round-trip'}
        self.flush()

    def load_features(self, key, recipe, identity):
        record = self.manifest['features'][key]
        if record['recipe'] != _jsonable(recipe) or record['checkpoint'] != identity:
            raise ValueError('Cached feature recipe/checkpoint differs: ' + key)
        path = self.root / record['file']
        if file_sha(path) != record['sha256']: raise ValueError('Changed feature artifact: ' + key)
        with np.load(path, allow_pickle=False) as pack:
            arrays = {name: pack[name] for name in pack.files}
        if set(arrays) != set(record['arrays']): raise ValueError('Feature parts differ')
        for name, value in arrays.items():
            meta = record['arrays'][name]
            if (list(value.shape) != meta['shape'] or str(value.dtype) != meta['dtype'] or
                    array_sha(value) != meta['sha256'] or not np.isfinite(value).all()):
                raise ValueError('Invalid cached train features')
        self.events.append({'kind': 'train_features', 'key': key})
        return arrays

    def fit(self, namespace, original_fit, X, y, prompts, cfg=None):
        identity = {'namespace': namespace, 'features': array_sha(X), 'y': array_sha(y),
                    'prompts': digest_json([str(v) for v in prompts]), 'cfg': _jsonable(cfg)}
        key = digest_json(identity)
        if self.write:
            result = original_fit(X, y, prompts, cfg)
            arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
            metadata = {k: _jsonable(v) for k, v in result.items() if k not in arrays}
            filename = 'gaussian-' + key + '.npz'
            np.savez_compressed(self.root / filename, **arrays)
            record = {'file': filename, 'sha256': file_sha(self.root / filename), 'identity': identity,
                      'metadata': metadata, 'array_names': sorted(arrays)}
            self.manifest['fits'][key] = record
            self.flush()
            loaded = self._load_fit(key)
            if metadata != {k: _jsonable(v) for k, v in loaded.items() if k not in arrays}:
                raise ValueError('Fitted metadata differs after serialization')
            if not all(np.array_equal(result[k], loaded[k]) for k in arrays):
                raise ValueError('Fitted tensors differ after serialization')
            self.manifest['parity']['fit-' + key] = {'passed': True, 'method': 'exact-parameter-round-trip'}
            self.flush()
            return result
        if key not in self.manifest['fits']:
            raise ValueError('Fitted Gaussian cache does not match inputs: ' + namespace)
        self.events.append({'kind': 'fitted_gaussian', 'namespace': namespace, 'key': key})
        return self._load_fit(key)

    def _load_fit(self, key):
        record = self.manifest['fits'][key]
        path = self.root / record['file']
        if file_sha(path) != record['sha256']: raise ValueError('Changed fitted Gaussian artifact')
        with np.load(path, allow_pickle=False) as pack:
            result = {name: pack[name] for name in pack.files}
        if sorted(result) != record['array_names'] or not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('Invalid fitted Gaussian arrays')
        return {**result, **record['metadata']}


def get_store():
    global _STORE
    if _STORE is None:
        import pandas as pd
        contract = json.loads(Path('/kaggle/working/cache-contract.json').read_text())
        paths = sorted(Path('/kaggle/input').rglob('train-cache-manifest.json'))
        if len(paths) != 1: raise ValueError('Attach exactly one completed train feature cache')
        locations = [Path('/kaggle/input/competitions/name-that-llm/train.csv'),
                     Path('/kaggle/input/name-that-llm/train.csv')]
        path = next(p for p in locations if p.is_file())
        frame = pd.read_csv(path); frame.text = frame.text.fillna('').astype(str)
        _STORE = TrainCache(paths[0].parent, contract, frame)
    return _STORE


def main_recipe(name, bs):
    return {'kind': 'nishant-multi-layer-mean', 'name': name, 'batch_size': bs}


def frozen_recipe(key, parts, decoder, bs):
    return {'kind': 'frozen-pools', 'name': key, 'parts': [list(v) for v in parts],
            'decoder': bool(decoder), 'batch_size': bs, 'max_len': 512}


def dhruv_recipe(name, spec):
    return {'kind': 'dhruv-multi-layer-mean', 'name': name, 'spec': spec, 'max_len': 384}


def install_main(namespace):
    store = get_store()
    original_features = namespace['lora_features']
    train_texts = namespace['train'].text.astype(str).tolist()
    def features(ckpt, base, texts, bs):
        name = Path(ckpt).name
        if name == 'lora8b_v1' and len(texts) >= len(train_texts) and list(texts[-len(train_texts):]) == train_texts:
            cached = store.load_features('main-' + name, main_recipe(name, bs), checkpoint_identity(base, ckpt))['Z']
            test_count = len(texts) - len(train_texts)
            if test_count <= 0: raise ValueError('Expected test features followed by official train')
            test, heads = original_features(ckpt, base, texts[:test_count], bs)
            return np.concatenate([test, cached], axis=0), heads
        return original_features(ckpt, base, texts, bs)
    namespace['lora_features'] = features
    original_fit = namespace['fit']
    namespace['fit'] = lambda X, y, prompts, cfg=None: store.fit('main', original_fit, X, y, prompts, cfg)


def install_frozen(namespace):
    store = get_store()
    def extract(key, parts, loader):
        if namespace['SMOKE']: raise ValueError('Synthetic features cannot use a production cache')
        model, tok, decoder, bs = loader()
        identity = checkpoint_identity(tok.name_or_path)
        train = store.load_features('frozen-' + key, frozen_recipe(key, parts, decoder, bs), identity)
        # The source frozen extractor can switch permanently to BF16 after a
        # nonfinite FP16 training batch. Preserve that state for test inference.
        dtype_name = store.manifest['features']['frozen-' + key]['runtime'].get('dtype_after_train')
        if dtype_name:
            dtype = getattr(namespace['torch'], dtype_name)
            if next(model.parameters()).dtype != dtype: model.to(dtype=dtype)
        test = namespace['pooled_states'](model, tok, namespace['TE'], parts, decoder, bs)
        del model, tok
        gc.collect()
        namespace['torch'].cuda.empty_cache()
        return {key: (train[key], test[key]) for key in train}
    namespace['extract_parts'] = extract
    original_fit = namespace['fit']
    namespace['fit'] = lambda X, y, prompts, cfg=None: store.fit('frozen', original_fit, X, y, prompts, cfg)


def dhruv_train_features(name, spec, base, adapter):
    return get_store().load_features('dhruv-' + name, dhruv_recipe(name, spec), checkpoint_identity(base, adapter))['Z']


def strict_pretrained_loading():
    """Reject absent pretrained weights while permitting deliberately omitted tail layers."""
    from transformers import AutoModel
    original = AutoModel.from_pretrained
    def checked(*args, **kwargs):
        requested = kwargs.get('output_loading_info', False)
        kwargs['output_loading_info'] = True
        model, info = original(*args, **kwargs)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise RuntimeError('Refusing incomplete pretrained weights')
        return (model, info) if requested else model
    AutoModel.from_pretrained = staticmethod(checked)
