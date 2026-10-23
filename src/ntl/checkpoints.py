"""Reject config-only adapter folders when selecting a real base checkpoint."""
import json as _weights_json
import os as _weights_os


def _ntl_has_model_weights(directory):
    for name in ('model.safetensors', 'pytorch_model.bin'):
        if _weights_os.path.isfile(_weights_os.path.join(directory, name)):
            return True
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        path = _weights_os.path.join(directory, name)
        if not _weights_os.path.isfile(path):
            continue
        try:
            with open(path, encoding='utf-8') as stream:
                mapping = _weights_json.load(stream).get('weight_map', {})
            shards = set(mapping.values())
            if shards and all(isinstance(shard, str) and
                              _weights_os.path.isfile(_weights_os.path.join(directory, shard))
                              for shard in shards):
                return True
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return False
