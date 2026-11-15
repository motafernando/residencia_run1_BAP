"""Small runtime helpers shared by quantized and unquantized runners."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def describe_model_device_map(model):
    """Return an auditable device map across Transformers model classes."""
    accelerate_map = getattr(model, 'hf_device_map', None)
    if accelerate_map:
        return {str(key): str(value) for key, value in accelerate_map.items()}

    device = getattr(model, 'device', None)
    if device is None:
        embeddings = model.get_input_embeddings()
        device = getattr(getattr(embeddings, 'weight', None), 'device', None)
    if device is None:
        raise RuntimeError('Loaded model exposes neither hf_device_map nor a parameter device')
    return {'': str(device)}


def ensure_chat_template(tokenizer, token_snapshot):
    """Load standalone chat_template.jinja when tokenizer_config omits it."""
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.chat_template
    template_path = Path(token_snapshot) / 'chat_template.jinja'
    if template_path.exists():
        tokenizer.chat_template = template_path.read_text(encoding='utf-8')
    return getattr(tokenizer, 'chat_template', None)


def prepare_generation_config(base_generation_config, settings, tokenizer):
    """Build deterministic generation config without leftover sampling flags."""
    generation = copy.deepcopy(base_generation_config)
    updated = dict(settings)
    updated['pad_token_id'] = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    generation.update(**updated)
    if generation.eos_token_id is None:
        generation.eos_token_id = tokenizer.eos_token_id
    if not getattr(generation, 'do_sample', False):
        for flag in ('temperature', 'top_p', 'top_k'):
            if hasattr(generation, flag):
                setattr(generation, flag, None)
    return generation


def _sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def cached_file_hash(path, raw_hash_fn=None, cache_path=None):
    """Return SHA-256 hex digest using HF blob name or a stat-keyed cache."""
    file_path = Path(path)
    resolved = file_path.resolve()
    if resolved.parent.name == 'blobs' and re.fullmatch(r'[0-9a-f]{64}', resolved.name):
        return resolved.name

    cache_file = (
        Path(cache_path)
        if cache_path is not None
        else Path(tempfile.gettempdir()) / 'bap_sha256_cache.json'
    )
    stat = resolved.stat()
    cache_key = f'{resolved}:{stat.st_size}:{stat.st_mtime_ns}'
    cache = {}
    if cache_file.exists():
        try:
            loaded = json.loads(cache_file.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                cache = loaded
        except Exception:
            cache = {}

    if cache_key in cache:
        return cache[cache_key]

    digest = (raw_hash_fn or _sha256_file)(file_path)
    cache[cache_key] = digest
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(f'{cache_file.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, cache_file)
    except OSError:
        pass
    return digest
