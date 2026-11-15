"""Small runtime helpers shared by quantized and unquantized runners."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
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


def load_runtime_overrides(path=None):
    """Load optional out-of-band runtime overrides (e.g., Colab control JSON)."""
    target = Path(path) if path is not None else Path(
        os.environ.get('BAP_RUNTIME_CONFIG', '/tmp/bap_colab_runtime.json')
    )
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def model_prompt_signature(cfg):
    """Identify deterministic model + prompt configuration independent of seed."""
    model = cfg['model']
    return (
        model['repo_id'],
        model.get('revision'),
        model.get('tokenizer_revision'),
        model.get('dtype'),
        json.dumps(model.get('quantization'), sort_keys=True),
        json.dumps(cfg.get('format'), sort_keys=True),
        cfg.get('prompt_variant', 'original'),
        cfg.get('sample'),
        cfg.get('inventory'),
    )


def model_bundle_key(cfg):
    """Identify when two manifests can safely reuse the same loaded model/tokenizer."""
    model = cfg['model']
    return (
        model['repo_id'],
        model.get('revision'),
        model.get('tokenizer_revision'),
        model.get('dtype'),
        json.dumps(model.get('device_map'), sort_keys=True),
        json.dumps(model.get('max_memory'), sort_keys=True),
        json.dumps(model.get('quantization'), sort_keys=True),
        model.get('attn_implementation', 'eager'),
        bool(model.get('local_files_only', False)),
    )


def manifest_run_identity(cfg):
    """Match run identity used by grid orchestrator to detect completed runs."""
    return (
        cfg['model']['repo_id'],
        cfg.get('prompt_variant', 'original'),
        cfg['seed'],
        json.dumps(cfg['model'].get('quantization'), sort_keys=True),
        cfg['sample'],
    )


def find_completed_manifest_run(output_root, config):
    """Return existing completed *-inference directory matching config identity, else None."""
    root = Path(output_root)
    if not root.exists():
        return None
    target = manifest_run_identity(config)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.endswith('-inference'):
            continue
        cfg_path = folder / 'config.json'
        run_path = folder / 'run.json'
        if not cfg_path.exists() or not run_path.exists():
            continue
        try:
            sib_cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            sib_state = json.loads(run_path.read_text(encoding='utf-8'))
            if (
                sib_state.get('status') == 'inference_finished_needs_manual_review'
                and manifest_run_identity(sib_cfg) == target
            ):
                return folder
        except Exception:
            continue
    return None


def cache_seed_key(condition, record_id, option_map=None):
    """Build canonical lookup key for deterministic cross-seed reuse."""
    if condition == 'intrinsic':
        return ('intrinsic', record_id)
    if option_map is None:
        raise ValueError('option_map required for generation cache key')
    return (condition, record_id, tuple(sorted(option_map.items())))


def extract_cache_entry(row):
    """Extract cacheable inference fields from an ok response row."""
    if row.get('status') != 'ok':
        return None
    condition = row.get('condition')
    if condition == 'intrinsic' and 'scores' in row:
        return {
            'scores': row['scores'],
            'prediction': row.get('prediction'),
            'selection_prediction': row.get('selection_prediction'),
            'rpa_prediction': row.get('rpa_prediction'),
            'tie': row.get('tie'),
        }
    if condition in ('zero_shot', 'three_shot') and 'generated_token_ids' in row and 'generated_text' in row:
        return {
            'rendered_prompt': row.get('rendered_prompt'),
            'effective_generation_config': row.get('effective_generation_config'),
            'input_token_ids': row.get('input_token_ids'),
            'generated_token_ids': row['generated_token_ids'],
            'generated_text': row['generated_text'],
        }
    return None


def apply_cached_row(cached, row, condition, prompt_text, candidates, option_map, parse_fn):
    """Populate row from cached deterministic inference output and update parsed fields."""
    row.update(copy.deepcopy(cached))
    if condition == 'intrinsic':
        row['prompt'] = prompt_text
    else:
        row['parsing'] = parse_fn(row['generated_text'], candidates, option_map)
        row['prediction'] = row['parsing']['selection_prediction']
        row['selection_prediction'] = row['parsing']['selection_prediction']
        row['rpa_prediction'] = row['parsing']['rpa_prediction']
        row['rpa_choice_prediction'] = row['parsing']['rpa_choice_prediction']
    row['status'] = 'ok'


def load_cross_seed_cache(output_root, config, effective_max_new_tokens, exclude_dir=None):
    """Load reusable deterministic outputs from sibling runs with identical prompt/model."""
    gen = config.get('generation') or {}
    if gen.get('do_sample', False) or gen.get('num_beams', 1) != 1:
        return {}
    root = Path(output_root)
    if not root.exists():
        return {}
    target_sig = model_prompt_signature(config)
    excluded = Path(exclude_dir).resolve() if exclude_dir is not None else None
    cache = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.endswith('-inference'):
            continue
        if excluded is not None and folder.resolve() == excluded:
            continue
        cfg_path = folder / 'config.json'
        journal_path = folder / 'responses.jsonl'
        if not cfg_path.exists() or not journal_path.exists():
            continue
        try:
            sib_cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            sib_gen = sib_cfg.get('generation') or {}
            if sib_gen.get('do_sample', False) or sib_gen.get('num_beams', 1) != 1:
                continue
            if model_prompt_signature(sib_cfg) != target_sig:
                continue
        except Exception:
            continue
        with journal_path.open('r', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if row.get('status') != 'ok':
                    continue
                cond = row.get('condition')
                if cond == 'intrinsic' and 'scores' in row:
                    key = cache_seed_key('intrinsic', row['id'])
                    cache[key] = extract_cache_entry(row)
                elif cond in ('zero_shot', 'three_shot') and 'generated_token_ids' in row and 'option_map' in row:
                    gen_ids = row['generated_token_ids']
                    sib_mnt = (row.get('effective_generation_config') or {}).get('max_new_tokens') or sib_gen.get('max_new_tokens', 48)
                    if sib_mnt == effective_max_new_tokens or (
                        len(gen_ids) < sib_mnt and len(gen_ids) <= effective_max_new_tokens
                    ):
                        entry = extract_cache_entry(row)
                        if entry is not None:
                            key = cache_seed_key(cond, row['id'], row['option_map'])
                            cache[key] = entry
    return cache


def resolve_batch_manifests(manifest, manifests_arg, prepare_only, resume_from, overrides):
    """Build manifest queue, expanding sibling manifests from runtime overrides when appropriate."""
    if manifests_arg:
        manifest_list = [str(p) for p in manifests_arg]
    else:
        manifest_list = [str(manifest)]
        siblings = (overrides or {}).get('batch_sibling_manifests') or []
        if not prepare_only and not resume_from and siblings:
            cur_name = Path(manifest).name
            sibling_names = [Path(p).name for p in siblings]
            if cur_name in sibling_names:
                seen_names = {cur_name}
                for item in siblings:
                    item_name = Path(item).name
                    if item_name not in seen_names:
                        manifest_list.append(str(item))
                        seen_names.add(item_name)
    return manifest_list


def append_grid_log(output_root, message):
    """Append a timestamped line to grid_progress.log inside output_root."""
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    log_path = Path(output_root) / 'grid_progress.log'
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(f'{stamp} {message}\n')
        handle.flush()


def _resolve_target_manifest_paths(names, search_dirs):
    resolved = []
    for item in names:
        candidate = Path(item)
        if candidate.exists():
            resolved.append(candidate)
            continue
        found = None
        for root in search_dirs:
            root_path = Path(root)
            if not root_path.exists():
                continue
            direct = root_path / candidate.name
            if direct.exists():
                found = direct
                break
            matches = list(root_path.rglob(candidate.name))
            if matches:
                found = matches[0]
                break
        if found is None:
            return None
        resolved.append(found)
    return resolved


def maybe_stop_parent_grid(output_root, search_dirs, overrides):
    """Stop parent run_colab_grid.py process cleanly once all target manifests complete."""
    target_names = (overrides or {}).get('stop_after_manifest_names')
    if not target_names:
        return False
    target_paths = _resolve_target_manifest_paths(target_names, search_dirs)
    if not target_paths:
        return False
    for path in target_paths:
        try:
            cfg = json.loads(Path(path).read_text(encoding='utf-8'))
        except Exception:
            return False
        if find_completed_manifest_run(output_root, cfg) is None:
            return False
    append_grid_log(output_root, 'ALL TARGET BLOCO_C MANIFESTS COMPLETED — STOPPING GRID PER RUNTIME CONFIG')
    if overrides.get('stop_parent_grid_on_completion'):
        ppid = os.getppid()
        cmdline_path = Path(f'/proc/{ppid}/cmdline')
        should_signal = True
        if cmdline_path.exists():
            try:
                cmd = cmdline_path.read_bytes().decode('utf-8', errors='replace')
                should_signal = 'run_colab_grid.py' in cmd
            except OSError:
                should_signal = False
        if should_signal:
            sys.stdout.flush()
            sys.stderr.flush()
            os.kill(ppid, signal.SIGTERM)
    return True
