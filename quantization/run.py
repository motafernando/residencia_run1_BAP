"""Manifest-driven WinoBias smoke. --prepare-only performs no model inference."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import time
import traceback

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from prepare import ROOT, digest, write_json
from protocol import make_inputs, metrics
from parser_v4 import parse_response
from resume_journal import append_row, recover_journal, validate_resume
from quantization_config import validate_quantization, build_quantization, summarize_quantized_model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def score_candidate(model, tokenizer, prefix, candidate, torch):
    # Tokenize joint text; fail rather than silently score a boundary mismatch.
    prefix_ids = tokenizer(prefix, add_special_tokens=True)['input_ids']
    all_ids = tokenizer(prefix + ' ' + candidate, add_special_tokens=True)['input_ids']
    if all_ids[:len(prefix_ids)] != prefix_ids or len(all_ids) <= len(prefix_ids):
        raise ValueError('Candidate tokenization changes the prefix boundary; manual review required')
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([all_ids], device=device)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        start = len(prefix_ids)
        # Materialize log probabilities only for candidate positions.
        logp = output.logits[0, start - 1:-1].float().log_softmax(dim=-1)
        targets = ids[0, start:].to(logp.device)
        values = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu().tolist()
    return {'candidate': candidate, 'sum_log_probability': sum(values),
            'token_ids': all_ids[len(prefix_ids):],
            'tokens': tokenizer.convert_ids_to_tokens(all_ids[len(prefix_ids):]),
            'token_log_probabilities': values, 'prefix_token_ids': prefix_ids}


def generate(model, tokenizer, text, config, torch):
    if config['format']['prompting'] == 'tokenizer_chat_template':
        template_kwargs = dict(config['format'].get('chat_template_kwargs') or {})
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                  tokenize=False, add_generation_prompt=True,
                                                  **template_kwargs)
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt')
    elif config['format']['prompting'] == 'raw_completion':
        rendered = text
        inputs = tokenizer(text, add_special_tokens=True, return_tensors='pt')
    else:
        raise ValueError('Unknown prompt format')
    inputs = inputs.to(model.get_input_embeddings().weight.device)
    settings = dict(config['generation'])
    settings['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # Preserve model-specific end-of-turn IDs; override only the smoke settings.
    generation = copy.deepcopy(model.generation_config)
    generation.update(**settings)
    if generation.eos_token_id is None:
        generation.eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=generation)
    continuation = output[0, inputs['input_ids'].shape[-1]:].cpu().tolist()
    return {'rendered_prompt': rendered,
            'effective_generation_config': generation.to_dict(),
            'input_token_ids': inputs['input_ids'][0].cpu().tolist(),
            'generated_token_ids': continuation,
            'generated_text': tokenizer.decode(continuation, skip_special_tokens=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(ROOT / 'configs' / 'smoke.json'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume-from', help='Existing inference folder under this manifest output_root')
    parser.add_argument('--output-root', help='Override manifest output_root (e.g. Google Drive path)')
    args = parser.parse_args()
    config = load(args.manifest)
    source = load(ROOT / config['sample'])
    inventory = load(ROOT / config['inventory'])
    records = source['records'] if isinstance(source, dict) else source
    prompt_variant = config.get('prompt_variant', 'original')
    inputs = [make_inputs(row, config['seed'], prompt_variant) for row in records]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output_root = Path(args.output_root).resolve() if args.output_root else (ROOT / config['output_root']).resolve()
    source_files = ['prepare.py', 'protocol.py', 'parser_v3.py', 'parser_v4.py', 'resume_journal.py', 'run.py', 'quantization_config.py']
    local_dir = Path(__file__).resolve().parent
    resolved_sources = {
        name: (local_dir / name if name in ('run.py', 'quantization_config.py') and (local_dir / name).exists() else ROOT / name)
        for name in source_files
    }
    current_code = {name: file_hash(path) for name, path in resolved_sources.items()}
    current_inputs = {name: file_hash(ROOT / config[name]) for name in ['sample', 'inventory']}
    if args.resume_from:
        if args.prepare_only:
            raise ValueError('Cannot resume a prepare-only run')
        out = Path(args.resume_from).resolve()
        if out.parent != output_root or not out.name.endswith('-inference'):
            raise ValueError('Resume folder is outside this manifest output_root')
        state = load(out / 'run.json')
        validate_resume(load(out / 'config.json'), config, state, current_code, current_inputs,
                        os.environ.get(config['image_id_env']))
        state.setdefault('resume_attempts', []).append({'started_utc': stamp, 'previous_status': state['status']})
    else:
        out = output_root / (stamp + ('-prepared' if args.prepare_only else '-inference'))
        out.mkdir(parents=True, exist_ok=False)
        for name, src_path in resolved_sources.items():
            shutil.copyfile(src_path, out / name.replace('.py', '.used.py'))
        write_json(out / 'config.json', config)
        write_json(out / 'sample.json', source)
        write_json(out / 'prompts.json', inputs)
        state = {'status': 'prepared_no_inference', 'created_utc': stamp,
             'sample_size': len(records), 'conditions': ['intrinsic', 'zero_shot', 'three_shot'],
             'dataset_revision': inventory['revision'], 'input_hashes': {},
             'code_hashes': current_code,
             'parser_version': 'v4_label_boundary', 'prompt_variant': prompt_variant,
             'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                             'image_id': os.environ.get(config['image_id_env'])},
             'model_selection': config['model']['selection_status']}
        state['input_hashes'] = current_inputs
    if args.prepare_only:
        write_json(out / 'run.json', state)
        print(json.dumps({'status': state['status'], 'output': str(out), 'prompts': len(inputs) * 3}))
        return
    lock = None
    try:
        import fcntl
        lock = (out / '.inference.lock').open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        if lock is not None:
            lock.close()
        raise RuntimeError('Another inference process owns this run') from exc
    except (ImportError, OSError):
        if lock is not None:
            lock.close()
            lock = None
    write_json(out / 'run.json', state)
    started = time.perf_counter()
    try:
        expected = {(record['id'], condition) for record in records for condition in state['conditions']}
        if len(expected) != len(records) * len(state['conditions']):
            raise ValueError('Duplicate sample IDs prevent reliable resumption')
        journal = out / 'responses.jsonl'
        completed = recover_journal(journal, expected)
        previous_checkpoint = state.get('checkpoint_files')
        previous_tokenizer = state.get('tokenizer_files')
        previous_device_map = state.get('model_device_map')
        previous_quantization = state.get('quantization')
        m = config['model']
        if m['repo_id'] not in m['allowed_repos']:
            raise ValueError('Choose one of the official study checkpoint candidates before inference')
        if not m.get('allow_unpinned_revision'):
            for field in ['revision', 'tokenizer_revision']:
                if not re.fullmatch(r'[0-9a-f]{40}', m[field] or ''):
                    raise ValueError(f'model.{field} must be a reviewed, complete 40-character commit SHA')
        validate_quantization(m.get('quantization'))
        if m.get('device_map') != {'': 0}:
            raise ValueError('Quantization profile requires the entire model on GPU 0')
        import torch
        import bitsandbytes as bnb
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from huggingface_hub import snapshot_download
        state['environment']['packages'] = {p: importlib.metadata.version(p) for p in
                                             ['torch', 'transformers', 'accelerate', 'huggingface-hub', 'tokenizers', 'safetensors', 'bitsandbytes']}
        freeze = ROOT / 'environment-freeze.txt'
        if freeze.exists():
            state['environment']['container_package_freeze'] = freeze.read_text(encoding='utf-8')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in the runtime; inference not started')
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        random.seed(config['seed'])
        state['environment']['cuda'] = torch.version.cuda
        state['environment']['gpu'] = torch.cuda.get_device_name(0)
        state['environment']['gpu_arch'] = list(torch.cuda.get_device_capability(0))
        state['environment']['torch_arch_list'] = torch.cuda.get_arch_list()
        state['environment']['cuda_free_total_before_load'] = list(torch.cuda.mem_get_info())
        state['status'] = 'loading_model'
        write_json(out / 'run.json', state)
        patterns = ['*.json', 'model*.safetensors', '*.model', '*.jinja', 'merges.txt', 'vocab.*', 'tokenizer*', 'special_tokens_map*']
        snapshot = snapshot_download(repo_id=m['repo_id'], revision=m['revision'],
                                     local_files_only=m['local_files_only'], allow_patterns=patterns)
        token_snapshot = snapshot_download(repo_id=m['repo_id'], revision=m['tokenizer_revision'],
                                           local_files_only=m['local_files_only'], allow_patterns=patterns)
        state['resolved_revisions'] = {'model': Path(snapshot).name, 'tokenizer': Path(token_snapshot).name}
        tokenizer = AutoTokenizer.from_pretrained(token_snapshot, local_files_only=True, trust_remote_code=False)
        if config['format']['prompting'] == 'tokenizer_chat_template' and not tokenizer.chat_template:
            raise ValueError('Selected tokenizer has no chat template; explicitly select raw_completion if appropriate')
        state['checkpoint_files'] = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                                     for p in sorted(Path(snapshot).iterdir()) if p.is_file()
                                     and p.suffix in {'.safetensors', '.json', '.model', '.jinja'}]
        state['tokenizer_files'] = [{'file': p.name, 'sha256': file_hash(p)}
                                    for p in sorted(Path(token_snapshot).iterdir()) if p.is_file()
                                    and ('token' in p.name or p.suffix in {'.model', '.jinja'})]
        if previous_checkpoint and previous_checkpoint != state['checkpoint_files']:
            raise ValueError('Resume rejected: checkpoint files changed')
        if previous_tokenizer and previous_tokenizer != state['tokenizer_files']:
            raise ValueError('Resume rejected: tokenizer files changed')
        memory = {int(k) if k.isdigit() else k: v for k, v in m['max_memory'].items()}
        offload = out / 'offload'
        load_kwargs = dict(
            torch_dtype=getattr(torch, m['dtype']),
            device_map=m['device_map'],
            max_memory=memory,
            offload_folder=str(offload),
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation=m.get('attn_implementation', 'eager'),
            quantization_config=build_quantization(m['quantization'], torch, BitsAndBytesConfig),
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(snapshot, **load_kwargs)
        except ValueError:
            AutoModelForImageTextToText = getattr(transformers, 'AutoModelForImageTextToText', None)
            if AutoModelForImageTextToText is None:
                raise
            model = AutoModelForImageTextToText.from_pretrained(snapshot, **load_kwargs)
        model.eval()
        state['model_device_map'] = {str(k): str(v) for k, v in model.hf_device_map.items()}
        state['quantization'] = summarize_quantized_model(model, bnb, m['quantization'])
        if previous_quantization and previous_quantization != state['quantization']:
            raise ValueError('Resume rejected: effective quantization or module layout changed')
        if previous_device_map and previous_device_map != state['model_device_map']:
            raise ValueError('Resume rejected: model device map changed')
        state['load_seconds'] = time.perf_counter() - started
        state['status'] = 'running'
        write_json(out / 'run.json', state)
        for record, prompt in zip(records, inputs):
                for condition in state['conditions']:
                    key = (record['id'], condition)
                    if key in completed and completed[key]['status'] == 'ok':
                        continue
                    row = {**record, 'condition': condition, 'option_map': prompt['option_map']}
                    tick = time.perf_counter()
                    try:
                        if condition == 'intrinsic':
                            scores = [{**score_candidate(model, tokenizer, prompt[condition], record['candidate_text'][c], torch),
                                       'candidate_id': c} for c in record['candidates']]
                            tied = scores[0]['sum_log_probability'] == scores[1]['sum_log_probability']
                            intrinsic_prediction = None if tied else max(
                                scores, key=lambda s: s['sum_log_probability'])['candidate_id']
                            row.update(scores=scores, prompt=prompt[condition],
                                       prediction=intrinsic_prediction,
                                       selection_prediction=intrinsic_prediction,
                                       rpa_prediction=intrinsic_prediction, tie=tied)
                        else:
                            row.update(generate(model, tokenizer, prompt[condition], config, torch))
                            row['parsing'] = parse_response(row['generated_text'], record['candidates'], prompt['option_map'])
                            row['prediction'] = row['parsing']['selection_prediction']
                            row['selection_prediction'] = row['parsing']['selection_prediction']
                            row['rpa_prediction'] = row['parsing']['rpa_prediction']
                            row['rpa_choice_prediction'] = row['parsing']['rpa_choice_prediction']
                        row['status'] = 'ok'
                    except Exception as error:
                        row.update(status='error', prediction=None, error_type=type(error).__name__, error=str(error))
                    row['seconds'] = time.perf_counter() - tick
                    append_row(journal, row)
                    completed[key] = row
                    print(f"{len(completed)}/{len(records)*3} {record['id']} {condition} {row['status']}", flush=True)
        rows = [completed[(record['id'], condition)] for record in records for condition in state['conditions']]
        write_json(out / 'metrics.json', metrics(rows, inventory['occupations']))
        errors = sum(r['status'] != 'ok' for r in rows)
        import psutil
        state.update(status='inference_finished_needs_manual_review' if not errors else 'inference_finished_with_errors',
                     records_written=len(rows), execution_errors=errors,
                     elapsed_seconds=time.perf_counter() - started,
                     gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        state['cpu_rss_at_end_bytes'] = psutil.Process().memory_info().rss
        write_json(out / 'run.json', state)
        if errors:
            raise RuntimeError(f'{errors} item/condition executions failed; see responses.jsonl')
    except Exception as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error),
                     elapsed_seconds=time.perf_counter() - started)
        write_json(out / 'run.json', state)
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        if lock is not None:
            lock.close()
    print(json.dumps({'status': state['status'], 'output': str(out)}))


if __name__ == '__main__':
    main()
