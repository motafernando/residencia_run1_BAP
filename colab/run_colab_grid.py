"""Resume-aware Colab grid runner for post-Gate 9 manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_ROOT = ROOT / 'configs' / 'colab'
PRIORITY1_BLOCO_B_PREFIXES = ('ministral3_8b-', 'qwen3_8b-')
PRIORITY2_BLOCO_B_PREFIXES = ('gemma4_e4b-', 'llama31_8b-')
SMOKE_ORDER = {
    name: index for index, name in enumerate([
        'qwen3_8b-refined-bf16_48-seed2026.json',
        'llama31_8b-refined-bf16_48-seed2026.json',
        'ministral3_8b-refined-bf16_48-seed2026.json',
        'gemma4_e4b-refined-bf16_48-seed2026.json',
        'llama31_8b-refined-int8_48-seed2026.json',
        'llama33_70b-refined-int8_48-seed2026.json',
    ])
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_identity(cfg):
    return (
        cfg['model']['repo_id'],
        cfg.get('prompt_variant', 'original'),
        cfg['seed'],
        json.dumps(cfg['model'].get('quantization'), sort_keys=True),
        cfg['sample'],
    )


def find_existing_runs(output_root):
    existing = {}
    if not output_root.exists():
        return existing
    for folder in sorted(output_root.iterdir()):
        if not folder.is_dir() or not folder.name.endswith('-inference'):
            continue
        cfg_path = folder / 'config.json'
        run_path = folder / 'run.json'
        if not cfg_path.exists() or not run_path.exists():
            continue
        try:
            cfg = load_json(cfg_path)
            state = load_json(run_path)
            existing[run_identity(cfg)] = (folder, state.get('status'))
        except Exception:
            continue
    return existing


BLOCK_CHOICES = ['smoke', 'bloco_a', 'bloco_b', 'bloco_c', 'priority1', 'priority2', 'all']


def _iter_block_files(block_name):
    b_dir = MANIFEST_ROOT / block_name
    if not b_dir.exists():
        return []
    files = list(b_dir.glob('*.json'))
    if block_name == 'smoke':
        files.sort(key=lambda p: SMOKE_ORDER.get(p.name, len(SMOKE_ORDER)))
    else:
        files.sort()
    return files


def collect_manifests(block, name_filter=None):
    if block == 'priority1':
        raw_files = _iter_block_files('bloco_c') + [
            p for p in _iter_block_files('bloco_b')
            if p.name.startswith(PRIORITY1_BLOCO_B_PREFIXES)
        ]
    elif block == 'priority2':
        raw_files = [
            p for p in _iter_block_files('bloco_b')
            if p.name.startswith(PRIORITY2_BLOCO_B_PREFIXES) and '-int8_48-' not in p.name
        ]
    else:
        blocks = ['smoke', 'bloco_a', 'bloco_b', 'bloco_c'] if block == 'all' else [block]
        raw_files = [p for b in blocks for p in _iter_block_files(b)]

    if not name_filter:
        return raw_files
    needle = name_filter.lower()
    return [p for p in raw_files if needle in p.name.lower()]


def is_heavy_70b(cfg):
    return '70b' in cfg['model']['repo_id'].lower()


def resolve_workers(block, requested):
    if requested is not None:
        return max(1, int(requested))
    return 3 if block in ('priority1', 'priority2') else 1


def log_progress(output_root, message, *, dry_run=False, lock=None):
    def _emit():
        print(message, flush=True)
        if dry_run:
            return
        stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        log_path = Path(output_root) / 'grid_progress.log'
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(f'{stamp} {message}\n')
            handle.flush()

    if lock is None:
        _emit()
    else:
        with lock:
            _emit()


def _build_job_command(manifest_path, cfg, match, output_root, *, max_new_tokens_cap=None):
    runner = ROOT / ('quantization/run.py' if cfg['model'].get('quantization') else 'run.py')
    cmd = [sys.executable, str(runner), '--manifest', str(manifest_path), '--output-root', str(output_root)]
    if match:
        cmd.extend(['--resume-from', str(match[0])])
    if max_new_tokens_cap is not None:
        cmd.extend(['--max-new-tokens-cap', str(int(max_new_tokens_cap))])
    return cmd


def _tail_text(path, lines=30):
    try:
        content = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
        return '\n'.join(content[-lines:])
    except OSError:
        return ''


def _execute_job(job, *, total, output_root, log_dir, stagger_s, stagger_state, stagger_lock, log_lock, stream_output, max_new_tokens_cap=None):
    idx, manifest_path, cfg, match = job
    cmd = _build_job_command(manifest_path, cfg, match, output_root, max_new_tokens_cap=max_new_tokens_cap)
    with stagger_lock:
        if stagger_state[0] and stagger_s > 0:
            time.sleep(stagger_s)
        stagger_state[0] = True

    label = f'RESUME ({match[1]}): {manifest_path.name} -> {match[0].name}' if match else f'START: {manifest_path.name}'
    log_progress(output_root, f'[{idx}/{total}] {label}', lock=log_lock)

    tick = time.perf_counter()
    if stream_output:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        worker_log = log_dir / f'{manifest_path.stem}.log'
        with worker_log.open('w', encoding='utf-8') as handle:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = _tail_text(worker_log)
            log_progress(
                output_root,
                f'[{idx}/{total}] FAILED ({manifest_path.name}, rc={proc.returncode}):\n{tail}',
                lock=log_lock,
            )
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    elapsed_s = time.perf_counter() - tick
    log_progress(output_root, f'[{idx}/{total}] DONE ({elapsed_s:.1f}s): {manifest_path.name}', lock=log_lock)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=BLOCK_CHOICES, required=True)
    parser.add_argument('--output-root', default=str(ROOT / 'results_colab'))
    parser.add_argument('--filter', help='Optional substring filter on manifest filename')
    parser.add_argument('--workers', type=int, help='Parallel workers for 8B runs (default: 3 for priority1/2, else 1; 70B always runs solo)')
    parser.add_argument('--stagger-seconds', type=float, default=2.5)
    parser.add_argument('--max-new-tokens-cap', type=int, default=None, help='Optional cap on generation.max_new_tokens')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = collect_manifests(args.block, args.filter)
    if not manifests:
        print('No manifests matched.')
        return

    workers = resolve_workers(args.block, args.workers)
    existing = find_existing_runs(output_root)
    pending = []
    for idx, manifest_path in enumerate(manifests, 1):
        cfg = load_json(manifest_path)
        ident = run_identity(cfg)
        match = existing.get(ident)
        if match and match[1] == 'inference_finished_needs_manual_review':
            log_progress(
                output_root,
                f'[{idx}/{len(manifests)}] SKIP (done): {manifest_path.name} -> {match[0].name}',
                dry_run=args.dry_run,
            )
            continue

        if match and match[1] == 'failed':
            match = None
        pending.append((idx, manifest_path, cfg, match))

    if args.dry_run:
        for idx, manifest_path, cfg, match in pending:
            label = f'RESUME ({match[1]}): {manifest_path.name} -> {match[0].name}' if match else f'START: {manifest_path.name}'
            log_progress(output_root, f'[{idx}/{len(manifests)}] {label}', dry_run=True)
            print('  DRY-RUN:', ' '.join(_build_job_command(manifest_path, cfg, match, output_root, max_new_tokens_cap=args.max_new_tokens_cap)), flush=True)
        return

    log_lock = threading.Lock()
    stagger_lock = threading.Lock()
    stagger_state = [False]
    log_dir = Path(tempfile.gettempdir()) / 'bap_worker_logs'

    heavy_jobs = [job for job in pending if is_heavy_70b(job[2])]
    light_jobs = [job for job in pending if not is_heavy_70b(job[2])]

    for job in heavy_jobs:
        _execute_job(
            job, total=len(manifests), output_root=output_root, log_dir=log_dir,
            stagger_s=args.stagger_seconds, stagger_state=stagger_state,
            stagger_lock=stagger_lock, log_lock=log_lock, stream_output=True,
            max_new_tokens_cap=args.max_new_tokens_cap,
        )

    if workers <= 1 or len(light_jobs) <= 1:
        for job in light_jobs:
            _execute_job(
                job, total=len(manifests), output_root=output_root, log_dir=log_dir,
                stagger_s=args.stagger_seconds, stagger_state=stagger_state,
                stagger_lock=stagger_lock, log_lock=log_lock, stream_output=True,
                max_new_tokens_cap=args.max_new_tokens_cap,
            )
        return

    log_progress(output_root, f'Launching {len(light_jobs)} 8B runs across {workers} parallel workers (logs: {log_dir})', lock=log_lock)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _execute_job, job, total=len(manifests), output_root=output_root, log_dir=log_dir,
                stagger_s=args.stagger_seconds, stagger_state=stagger_state,
                stagger_lock=stagger_lock, log_lock=log_lock, stream_output=False,
                max_new_tokens_cap=args.max_new_tokens_cap,
            )
            for job in light_jobs
        ]
        try:
            for future in as_completed(futures):
                future.result()
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise


if __name__ == '__main__':
    main()
