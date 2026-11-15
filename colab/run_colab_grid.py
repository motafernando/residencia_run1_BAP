"""Resume-aware Colab grid runner for post-Gate 9 manifests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_ROOT = ROOT / 'configs' / 'colab'
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


def collect_manifests(block, name_filter=None):
    blocks = ['smoke', 'bloco_a', 'bloco_b', 'bloco_c'] if block == 'all' else [block]
    files = []
    for b in blocks:
        b_dir = MANIFEST_ROOT / b
        if not b_dir.exists():
            continue
        block_files = list(b_dir.glob('*.json'))
        if b == 'smoke':
            block_files.sort(key=lambda p: SMOKE_ORDER.get(p.name, len(SMOKE_ORDER)))
        else:
            block_files.sort()
        for p in block_files:
            if name_filter and name_filter.lower() not in p.name.lower():
                continue
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=['smoke', 'bloco_a', 'bloco_b', 'bloco_c', 'all'], required=True)
    parser.add_argument('--output-root', default=str(ROOT / 'results_colab'))
    parser.add_argument('--filter', help='Optional substring filter on manifest filename')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = collect_manifests(args.block, args.filter)
    if not manifests:
        print('No manifests matched.')
        return

    existing = find_existing_runs(output_root)
    for idx, manifest_path in enumerate(manifests, 1):
        cfg = load_json(manifest_path)
        ident = run_identity(cfg)
        match = existing.get(ident)
        if match and match[1] == 'inference_finished_needs_manual_review':
            print(f'[{idx}/{len(manifests)}] SKIP (done): {manifest_path.name} -> {match[0].name}')
            continue

        if match and match[1] == 'failed':
            match = None

        runner = ROOT / ('quantization/run.py' if cfg['model'].get('quantization') else 'run.py')
        cmd = [sys.executable, str(runner), '--manifest', str(manifest_path), '--output-root', str(output_root)]
        if match:
            print(f'[{idx}/{len(manifests)}] RESUME ({match[1]}): {manifest_path.name} -> {match[0].name}')
            cmd.extend(['--resume-from', str(match[0])])
        else:
            print(f'[{idx}/{len(manifests)}] START: {manifest_path.name}')

        if args.dry_run:
            print('  DRY-RUN:', ' '.join(cmd))
            continue

        subprocess.run(cmd, cwd=str(ROOT), check=True)
        existing = find_existing_runs(output_root)


if __name__ == '__main__':
    main()
