"""Fail-fast checks for the Colab experiment runtime."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_colab_grid import collect_manifests, load_json


def load_exact_requirements(path):
    requirements = {}
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '==' not in line:
            raise ValueError(f'Colab dependency must be exactly pinned: {line}')
        name, version = line.split('==', 1)
        requirements[name] = version
    return requirements


def validate_installed_versions(required, installed):
    mismatches = {
        name: {'required': version, 'installed': installed.get(name)}
        for name, version in required.items()
        if installed.get(name) != version
    }
    if mismatches:
        details = ', '.join(
            f"{name}={values['installed']!r} (required {values['required']})"
            for name, values in mismatches.items()
        )
        raise RuntimeError(f'Colab package versions do not match the lock: {details}')


def installed_versions(required):
    versions = {}
    for name in required:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def unique_models(block):
    models = {}
    for manifest in collect_manifests(block):
        cfg = load_json(manifest)
        model = cfg['model']
        models[(model['repo_id'], model['revision'])] = model
    return list(models.values())


def check_model_access(models, token):
    from huggingface_hub import hf_hub_download

    failures = []
    for model in models:
        repo_id = model['repo_id']
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename='config.json',
                revision=model['revision'],
                token=token,
            )
            print(f'ACCESS OK: {repo_id}')
        except Exception as error:
            failures.append({'repo_id': repo_id, 'error_type': type(error).__name__, 'error': str(error)})
            print(f'ACCESS FAILED: {repo_id} ({type(error).__name__})')
    if failures:
        raise RuntimeError('Model access preflight failed:\n' + json.dumps(failures, indent=2))


def check_gpu(block):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / (1024 ** 3)
    if block in ('smoke', 'bloco_c', 'all') and total_gib < 76:
        raise RuntimeError(f'{block} includes Llama 70B INT8 and requires at least 76 GiB VRAM; found {total_gib:.1f}')
    return {'name': properties.name, 'total_vram_gib': total_gib}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=['smoke', 'bloco_a', 'bloco_b', 'bloco_c', 'all'], default='smoke')
    parser.add_argument('--requirements', default=str(ROOT / 'colab' / 'requirements.txt'))
    parser.add_argument('--report')
    args = parser.parse_args()

    required = load_exact_requirements(args.requirements)
    installed = installed_versions(required)
    validate_installed_versions(required, installed)

    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN is missing from the Colab environment')

    report = {
        'status': 'preflight_ok',
        'block': args.block,
        'packages': installed,
        'gpu': check_gpu(args.block),
        'models': [model['repo_id'] for model in unique_models(args.block)],
    }
    check_model_access(unique_models(args.block), token)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
