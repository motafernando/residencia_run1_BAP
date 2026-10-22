"""Fetch only the reviewed Transformers snapshot and verify upstream file hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

from prepare import ROOT, write_json

# Use resumable HTTP transfers; no second chunk cache is needed for this smoke.
os.environ['HF_HUB_DISABLE_XET'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='configs/mistral-smoke.json')
    args = parser.parse_args()
    config = json.loads((ROOT / args.manifest).read_text(encoding='utf-8'))
    model = config['model']
    if model['repo_id'] != 'mistralai/Mistral-7B-Instruct-v0.3':
        raise ValueError('This downloader is restricted to the approved Mistral smoke')
    revision = model['revision']
    if not re.fullmatch('[0-9a-f]{40}', revision or '') or model['tokenizer_revision'] != revision:
        raise ValueError('Review full model/tokenizer revisions before download')
    history = json.loads((ROOT / 'checkpoint-history.json').read_text(encoding='utf-8'))
    if not any(row['resolved_sha'] == revision for row in history):
        raise ValueError('Revision is absent from the reviewed history')
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(model['repo_id'], revision=revision, files_metadata=True, timeout=30)
    if info.sha != revision:
        raise ValueError('Repository resolved to a different revision')
    files = [item for item in info.siblings if item.rfilename.endswith('.json')
             or item.rfilename.startswith('tokenizer') or item.rfilename.startswith('model-')
             and item.rfilename.endswith('.safetensors')]
    # We intentionally exclude consolidated.safetensors, an alternate copy of the weights.
    cache = ROOT / 'cache' / 'huggingface' / 'hub'
    state = {'status': 'downloading', 'repo_id': model['repo_id'], 'revision': revision,
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'total_bytes': sum(item.size or 0 for item in files), 'files': []}
    status_path = ROOT / 'download-status.json'
    write_json(status_path, state)
    started = time.perf_counter()
    try:
        for item in files:
            name = item.rfilename
            state['current_file'] = name
            write_json(status_path, state)
            print('Downloading', name, item.size, 'bytes', flush=True)
            path = Path(hf_hub_download(model['repo_id'], filename=name, revision=revision,
                                       cache_dir=str(cache)))
            sha256 = hashlib.sha256()
            git_sha1 = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    sha256.update(block)
                    git_sha1.update(block)
            expected = item.lfs.sha256 if item.lfs else item.blob_id
            actual = sha256.hexdigest() if item.lfs else git_sha1.hexdigest()
            if actual != expected:
                raise ValueError('Hash mismatch for ' + name)
            state['files'].append({'file': name, 'bytes': path.stat().st_size,
                                   'sha256': sha256.hexdigest(), 'upstream_hash_verified': True})
            write_json(status_path, state)
            print('Verified', name, flush=True)
        state.update(status='download_complete_verified', elapsed_seconds=time.perf_counter() - started)
        state.pop('current_file', None)
        write_json(status_path, state)
        print('Snapshot complete and verified', revision, flush=True)
    except Exception as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error),
                     elapsed_seconds=time.perf_counter() - started)
        write_json(status_path, state)
        raise


if __name__ == '__main__':
    main()
