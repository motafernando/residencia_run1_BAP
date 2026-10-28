"""Durable, auditable JSONL journal for interrupted inference."""
import json
import os
from pathlib import Path


def validate_resume(saved_config, config, state, code_hashes, input_hashes, image_id):
    if saved_config != config or state.get('code_hashes') != code_hashes or state.get('input_hashes') != input_hashes:
        raise ValueError('Resume rejected: configuration, code or input changed')
    if state.get('environment', {}).get('image_id') != image_id:
        raise ValueError('Resume rejected: Docker image changed')
    if state.get('status') == 'inference_finished_needs_manual_review':
        raise ValueError('Resume rejected: inference already finished')


def append_row(path, row):
    with Path(path).open('ab') as handle:
        handle.write((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))
        handle.flush()
        os.fsync(handle.fileno())


def recover_journal(path, expected):
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    last_newline = data.rfind(b'\n') + 1
    if last_newline < len(data):
        fragment = data[last_newline:]
        count = 1
        while (path.parent / f'responses-incomplete-{count}.bin').exists():
            count += 1
        archive = path.parent / f'responses-incomplete-{count}.bin'
        with archive.open('xb') as saved:
            saved.write(fragment)
            saved.flush()
            os.fsync(saved.fileno())
        with path.open('r+b') as handle:
            handle.truncate(last_newline)
            handle.flush()
            os.fsync(handle.fileno())
        data = data[:last_newline]
    result = {}
    for number, raw in enumerate(data.splitlines(), 1):
        try:
            row = json.loads(raw)
            key = (row['id'], row['condition'])
            if key not in expected or row['status'] not in ('ok', 'error'):
                raise ValueError('unknown key or status')
            if key in result and result[key]['status'] == 'ok':
                raise ValueError('duplicate successful result')
            result[key] = row
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'Invalid journal row {number}: {exc}') from exc
    return result
