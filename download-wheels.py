"""Fetch pinned Linux Python 3.11 wheels and verify their official PyPI hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
from prepare import ROOT, write_json

folder = ROOT / 'wheelhouse'
folder.mkdir(exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'download', '--dest', str(folder),
                '--only-binary=:all:', '--no-deps', '--platform', 'manylinux2014_x86_64',
                '--python-version', '311', '--implementation', 'cp', '--abi', 'cp311',
                '-r', str(ROOT / 'requirements.txt')], check=True)
verified = []
for wheel in sorted(folder.glob('*.whl')):
    name, version = wheel.name.split('-')[:2]
    url = f'https://pypi.org/pypi/{name}/{version}/json'
    metadata = json.load(urllib.request.urlopen(url, timeout=30))
    package = next(item for item in metadata['urls'] if item['filename'] == wheel.name)
    actual = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if actual != package['digests']['sha256']:
        raise ValueError('PyPI hash mismatch: ' + wheel.name)
    verified.append({'file': wheel.name, 'sha256': actual, 'source': package['url'],
                     'bytes': wheel.stat().st_size})
write_json(ROOT / 'wheelhouse-manifest.json', {'python': '3.11', 'platform': 'manylinux2014_x86_64',
                                              'files': verified, 'all_pypi_hashes_verified': True})
print('Verified PyPI wheels:', len(verified))
