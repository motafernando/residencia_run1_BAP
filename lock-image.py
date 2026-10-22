"""Resolve the official Docker base tag to an immutable digest without pulling layers."""
import json
import urllib.request
from prepare import ROOT, write_json

repository = 'pytorch/pytorch'
tag = '2.7.1-cuda12.8-cudnn9-runtime'
token_url = ('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'
             + repository + ':pull')
with urllib.request.urlopen(token_url, timeout=30) as response:
    token = json.load(response)['token']
request = urllib.request.Request(f'https://registry-1.docker.io/v2/{repository}/manifests/{tag}',
    headers={'Authorization': 'Bearer ' + token,
             'Accept': 'application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json'})
with urllib.request.urlopen(request, timeout=30) as response:
    digest = response.headers['Docker-Content-Digest']
    manifest = json.load(response)
lock = {'repository': repository, 'tag': tag, 'digest': digest,
        'pinned_reference': repository + '@' + digest,
        'registry': 'https://registry-1.docker.io', 'manifest': manifest}
target = ROOT / 'image-lock.json'
if target.exists():
    if json.loads(target.read_text(encoding='utf-8'))['digest'] != digest:
        raise RuntimeError('Tag changed since the previous lock; review before replacing image-lock.json')
else:
    write_json(target, lock)
print(lock['pinned_reference'])
