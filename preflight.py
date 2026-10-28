"""Check CUDA and the pinned tokenizer without loading model weights."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from prepare import ROOT, write_json
from protocol import make_inputs

config_path = ROOT / 'configs' / 'smoke.json'
config = json.loads(config_path.read_text())
sample = json.loads((ROOT / config['sample']).read_text())
model = config['model']
snapshot = Path('/cache/huggingface/hub') / ('models--' + model['repo_id'].replace('/', '--')) / 'snapshots' / model['tokenizer_revision']
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable inside the container')
tensor = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
kernel_value = float((tensor @ tensor)[0, 0].item())
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
rows = []
records = sample['records'] if isinstance(sample, dict) else sample
for record in records:
    prompts = make_inputs(record, config['seed'], config.get('prompt_variant', 'original'))
    prefix = prompts['intrinsic']
    prefix_ids = tokenizer(prefix, add_special_tokens=True)['input_ids']
    boundaries = []
    for key in record['candidates']:
        text = record['candidate_text'][key]
        joint = tokenizer(prefix + ' ' + text, add_special_tokens=True)['input_ids']
        boundaries.append({'candidate': text, 'prefix_unchanged': joint[:len(prefix_ids)] == prefix_ids,
                           'candidate_token_ids': joint[len(prefix_ids):]})
    lengths = {}
    for condition in ['zero_shot', 'three_shot']:
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompts[condition]}],
                                                  tokenize=False, add_generation_prompt=True)
        lengths[condition] = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
    rows.append({'id': record['id'], 'boundaries': boundaries, 'prompt_token_lengths': lengths})
result = {'status': 'cuda_and_tokenizer_checked_no_model_inference',
          'prompt_variant': config.get('prompt_variant', 'original'), 'seed': config['seed'],
          'manifest_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
          'image_id': os.environ.get('EXPERIMENT_IMAGE_ID'),
          'repo_id': model['repo_id'], 'tokenizer_revision': model['tokenizer_revision'],
          'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(), 'architecture': list(torch.cuda.get_device_capability()),
          'compiled_architectures': torch.cuda.get_arch_list(), 'bf16_matmul_result': kernel_value,
          'cuda_free_total_bytes': list(torch.cuda.mem_get_info()),
          'all_candidate_boundaries_valid': all(b['prefix_unchanged'] and b['candidate_token_ids']
                                                for r in rows for b in r['boundaries']), 'rows': rows}
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
target = ROOT / 'results' / (stamp + '-preflight.json')
write_json(target, result)
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
if not result['all_candidate_boundaries_valid'] or kernel_value != 16.0:
    raise RuntimeError('Preflight requires review before inference')
