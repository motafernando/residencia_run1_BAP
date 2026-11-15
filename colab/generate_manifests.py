"""Generate Colab manifests for post-Gate 9 multi-seed and new-model runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'configs' / 'colab'

PROMPTS_MAIN = ['original', 'refined', 'evaluation_gender_bias']
SEEDS_5 = [2026, 2027, 2028, 2029, 2030]
SEEDS_4_NEW = [2027, 2028, 2029, 2030]
SEEDS_3 = [2026, 2027, 2028]

INT8_PROFILE = {'method': 'bitsandbytes', 'load_in_8bit': True}

ALLOWED_REPOS = [
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'meta-llama/Llama-3.1-8B-Instruct',
    'mistralai/Ministral-3-8B-Instruct-2512',
    'Qwen/Qwen3-8B',
    'google/gemma-4-E4B-it',
    'meta-llama/Llama-3.3-70B-Instruct',
]

MODELS = {
    'llama3_8b': {
        'repo_id': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'revision': '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2',
        'tokenizer_revision': '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2',
        'allow_unpinned_revision': False,
        'selection_status': 'gate9_baseline_multiseed',
    },
    'mistral7b_v03': {
        'repo_id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'revision': '0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a',
        'tokenizer_revision': '0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a',
        'allow_unpinned_revision': False,
        'selection_status': 'gate9_baseline_multiseed',
    },
    'llama31_8b': {
        'repo_id': 'meta-llama/Llama-3.1-8B-Instruct',
        'revision': 'main',
        'tokenizer_revision': 'main',
        'allow_unpinned_revision': True,
        'selection_status': 'same_family_latest_weights',
    },
    'ministral3_8b': {
        'repo_id': 'mistralai/Ministral-3-8B-Instruct-2512',
        'revision': 'main',
        'tokenizer_revision': 'main',
        'allow_unpinned_revision': True,
        'selection_status': 'same_family_latest_release',
    },
    'qwen3_8b': {
        'repo_id': 'Qwen/Qwen3-8B',
        'revision': 'main',
        'tokenizer_revision': 'main',
        'allow_unpinned_revision': True,
        'selection_status': 'peer_top_tier_8b',
        'chat_template_kwargs': {'enable_thinking': False},
    },
    'gemma4_e4b': {
        'repo_id': 'google/gemma-4-E4B-it',
        'revision': 'main',
        'tokenizer_revision': 'main',
        'allow_unpinned_revision': True,
        'selection_status': 'peer_top_tier_8b',
    },
    'llama33_70b': {
        'repo_id': 'meta-llama/Llama-3.3-70B-Instruct',
        'revision': 'main',
        'tokenizer_revision': 'main',
        'allow_unpinned_revision': True,
        'selection_status': 'flagship_70b_int8',
    },
}


def build_manifest(block, model_key, prompt_variant, seed, quantized=False, smoke=False):
    spec = MODELS[model_key]
    quant_label = 'int8' if quantized else 'bf16'
    sample = 'data/prepared/smoke24.json' if smoke else 'data/prepared/all_records.json'
    fmt = {
        'intrinsic': 'raw_completion',
        'prompting': 'tokenizer_chat_template',
        'provenance': 'colab_post_gate9_protocol',
    }
    if 'chat_template_kwargs' in spec:
        fmt['chat_template_kwargs'] = dict(spec['chat_template_kwargs'])

    model_cfg = {
        'repo_id': spec['repo_id'],
        'revision': spec['revision'],
        'tokenizer_revision': spec['tokenizer_revision'],
        'allow_unpinned_revision': spec['allow_unpinned_revision'],
        'selection_status': spec['selection_status'],
        'allowed_repos': ALLOWED_REPOS,
        'dtype': 'bfloat16',
        'device_map': {'': 0} if quantized else 'auto',
        'max_memory': {'0': '76GiB', 'cpu': '32GiB'},
        'local_files_only': False,
        'quantization': INT8_PROFILE if quantized else None,
    }
    return {
        'schema_version': 1,
        'purpose': f'Colab {block}: {model_key} {prompt_variant} {quant_label}/48 seed={seed}' + (' (smoke24)' if smoke else ''),
        'block': block,
        'seed': seed,
        'model': model_cfg,
        'format': fmt,
        'generation': {
            'do_sample': False,
            'max_new_tokens': 48,
            'num_beams': 1,
        },
        'sample': sample,
        'inventory': 'data/prepared/inventory.json',
        'output_root': 'results_colab',
        'image_id_env': 'EXPERIMENT_IMAGE_ID',
        'prompt_variant': prompt_variant,
    }


def write_manifest(block, filename, manifest):
    target = OUT_DIR / block / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    counts = {}

    # Smoke manifests (24 items) for all new models before full runs
    smoke_specs = [
        ('llama31_8b', 'refined', False),
        ('llama31_8b', 'refined', True),
        ('ministral3_8b', 'refined', False),
        ('qwen3_8b', 'refined', False),
        ('gemma4_e4b', 'refined', False),
        ('llama33_70b', 'refined', True),
    ]
    for model_key, prompt, quant in smoke_specs:
        q = 'int8' if quant else 'bf16'
        write_manifest('smoke', f'{model_key}-{prompt}-{q}_48-seed2026.json',
                       build_manifest('smoke', model_key, prompt, 2026, quantized=quant, smoke=True))
    counts['smoke'] = len(smoke_specs)

    # Bloco A: Gate 9 baselines across 4 new seeds (2027-2030)
    a_count = 0
    for model_key in ['llama3_8b', 'mistral7b_v03']:
        for prompt in PROMPTS_MAIN:
            for seed in SEEDS_4_NEW:
                write_manifest('bloco_a', f'{model_key}-{prompt}-bf16_48-seed{seed}.json',
                               build_manifest('bloco_a', model_key, prompt, seed, quantized=False))
                a_count += 1
    counts['bloco_a'] = a_count

    # Bloco B: Latest weights & peer top-tier 8B models (5 seeds in bf16 + 3 seeds Llama 3.1 int8 bridge)
    b_count = 0
    for model_key in ['llama31_8b', 'ministral3_8b', 'qwen3_8b', 'gemma4_e4b']:
        for prompt in PROMPTS_MAIN:
            for seed in SEEDS_5:
                write_manifest('bloco_b', f'{model_key}-{prompt}-bf16_48-seed{seed}.json',
                               build_manifest('bloco_b', model_key, prompt, seed, quantized=False))
                b_count += 1
    for seed in SEEDS_3:
        write_manifest('bloco_b', f'llama31_8b-refined-int8_48-seed{seed}.json',
                       build_manifest('bloco_b', 'llama31_8b', 'refined', seed, quantized=True))
        b_count += 1
    counts['bloco_b'] = b_count

    # Bloco C: 70B INT8 (3 seeds, refined first, then original & evaluation_gender_bias)
    c_count = 0
    for prompt in ['refined', 'original', 'evaluation_gender_bias']:
        for seed in SEEDS_3:
            write_manifest('bloco_c', f'llama33_70b-{prompt}-int8_48-seed{seed}.json',
                           build_manifest('bloco_c', 'llama33_70b', prompt, seed, quantized=True))
            c_count += 1
    counts['bloco_c'] = c_count

    print('Generated manifests:', json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
