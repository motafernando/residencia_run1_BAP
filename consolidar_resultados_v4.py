"""Recompute six completed WinoBias runs without modifying historical outputs."""
import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from parser_v3 import parse_response as parse_v3
from parser_v4 import parse_response as parse_v4
from protocol import metrics


ROOT = Path(__file__).resolve().parent
RUNS = {
    'mistral_original': 'results/20261021T224558462021Z-inference',
    'mistral_refined': 'results/grid-20261026t211027z/full/refined/20261026T212312065659Z-inference',
    'mistral_evaluation_task': 'results/grid-20261026t211027z/full/evaluation_task/20261027T001137163705Z-inference',
    'mistral_evaluation_gender_bias': 'results/grid-20261026t211027z/full/evaluation_gender_bias/20261027T030929544031Z-inference',
    'llama_original': 'results/llama-fullgpu-round-20261027/full/20261027T125049826965Z-inference',
    'llama_refined': 'results/llama-refined-fullgpu-20261027/full/20261027T190835747818Z-inference',
}
CONDITIONS = ('intrinsic', 'zero_shot', 'three_shot')


def pooled_osb(rows, occupations):
    result = {}
    for condition in CONDITIONS:
        group = [r for r in rows if r['condition'] == condition and r['status'] == 'ok']
        values = []
        for occ in occupations:
            rates = []
            for gender in ('male', 'female'):
                exposed = [r for r in group if r['gender'] == gender and occ in r['candidates']]
                if not exposed:
                    raise ValueError(f'No eligible samples for {condition}, {occ}, {gender}')
                rates.append(sum(r['selection_prediction'] == occ for r in exposed) / len(exposed))
            values.append(rates[0] - rates[1])
        result[condition] = values
    return result


def summarize(name, run_dir, occupations):
    path = run_dir / 'responses.jsonl'
    rows, changes = [], Counter()
    with path.open(encoding='utf8') as handle:
        for line in handle:
            source = json.loads(line)
            row = dict(source)
            if row['condition'] == 'intrinsic':
                row['selection_prediction'] = row['prediction']
                row['rpa_prediction'] = row['prediction']
                row['rpa_choice_prediction'] = row['prediction']
            elif row['status'] == 'ok':
                old = parse_v3(row['generated_text'], row['candidates'], row['option_map'])
                parsed = parse_v4(row['generated_text'], row['candidates'], row['option_map'])
                for key in ('selection_prediction', 'rpa_prediction', 'rpa_choice_prediction', 'status', 'letter_text_conflict'):
                    changes[f'{row["condition"]}:{key}'] += old.get(key) != parsed.get(key)
                row.update(parsing=parsed, selection_prediction=parsed['selection_prediction'],
                           rpa_prediction=parsed['rpa_prediction'], rpa_choice_prediction=parsed['rpa_choice_prediction'])
            rows.append(row)
    keys = [(r['id'], r['condition']) for r in rows]
    if len(rows) != 9504 or len(set(keys)) != 9504 or any(r['status'] != 'ok' for r in rows):
        raise ValueError(f'{name}: incomplete, duplicate, or failed records')
    m = metrics(rows, occupations)
    pooled = pooled_osb(rows, occupations)
    result = {'response_file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'parser': 'v4_label_boundary', 'rows': len(rows), 'v3_v4_changes': dict(changes),
              'conditions': {}, 'rho_overall_40': {c: statistics.correlation(pooled['intrinsic'], pooled[c])
                                                    for c in ('zero_shot', 'three_shot')}}
    for condition in CONDITIONS:
        data = [r for r in rows if r['condition'] == condition]
        type2 = [r for r in data if r['type'] == 2]
        v = m['conditions'][condition]
        choice_correct = sum(r['rpa_choice_prediction'] == r['gold'] for r in type2)
        a1, a2 = (v['bias_by_type'][str(i)]['a_sb_40'] for i in (1, 2))
        lengths = [len(r['generated_token_ids']) for r in data if 'generated_token_ids' in r]
        result['conditions'][condition] = {
            'count': len(data), 'rpa_strict_correct': v['rpa_type2']['correct'], 'rpa_n': len(type2),
            'rpa_strict': v['rpa_type2']['rate'], 'rpa_choice_correct': choice_correct,
            'rpa_choice': choice_correct / len(type2), 'a_sb_type1': a1, 'a_sb_type2': a2,
            'a_sb_average_types': (a1 + a2) / 2, 'explicit_unknown': sum(r['selection_prediction'] == 'Unknown' for r in data),
            'unresolved_selection': sum(r['selection_prediction'] is None for r in data),
            'multiple_candidate_mentions_full_text': v['full_text_multiple_candidate_mentions'],
            'bare_letter': sum(r.get('parsing', {}).get('selection_source') == 'letter_only' for r in data),
            'letter_text_conflicts': sum(bool(r.get('parsing', {}).get('letter_text_conflict')) for r in data),
            'strict_unresolved_type2': v['rpa_unresolved_type2'],
            'generated_tokens_mean': statistics.mean(lengths) if lengths else None,
            'generated_at_48_token_limit': sum(n >= 48 for n in lengths),
            'seconds_sum': sum(r.get('seconds', 0) for r in data),
        }
    return result, {(r['id'], r['condition']): r for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='analises/consolidacao-20261027-v2')
    args = parser.parse_args()
    target = ROOT / args.output
    target.mkdir(parents=True, exist_ok=False)
    occupations = json.loads((ROOT / 'data/prepared/inventory.json').read_text(encoding='utf8'))['occupations']
    all_results, datasets = {}, {}
    for name, relative in RUNS.items():
        all_results[name], datasets[name] = summarize(name, ROOT / relative, occupations)
    pairs = {}
    for model in ('mistral', 'llama'):
        original, refined = datasets[f'{model}_original'], datasets[f'{model}_refined']
        if original.keys() != refined.keys():
            raise ValueError(f'{model}: sample keys differ across paired runs')
        pairs[model] = {}
        for condition in CONDITIONS:
            a = [r for (key, c), r in original.items() if c == condition]
            pairs[model][condition] = {
                'n': len(a),
                'selection_changed': sum(r['selection_prediction'] != refined[(r['id'], condition)]['selection_prediction'] for r in a),
                'gold_changed': sum(r['gold'] != refined[(r['id'], condition)]['gold'] for r in a),
                'option_map_changed': sum(r['option_map'] != refined[(r['id'], condition)]['option_map'] for r in a),
            }
    out = {'version': 'v4_label_boundary', 'parser_sha256': hashlib.sha256((ROOT / 'parser_v4.py').read_bytes()).hexdigest(),
           'source_policy': 'Original response files are read-only; seed 2026.',
           'rho_policy': 'Pearson across 40 pooled per-occupation O-SB values, intrinsic versus prompting.',
           'runs': all_results, 'paired_original_vs_refined': pairs}
    (target / 'resultados.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('wrote', target / 'resultados.json')
    for name, result in all_results.items():
        print(name, ' '.join(f"{c}:RPA={result['conditions'][c]['rpa_strict']*100:.2f}/choice={result['conditions'][c]['rpa_choice']*100:.2f}/ASB1={result['conditions'][c]['a_sb_type1']*100:.2f}/ASB2={result['conditions'][c]['a_sb_type2']*100:.2f}" for c in CONDITIONS))


if __name__ == '__main__':
    main()
