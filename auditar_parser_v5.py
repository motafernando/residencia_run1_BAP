"""Compare experimental response interpretation with the frozen v4 baseline."""
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from consolidar_resultados_v4 import CONDITIONS, ROOT, RUNS, pooled_osb, summarize
from parser_v5 import parse_response
from protocol import metrics


def main():
    target = ROOT / 'analises' / 'parser-v5-exploratorio-20261027-v11'
    target.mkdir(parents=True, exist_ok=False)
    occupations = json.loads((ROOT / 'data/prepared/inventory.json').read_text(encoding='utf8'))['occupations']
    results, all_changes, sample_groups = {}, [], defaultdict(list)
    for name, relative in RUNS.items():
        base_summary, original = summarize(name, ROOT / relative, occupations)
        newer, changes = [], Counter()
        grouped = defaultdict(Counter)
        for key, item in original.items():
            row = dict(item)
            if row['condition'] != 'intrinsic':
                old = row['parsing']
                parsed = parse_response(row['generated_text'], row['candidates'], row['option_map'])
                if parsed['rpa_prediction'] != row['rpa_prediction']:
                    raise ValueError(f'{name} {key}: frozen strict RPA changed')
                row.update(parsing=parsed, selection_prediction=parsed['selection_prediction'],
                           rpa_choice_prediction=parsed['rpa_choice_prediction'])
                a, b = old['selection_prediction'], parsed['selection_prediction']
                kind = ('recovered' if a is None and b is not None else 'lost' if a is not None and b is None else
                        'switched' if a is not None and b is not None and a != b else 'unchanged')
                condition = row['condition']
                changes[f'{condition}:{kind}'] += 1
                changes[f'{condition}:letter_text_conflict'] += bool(parsed.get('letter_text_conflict'))
                changes[f'{condition}:answer_reasoning_conflict'] += bool(parsed.get('answer_reasoning_conflict'))
                changes[f'{condition}:ambiguous'] += bool(parsed.get('choice_ambiguous'))
                if kind != 'unchanged':
                    grouped[condition][f'{kind}:type{row["type"]}:{row["gender"]}'] += 1
                audit = {'run': name, 'id': row['id'], 'condition': condition, 'type': row['type'],
                         'gender': row['gender'], 'sentence': row['sentence'], 'candidates': row['candidates'],
                         'option_map': row['option_map'], 'generated_text': row['generated_text'],
                         'v4_selection': a, 'v5_selection': b, 'v5_status': parsed['status'],
                         'selection_source': parsed['selection_source'], 'evidence': parsed.get('choice_evidence'),
                         'reasoning_evidence': parsed.get('reasoning_evidence'),
                         'excluded_generated_question': parsed.get('excluded_generated_question'),
                         'needs_review': parsed.get('choice_needs_review'),
                         'label_conflict': parsed.get('letter_text_conflict'),
                         'reasoning_conflict': parsed.get('answer_reasoning_conflict')}
                if kind != 'unchanged':
                    all_changes.append({**audit, 'gold_for_task_analysis': row['gold']})
                    sample_groups[(name, condition, kind)].append(audit)
                elif parsed.get('letter_text_conflict') or parsed.get('answer_reasoning_conflict'):
                    sample_groups[(name, condition, 'flagged')].append(audit)
            newer.append(row)
        old_rows = list(original.values())
        old_metrics, new_metrics = metrics(old_rows, occupations), metrics(newer, occupations)
        old_osb, new_osb = pooled_osb(old_rows, occupations), pooled_osb(newer, occupations)
        conditions = {}
        for condition in CONDITIONS:
            old_group = [r for r in old_rows if r['condition'] == condition]
            new_group = [r for r in newer if r['condition'] == condition]
            old_t2 = [r for r in old_group if r['type'] == 2]
            new_t2 = [r for r in new_group if r['type'] == 2]
            a, b = old_metrics['conditions'][condition], new_metrics['conditions'][condition]
            if a['rpa_type2'] != b['rpa_type2']:
                raise ValueError(f'{name} {condition}: strict RPA aggregate changed')
            conditions[condition] = {
                'n': len(old_group), 'n_type2': len(old_t2), 'rpa_strict': a['rpa_type2'],
                'choice_correct_v4': sum(r['rpa_choice_prediction'] == r['gold'] for r in old_t2),
                'choice_correct_v5': sum(r['rpa_choice_prediction'] == r['gold'] for r in new_t2),
                'unresolved_v4': sum(r['selection_prediction'] is None for r in old_group),
                'unresolved_v5': sum(r['selection_prediction'] is None for r in new_group),
                'unknown_v4': sum(r['selection_prediction'] == 'Unknown' for r in old_group),
                'unknown_v5': sum(r['selection_prediction'] == 'Unknown' for r in new_group),
                'a_sb_v4': {str(t): a['bias_by_type'][str(t)]['a_sb_40'] for t in (1, 2)},
                'a_sb_v5': {str(t): b['bias_by_type'][str(t)]['a_sb_40'] for t in (1, 2)},
                'occupations_v4': {str(t): a['bias_by_type'][str(t)]['occupations'] for t in (1, 2)},
                'occupations_v5': {str(t): b['bias_by_type'][str(t)]['occupations'] for t in (1, 2)},
                'changes_by_type_gender': dict(grouped[condition]),
            }
        results[name] = {'source_responses': base_summary['response_file'], 'source_sha256': base_summary['sha256'],
                         'changes': dict(changes), 'conditions': conditions,
                         'rho_v4': {c: statistics.correlation(old_osb['intrinsic'], old_osb[c]) for c in CONDITIONS[1:]},
                         'rho_v5': {c: statistics.correlation(new_osb['intrinsic'], new_osb[c]) for c in CONDITIONS[1:]}}
    with (target / 'mudancas.jsonl').open('x', encoding='utf8') as out:
        for record in all_changes:
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
    with (target / 'amostra-auditoria-sem-gabarito.jsonl').open('x', encoding='utf8') as out:
        for key, group in sorted(sample_groups.items()):
            ranked = sorted(group, key=lambda r: hashlib.sha256(f"{r['run']}:{r['id']}:{r['condition']}".encode()).hexdigest())
            for record in ranked[:10]:
                out.write(json.dumps({**record, 'sample_stratum': key[2]}, ensure_ascii=False) + '\n')
    with (target / 'amostra-cega-sem-sugestao.jsonl').open('x', encoding='utf8') as out:
        for key, group in sorted(sample_groups.items()):
            ranked = sorted(group, key=lambda r: hashlib.sha256(f"{r['run']}:{r['id']}:{r['condition']}".encode()).hexdigest())
            for record in ranked[:10]:
                blinded = {k: record[k] for k in ('run', 'id', 'condition', 'type', 'gender', 'sentence',
                                                   'candidates', 'option_map', 'generated_text')}
                out.write(json.dumps(blinded, ensure_ascii=False) + '\n')
    data = {'v5_parser_sha256': hashlib.sha256((ROOT / 'parser_v5.py').read_bytes()).hexdigest(),
            'v4_parser_sha256': hashlib.sha256((ROOT / 'parser_v4.py').read_bytes()).hexdigest(),
            'policy': 'v5 is an experimental, gold-independent interpretation; strict RPA is frozen to v4.',
            'runs': results}
    (target / 'resultados.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('wrote', target, 'changes', len(all_changes))
    for name, result in results.items():
        print(name, result['changes'])


if __name__ == '__main__':
    main()
