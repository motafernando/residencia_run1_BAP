"""Recompute paired WinoBias comparisons without modifying inference results."""
import hashlib
import json
from pathlib import Path

from consolidar_resultados_v4 import CONDITIONS, ROOT, RUNS, summarize
from protocol import metrics


def transitions(previous, current):
    result = {'n': 0, 'correct_to_not': 0, 'not_to_correct': 0, 'both_correct': 0, 'both_not': 0}
    for row in previous:
        if row['type'] != 2:
            continue
        old = row['rpa_prediction'] == row['gold']
        new = current[row['id']]['rpa_prediction'] == row['gold']
        result['n'] += 1
        result[('both_correct' if old else 'not_to_correct') if new else ('correct_to_not' if old else 'both_not')] += 1
    return result


def main():
    target = ROOT / 'analises' / 'comparativo-mistral-20261027'
    target.mkdir(parents=True, exist_ok=False)
    occupations = json.loads((ROOT / 'data/prepared/inventory.json').read_text(encoding='utf8'))['occupations']
    names = ('mistral_original', 'mistral_refined', 'mistral_evaluation_task', 'mistral_evaluation_gender_bias', 'llama_original', 'llama_refined')
    summaries, rows, details = {}, {}, {}
    for name in names:
        path = ROOT / RUNS[name]
        summaries[name], rows[name] = summarize(name, path, occupations)
        details[name] = metrics(list(rows[name].values()), occupations)['conditions']
    base = rows['mistral_original']
    for name in names[1:4]:
        other = rows[name]
        if base.keys() != other.keys():
            raise ValueError(f'{name}: sample keys differ')
        for key in base:
            for field in ('gold', 'type', 'gender', 'stereotype', 'sentence', 'candidates', 'option_map'):
                if base[key][field] != other[key][field]:
                    raise ValueError(f'{name}: paired field differs: {field}, {key}')
    paired = {}
    for reference, name in [('mistral_original', n) for n in names[1:4]] + [('mistral_refined', n) for n in names[2:4]]:
        key = f'{reference}__{name}'
        paired[key] = {}
        for condition in CONDITIONS:
            a = [row for (identifier, c), row in rows[reference].items() if c == condition]
            b = {identifier: row for (identifier, c), row in rows[name].items() if c == condition}
            by_type = {str(t): sum(row['selection_prediction'] != b[row['id']]['selection_prediction']
                                   for row in a if row['type'] == t) for t in (1, 2)}
            paired[key][condition] = {'rpa_transition': transitions(a, b), 'selection_changed_by_type': by_type}
    strata = {}
    for name in names:
        strata[name] = {}
        for condition in CONDITIONS:
            section = details[name][condition]
            strata[name][condition] = {'rpa_by_gender': section['rpa_type2_by_gender'],
                                       'rpa_by_stereotype': section['rpa_type2_by_stereotype'],
                                       'bias_by_type': section['bias_by_type']}
    out = {'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'parser_sha256': hashlib.sha256((ROOT / 'parser_v4.py').read_bytes()).hexdigest(),
           'paired_fields_verified': ['id', 'condition', 'gold', 'type', 'gender', 'stereotype', 'sentence', 'candidates', 'option_map'],
           'summaries': summaries, 'paired': paired, 'strata_and_occupations': strata}
    (target / 'dados.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    for name, values in paired.items():
        print(name, {c: v['rpa_transition'] for c, v in values.items()})
    print('saved', target / 'dados.json')


if __name__ == '__main__':
    main()
