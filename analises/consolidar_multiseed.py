"""Consolidate Gate 9 and Colab multi-seed runs using parser_v15."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser_v15
from protocol import metrics as compute_metrics


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_jsonl(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def osb_vector_combined(rows, condition, occupations):
    """Aggregate O-SB across Type 1 + Type 2 per occupation (Gate 9 definition)."""
    group = [r for r in rows if r['condition'] == condition and r['status'] == 'ok']
    vec = []
    for occ in occupations:
        counts = {}
        for gender in ['male', 'female']:
            exposed = [r for r in group if r['gender'] == gender and occ in r['candidates']]
            sel = sum(r.get('selection_prediction') == occ for r in exposed)
            counts[gender] = sel / len(exposed) if exposed else None
        if counts['male'] is None or counts['female'] is None:
            vec.append(None)
        else:
            vec.append(counts['male'] - counts['female'])
    return vec


def precision_label(cfg):
    q = cfg['model'].get('quantization')
    if not q:
        return f"{cfg['model'].get('dtype', 'bf16')}/{cfg['generation']['max_new_tokens']}"
    if q.get('load_in_8bit'):
        return f"int8/{cfg['generation']['max_new_tokens']}"
    if q.get('load_in_4bit'):
        return f"nf4/{cfg['generation']['max_new_tokens']}"
    return f"quant/{cfg['generation']['max_new_tokens']}"


def reparse_rows(rows, registry=None):
    reparsed = []
    for r in rows:
        row = dict(r)
        if row['condition'] in ('zero_shot', 'three_shot') and row.get('status') == 'ok':
            parsed = parser_v15.parse_record(row, registry=registry)
            rule = parsed['rule_output']
            sel = parsed['selection_prediction']
            row['selection_prediction'] = sel
            row['prediction'] = sel
            row['rpa_prediction'] = None if sel in (None, 'Unknown') else sel
            row['v15_tier'] = parsed['v15_processing']['tier']
            row['letter_text_conflict'] = rule.get('letter_text_conflict', False)
        reparsed.append(row)
    return reparsed


def summarize_run(folder, occupations, registry=None):
    cfg = load_json(folder / 'config.json')
    state = load_json(folder / 'run.json')
    if state.get('status') != 'inference_finished_needs_manual_review':
        return None, None
    raw_rows = load_jsonl(folder / 'responses.jsonl')
    rows = reparse_rows(raw_rows, registry=registry)
    m = compute_metrics(rows, occupations)

    def acc_type2(cond):
        t2 = [r for r in rows if r['condition'] == cond and r['status'] == 'ok' and r['type'] == 2]
        if not t2:
            return None
        return sum(r.get('selection_prediction') == r['gold'] for r in t2) / len(t2)

    def asb_mean(cond):
        bias = m['conditions'][cond]['bias_by_type']
        t1 = bias['1']['a_sb_40'] if bias['1']['a_sb_40'] is not None else bias['1']['mean_absolute_defined_subset']
        t2 = bias['2']['a_sb_40'] if bias['2']['a_sb_40'] is not None else bias['2']['mean_absolute_defined_subset']
        return (t1 + t2) / 2.0 if t1 is not None and t2 is not None else None

    def unk_null(cond):
        sub = [r for r in rows if r['condition'] == cond and r['status'] == 'ok']
        unk = sum(r.get('selection_prediction') == 'Unknown' for r in sub)
        nul = sum(r.get('selection_prediction') is None for r in sub)
        return unk, nul

    v_intr = osb_vector_combined(rows, 'intrinsic', occupations)
    v_0s = osb_vector_combined(rows, 'zero_shot', occupations)
    v_3s = osb_vector_combined(rows, 'three_shot', occupations)
    unk_0, nul_0 = unk_null('zero_shot')
    unk_3, nul_3 = unk_null('three_shot')

    summary = {
        'run_dir': folder.name,
        'repo_id': cfg['model']['repo_id'],
        'precision': precision_label(cfg),
        'prompt_variant': cfg.get('prompt_variant', 'original'),
        'seed': cfg['seed'],
        'n_records': len(rows) // 3,
        'acc_intrinsic': acc_type2('intrinsic'),
        'acc_zero_shot': acc_type2('zero_shot'),
        'acc_three_shot': acc_type2('three_shot'),
        'asb_intrinsic': asb_mean('intrinsic'),
        'asb_zero_shot': asb_mean('zero_shot'),
        'asb_three_shot': asb_mean('three_shot'),
        'unk_zero_shot': unk_0,
        'null_zero_shot': nul_0,
        'unk_three_shot': unk_3,
        'null_three_shot': nul_3,
        'rho_zero_shot': pearson(v_intr, v_0s),
        'rho_three_shot': pearson(v_intr, v_3s),
    }
    return summary, rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', action='append', required=True, help='One or more results directories')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--inventory', default=str(ROOT / 'data' / 'prepared' / 'inventory.json'))
    parser.add_argument('--use-registry', action='store_true', help='Use Gate 9 frozen contextual registry')
    args = parser.parse_args()

    occupations = load_json(args.inventory)['occupations']
    registry = parser_v15.load_frozen_registry() if args.use_registry else None

    run_summaries = []
    grouped_rows = defaultdict(list)  # (repo_id, precision, prompt_variant) -> [(seed, rows)]

    for root_str in args.results_root:
        root_path = Path(root_str).resolve()
        if not root_path.exists():
            continue
        for folder in sorted(root_path.iterdir()):
            if not folder.is_dir() or not (folder / 'responses.jsonl').exists():
                continue
            summary, rows = summarize_run(folder, occupations, registry=registry)
            if summary is None:
                continue
            run_summaries.append(summary)
            key = (summary['repo_id'], summary['precision'], summary['prompt_variant'])
            grouped_rows[key].append((summary['seed'], rows))

    out_dir = Path(args.output_dir).resolve()
    if run_summaries:
        write_csv(out_dir / 'runs_summary.csv', run_summaries, list(run_summaries[0].keys()))

    # Multi-seed aggregation
    agg_rows = []
    flip_rows = []
    metric_cols = [
        'acc_intrinsic', 'acc_zero_shot', 'acc_three_shot',
        'asb_intrinsic', 'asb_zero_shot', 'asb_three_shot',
        'rho_zero_shot', 'rho_three_shot',
    ]
    by_group = defaultdict(list)
    for s in run_summaries:
        by_group[(s['repo_id'], s['precision'], s['prompt_variant'])].append(s)

    for (repo_id, prec, prompt), items in sorted(by_group.items()):
        agg = {
            'repo_id': repo_id,
            'precision': prec,
            'prompt_variant': prompt,
            'n_seeds': len(items),
            'seeds': ','.join(str(x['seed']) for x in sorted(items, key=lambda d: d['seed'])),
        }
        for col in metric_cols:
            vals = [x[col] for x in items if x[col] is not None]
            agg[f'{col}_mean'] = statistics.mean(vals) if vals else None
            agg[f'{col}_std'] = statistics.stdev(vals) if len(vals) >= 2 else 0.0 if len(vals) == 1 else None
        agg_rows.append(agg)

        # Item-level flip rate across seeds for this group
        seed_runs = grouped_rows[(repo_id, prec, prompt)]
        if len(seed_runs) >= 2:
            for cond in ['intrinsic', 'zero_shot', 'three_shot']:
                per_id = defaultdict(set)
                for _, rows in seed_runs:
                    for r in rows:
                        if r['condition'] == cond and r['status'] == 'ok':
                            per_id[r['id']].add(r.get('selection_prediction'))
                n_items = len(per_id)
                n_flipped = sum(1 for preds in per_id.values() if len(preds) > 1)
                flip_rows.append({
                    'repo_id': repo_id,
                    'precision': prec,
                    'prompt_variant': prompt,
                    'condition': cond,
                    'n_seeds': len(seed_runs),
                    'n_items': n_items,
                    'n_flipped': n_flipped,
                    'flip_rate': n_flipped / n_items if n_items else None,
                })

    if agg_rows:
        write_csv(out_dir / 'multiseed_summary.csv', agg_rows, list(agg_rows[0].keys()))
    if flip_rows:
        write_csv(out_dir / 'flip_rates.csv', flip_rows, list(flip_rows[0].keys()))

    print(json.dumps({
        'runs_processed': len(run_summaries),
        'groups': len(agg_rows),
        'flip_rows': len(flip_rows),
        'output_dir': str(out_dir),
    }, indent=2))


if __name__ == '__main__':
    main()
