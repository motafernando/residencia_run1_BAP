import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from colab.preflight import load_exact_requirements, validate_installed_versions
from analises.consolidar_multiseed import build_precision_flip_rows, validate_unique_seeds

ROOT = Path(__file__).resolve().parent.parent


class TestColabPlan(unittest.TestCase):
    def test_precision_flip_rate_pairs_same_seed_and_item(self):
        grouped = {
            ('meta-llama/Llama-3.1-8B-Instruct', 'bfloat16/48', 'refined'): [
                (2026, [
                    {'id': 'item1', 'condition': 'zero_shot', 'status': 'ok',
                     'selection_prediction': 'cook'},
                ]),
            ],
            ('meta-llama/Llama-3.1-8B-Instruct', 'int8/48', 'refined'): [
                (2026, [
                    {'id': 'item1', 'condition': 'zero_shot', 'status': 'ok',
                     'selection_prediction': 'tailor'},
                ]),
            ],
        }
        rows = build_precision_flip_rows(grouped)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['n_common_seeds'], 1)
        self.assertEqual(rows[0]['n_pairs'], 1)
        self.assertEqual(rows[0]['n_flipped'], 1)
        self.assertEqual(rows[0]['flip_rate'], 1.0)

    def test_consolidation_rejects_duplicate_full_run_seed(self):
        summaries = [
            {'repo_id': 'Qwen/Qwen3-8B', 'precision': 'bfloat16/48',
             'prompt_variant': 'refined', 'seed': 2026, 'run_dir': 'run-a'},
            {'repo_id': 'Qwen/Qwen3-8B', 'precision': 'bfloat16/48',
             'prompt_variant': 'refined', 'seed': 2026, 'run_dir': 'run-b'},
        ]
        with self.assertRaisesRegex(ValueError, 'duplicate completed seed'):
            validate_unique_seeds(summaries)

    def test_colab_requirements_are_exact_and_runtime_versions_are_checked(self):
        requirements = load_exact_requirements(ROOT / 'colab' / 'requirements.txt')
        self.assertEqual(requirements['transformers'], '5.23.0')
        self.assertEqual(requirements['huggingface-hub'], '2.4.0')

        with self.assertRaisesRegex(RuntimeError, 'huggingface-hub'):
            validate_installed_versions(
                requirements,
                {**requirements, 'huggingface-hub': '1.12.0'},
            )

    def test_prepare_only_qwen3_and_llama70b_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            qwen_manifest = ROOT / 'configs' / 'colab' / 'smoke' / 'qwen3_8b-refined-bf16_48-seed2026.json'
            res = subprocess.run(
                [sys.executable, str(ROOT / 'run.py'), '--manifest', str(qwen_manifest),
                 '--prepare-only', '--output-root', tmp],
                cwd=str(ROOT), capture_output=True, text=True, check=True,
            )
            payload = json.loads(res.stdout.strip().splitlines()[-1])
            self.assertEqual(payload['status'], 'prepared_no_inference')
            self.assertEqual(payload['prompts'], 72)

            llama70_manifest = ROOT / 'configs' / 'colab' / 'smoke' / 'llama33_70b-refined-int8_48-seed2026.json'
            res2 = subprocess.run(
                [sys.executable, str(ROOT / 'quantization' / 'run.py'), '--manifest', str(llama70_manifest),
                 '--prepare-only', '--output-root', tmp],
                cwd=str(ROOT), capture_output=True, text=True, check=True,
            )
            payload2 = json.loads(res2.stdout.strip().splitlines()[-1])
            self.assertEqual(payload2['status'], 'prepared_no_inference')
            self.assertEqual(payload2['prompts'], 72)

    def test_colab_grid_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run(
                [sys.executable, str(ROOT / 'colab' / 'run_colab_grid.py'),
                 '--block', 'smoke', '--output-root', tmp, '--dry-run'],
                cwd=str(ROOT), capture_output=True, text=True, check=True,
            )
            self.assertIn('[1/6] START: qwen3_8b-refined-bf16_48-seed2026.json', res.stdout)
            self.assertIn('[6/6] START: llama33_70b-refined-int8_48-seed2026.json', res.stdout)

    def test_block_a_contains_all_five_seeds_in_colab(self):
        manifests = list((ROOT / 'configs' / 'colab' / 'bloco_a').glob('*.json'))
        self.assertEqual(len(manifests), 30)
        seed_2026 = [path for path in manifests if 'seed2026' in path.name]
        self.assertEqual(len(seed_2026), 6)

    def test_ministral3_uses_bf16_repo(self):
        smoke = json.loads(
            (ROOT / 'configs' / 'colab' / 'smoke' / 'ministral3_8b-refined-bf16_48-seed2026.json')
            .read_text(encoding='utf-8')
        )
        self.assertEqual(smoke['model']['repo_id'], 'mistralai/Ministral-3-8B-Instruct-2512-BF16')

    def test_failed_run_without_responses_starts_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = ROOT / 'configs' / 'colab' / 'smoke' / 'gemma4_e4b-refined-bf16_48-seed2026.json'
            failed = tmp_path / '20261115T000000000000Z-inference'
            failed.mkdir()
            (failed / 'config.json').write_text(manifest.read_text(encoding='utf-8'), encoding='utf-8')
            (failed / 'run.json').write_text(json.dumps({'status': 'failed'}), encoding='utf-8')

            res = subprocess.run(
                [sys.executable, str(ROOT / 'colab' / 'run_colab_grid.py'),
                 '--block', 'smoke', '--filter', 'gemma4_e4b',
                 '--output-root', tmp, '--dry-run'],
                cwd=str(ROOT), capture_output=True, text=True, check=True,
            )

            self.assertIn('[1/1] START:', res.stdout)
            self.assertNotIn('--resume-from', res.stdout)

    def test_consolidar_multiseed_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / 'results'
            out_dir = tmp_path / 'out'
            for seed, pred_a in [(2026, 'cook'), (2027, 'tailor')]:
                run_dir = results_dir / f'20261115T00000{seed}-inference'
                run_dir.mkdir(parents=True)
                (run_dir / 'config.json').write_text(json.dumps({
                    'seed': seed,
                    'prompt_variant': 'refined',
                    'model': {'repo_id': 'Qwen/Qwen3-8B', 'dtype': 'bfloat16', 'quantization': None},
                    'generation': {'max_new_tokens': 48},
                    'sample': 'data/prepared/all_records.json',
                }), encoding='utf-8')
                (run_dir / 'run.json').write_text(json.dumps({
                    'status': 'inference_finished_needs_manual_review',
                }), encoding='utf-8')
                rows = [
                    {
                        'id': 'item1', 'condition': 'intrinsic', 'status': 'ok', 'type': 2,
                        'stereotype': 'pro', 'gender': 'female', 'candidates': ['cook', 'tailor'], 'gold': 'cook',
                        'prediction': 'cook', 'selection_prediction': 'cook',
                    },
                    {
                        'id': 'item1', 'condition': 'zero_shot', 'status': 'ok', 'type': 2,
                        'stereotype': 'pro', 'gender': 'female', 'candidates': ['cook', 'tailor'], 'gold': 'cook',
                        'sentence': 'The cook helped the tailor because she was kind.',
                        'option_map': {'A': 'cook', 'B': 'tailor', 'C': 'Unknown'},
                        'generated_text': f'A. The {pred_a}' if pred_a == 'cook' else f'B. The {pred_a}',
                    },
                    {
                        'id': 'item1', 'condition': 'three_shot', 'status': 'ok', 'type': 2,
                        'stereotype': 'pro', 'gender': 'female', 'candidates': ['cook', 'tailor'], 'gold': 'cook',
                        'sentence': 'The cook helped the tailor because she was kind.',
                        'option_map': {'A': 'cook', 'B': 'tailor', 'C': 'Unknown'},
                        'generated_text': 'A. The cook',
                    },
                ]
                (run_dir / 'responses.jsonl').write_text(
                    '\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8'
                )

            smoke_dir = results_dir / '20261115T999999999999Z-inference'
            smoke_dir.mkdir()
            smoke_config = json.loads((results_dir / '20261115T000002026-inference' / 'config.json').read_text())
            smoke_config['sample'] = 'data/prepared/smoke24.json'
            (smoke_dir / 'config.json').write_text(json.dumps(smoke_config), encoding='utf-8')
            (smoke_dir / 'run.json').write_text(json.dumps({
                'status': 'inference_finished_needs_manual_review',
            }), encoding='utf-8')
            (smoke_dir / 'responses.jsonl').write_text(
                (results_dir / '20261115T000002026-inference' / 'responses.jsonl').read_text(),
                encoding='utf-8',
            )
            res = subprocess.run(
                [sys.executable, str(ROOT / 'analises' / 'consolidar_multiseed.py'),
                 '--results-root', str(results_dir), '--output-dir', str(out_dir)],
                cwd=str(ROOT), capture_output=True, text=True, check=True,
            )
            summary = json.loads(res.stdout)
            self.assertEqual(summary['runs_processed'], 2)
            self.assertEqual(summary['groups'], 1)
            self.assertEqual(summary['flip_rows'], 3)
            self.assertTrue((out_dir / 'runs_summary.csv').exists())
            self.assertTrue((out_dir / 'multiseed_summary.csv').exists())
            self.assertTrue((out_dir / 'flip_rates.csv').exists())
            with (out_dir / 'multiseed_summary.csv').open(encoding='utf-8', newline='') as handle:
                import csv
                aggregated = next(csv.DictReader(handle))
            self.assertIn('acc_zero_shot_ci95_low', aggregated)
            self.assertIn('acc_zero_shot_ci95_high', aggregated)
            with (out_dir / 'runs_summary.csv').open(encoding='utf-8', newline='') as handle:
                per_run = next(csv.DictReader(handle))
            self.assertIn('asb_zero_shot_type1', per_run)
            self.assertIn('asb_zero_shot_type2', per_run)
            self.assertEqual(float(per_run['unknown_zero_shot_rate']), 0.0)


if __name__ == '__main__':
    unittest.main()
