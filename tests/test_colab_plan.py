import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class TestColabPlan(unittest.TestCase):
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
            self.assertIn('[1/6] START:', res.stdout)
            self.assertIn('llama33_70b-refined-int8_48-seed2026.json', res.stdout)

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


if __name__ == '__main__':
    unittest.main()
