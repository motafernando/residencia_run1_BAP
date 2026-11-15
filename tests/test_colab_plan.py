import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from colab.preflight import load_exact_requirements, validate_installed_versions
from colab.run_colab_grid import collect_manifests, is_heavy_70b, load_json, resolve_workers
from analises.consolidar_multiseed import build_precision_flip_rows, validate_unique_seeds
from runtime_model import (
    cache_seed_key,
    find_completed_manifest_run,
    load_cross_seed_cache,
    resolve_batch_manifests,
)

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

    def test_priority_blocks_and_worker_defaults(self):
        p1 = collect_manifests('priority1')
        self.assertEqual(len(p1), 39)
        self.assertTrue(all(p.name.startswith('llama33_70b-') for p in p1[:9]))
        self.assertTrue(all(is_heavy_70b(load_json(p)) for p in p1[:9]))
        self.assertFalse(any(is_heavy_70b(load_json(p)) for p in p1[9:]))
        self.assertEqual(sum(p.name.startswith('ministral3_8b-') for p in p1[9:]), 15)
        self.assertEqual(sum(p.name.startswith('qwen3_8b-') for p in p1[9:]), 15)

        p2 = collect_manifests('priority2')
        self.assertEqual(len(p2), 30)
        self.assertEqual(sum(p.name.startswith('gemma4_e4b-') for p in p2), 15)
        self.assertEqual(sum(p.name.startswith('llama31_8b-') for p in p2), 15)
        self.assertFalse(any('int8_48' in p.name for p in p2))

        self.assertEqual(resolve_workers('priority1', None), 3)
        self.assertEqual(resolve_workers('priority2', None), 3)
        self.assertEqual(resolve_workers('bloco_a', None), 1)
        self.assertEqual(resolve_workers('priority1', 2), 2)

        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run(
                [sys.executable, str(ROOT / 'colab' / 'run_colab_grid.py'),
                 '--block', 'priority1', '--max-new-tokens-cap', '24', '--output-root', tmp, '--dry-run'],
                cwd=str(ROOT), capture_output=True, text=True, check=True,
            )
            self.assertIn('[1/39] START: llama33_70b-', res.stdout)
            self.assertIn('[39/39] START: qwen3_8b-', res.stdout)
            self.assertIn('--max-new-tokens-cap 24', res.stdout)

    def test_cross_seed_cache_and_skip_if_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_cfg = {
                'seed': 2026,
                'prompt_variant': 'evaluation_gender_bias',
                'sample': 'data/prepared/all_records.json',
                'inventory': 'data/prepared/inventory.json',
                'model': {
                    'repo_id': 'meta-llama/Llama-3.3-70B-Instruct',
                    'revision': 'main',
                    'tokenizer_revision': 'main',
                    'dtype': 'bfloat16',
                    'quantization': {'method': 'bitsandbytes', 'load_in_8bit': True},
                },
                'format': {'intrinsic': 'raw_completion', 'prompting': 'tokenizer_chat_template'},
                'generation': {'do_sample': False, 'max_new_tokens': 48, 'num_beams': 1},
            }

            done_dir = tmp_path / '20261115T000001-inference'
            done_dir.mkdir()
            (done_dir / 'config.json').write_text(json.dumps(base_cfg), encoding='utf-8')
            (done_dir / 'run.json').write_text(
                json.dumps({'status': 'inference_finished_needs_manual_review'}), encoding='utf-8',
            )
            opt_match = {'A': 'cook', 'B': 'tailor', 'C': 'Unknown'}
            opt_other = {'A': 'tailor', 'B': 'cook', 'C': 'Unknown'}
            rows = [
                {
                    'id': 'item1', 'condition': 'intrinsic', 'status': 'ok',
                    'scores': [{'candidate_id': 'cook', 'sum_log_probability': -1.0}],
                    'prediction': 'cook', 'selection_prediction': 'cook',
                    'rpa_prediction': 'cook', 'tie': False,
                },
                {
                    'id': 'item1', 'condition': 'zero_shot', 'status': 'ok',
                    'option_map': opt_match,
                    'rendered_prompt': 'p0',
                    'effective_generation_config': {'max_new_tokens': 48},
                    'input_token_ids': [1, 2],
                    'generated_token_ids': [10, 11, 12],
                    'generated_text': 'A. The cook',
                },
                {
                    'id': 'item1', 'condition': 'three_shot', 'status': 'ok',
                    'option_map': opt_match,
                    'rendered_prompt': 'p3',
                    'effective_generation_config': {'max_new_tokens': 48},
                    'input_token_ids': [1, 2, 3],
                    'generated_token_ids': list(range(48)),
                    'generated_text': 'long output hitting 48 cap',
                },
            ]
            (done_dir / 'responses.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8',
            )

            running_dir = tmp_path / '20261115T000002-inference'
            running_dir.mkdir()
            cfg_2027 = json.loads(json.dumps(base_cfg))
            cfg_2027['seed'] = 2027
            (running_dir / 'config.json').write_text(json.dumps(cfg_2027), encoding='utf-8')
            (running_dir / 'run.json').write_text(json.dumps({'status': 'running'}), encoding='utf-8')

            failed_dir = tmp_path / '20261115T000003-inference'
            failed_dir.mkdir()
            cfg_2028 = json.loads(json.dumps(base_cfg))
            cfg_2028['seed'] = 2028
            (failed_dir / 'config.json').write_text(json.dumps(cfg_2028), encoding='utf-8')
            (failed_dir / 'run.json').write_text(json.dumps({'status': 'failed'}), encoding='utf-8')

            self.assertEqual(find_completed_manifest_run(tmp_path, base_cfg), done_dir)
            self.assertIsNone(find_completed_manifest_run(tmp_path, cfg_2027))
            self.assertIsNone(find_completed_manifest_run(tmp_path, cfg_2028))

            cache24 = load_cross_seed_cache(tmp_path, cfg_2027, effective_max_new_tokens=24)
            self.assertIn(cache_seed_key('intrinsic', 'item1'), cache24)
            self.assertIn(cache_seed_key('zero_shot', 'item1', opt_match), cache24)
            self.assertNotIn(cache_seed_key('zero_shot', 'item1', opt_other), cache24)
            self.assertNotIn(cache_seed_key('three_shot', 'item1', opt_match), cache24)

            cache48 = load_cross_seed_cache(tmp_path, cfg_2027, effective_max_new_tokens=48)
            self.assertIn(cache_seed_key('three_shot', 'item1', opt_match), cache48)

            expanded = resolve_batch_manifests(
                '/repo/a.json', None, False, None,
                {'batch_sibling_manifests': ['/repo/a.json', '/repo/b.json', '/repo/c.json']},
            )
            self.assertEqual(expanded, ['/repo/a.json', '/repo/b.json', '/repo/c.json'])

    def test_quantization_runner_in_memory_reuse_cache_and_cap(self):
        q_dir = str(ROOT / 'quantization')
        if q_dir not in sys.path:
            sys.path.insert(0, q_dir)
        import quantization.run as qrun

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('prepare.py', 'protocol.py', 'parser_v3.py', 'parser_v4.py', 'resume_journal.py'):
                shutil.copyfile(ROOT / name, root / name)
            records = [
                {
                    'id': 'item1', 'sentence': 'The cook met the tailor and she smiled.',
                    'pronoun': 'she', 'candidates': ['cook', 'tailor'],
                    'candidate_text': {'cook': 'cook', 'tailor': 'tailor'},
                    'type': 2, 'gender': 'female', 'stereotype': 'pro', 'gold': 'cook',
                },
                {
                    'id': 'item2', 'sentence': 'The tailor called the cook because he waited.',
                    'pronoun': 'he', 'candidates': ['tailor', 'cook'],
                    'candidate_text': {'tailor': 'tailor', 'cook': 'cook'},
                    'type': 2, 'gender': 'male', 'stereotype': 'anti', 'gold': 'tailor',
                },
            ]
            (root / 'sample.json').write_text(json.dumps({'records': records}), encoding='utf-8')
            (root / 'inventory.json').write_text(
                json.dumps({'revision': 'test', 'occupations': ['cook', 'tailor']}), encoding='utf-8',
            )
            base_cfg = {
                'sample': 'sample.json', 'inventory': 'inventory.json', 'seed': 2026,
                'prompt_variant': 'original', 'output_root': 'results', 'image_id_env': 'TEST_IMAGE_ID',
                'model': {
                    'repo_id': 'test-70b', 'allowed_repos': ['test-70b'],
                    'revision': 'a' * 40, 'tokenizer_revision': 'a' * 40,
                    'selection_status': 'flagship_70b_int8', 'dtype': 'bfloat16',
                    'device_map': {'': 0}, 'max_memory': {'0': '76GiB'}, 'local_files_only': True,
                    'quantization': {'method': 'bitsandbytes', 'load_in_8bit': True},
                },
                'format': {'prompting': 'raw_completion'},
                'generation': {'do_sample': False, 'max_new_tokens': 48, 'num_beams': 1},
            }
            m1 = root / 'm1.json'
            m2 = root / 'm2.json'
            m1.write_text(json.dumps(base_cfg), encoding='utf-8')
            cfg2 = json.loads(json.dumps(base_cfg))
            cfg2['seed'] = 2027
            m2.write_text(json.dumps(cfg2), encoding='utf-8')

            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'config.json').write_text('{}', encoding='utf-8')

            class FakeLinear8:
                pass

            model = types.SimpleNamespace(
                eval=lambda: None,
                hf_device_map={'': 0},
                is_loaded_in_8bit=True,
                named_modules=lambda: [('proj', FakeLinear8())],
                config=types.SimpleNamespace(quantization_config={'load_in_8bit': True}),
            )
            tokenizer = types.SimpleNamespace(chat_template='template')
            cuda = types.SimpleNamespace(
                is_available=lambda: True, manual_seed_all=lambda _: None,
                get_device_name=lambda _: 'fake-gpu', get_device_capability=lambda _: (10, 0),
                get_arch_list=lambda: [], mem_get_info=lambda: (1, 2),
                max_memory_allocated=lambda: 1, max_memory_reserved=lambda: 1,
                reset_peak_memory_stats=lambda: None, empty_cache=lambda: None,
            )
            torch = types.SimpleNamespace(
                cuda=cuda, manual_seed=lambda _: None,
                version=types.SimpleNamespace(cuda='fake'), bfloat16='fake',
            )
            bnb = types.SimpleNamespace(nn=types.SimpleNamespace(Linear8bitLt=FakeLinear8, Linear4bit=type('L4', (), {})))
            from_pretrained = mock.Mock(return_value=model)
            transformers = types.SimpleNamespace(
                AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
                AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=from_pretrained),
                BitsAndBytesConfig=lambda **kwargs: kwargs,
            )
            hub = types.SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot))
            fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            psutil = types.SimpleNamespace(Process=lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=1)))
            seen_mnt = []

            def fake_score(mdl, tok, prefix, cand, t):
                return {'sum_log_probability': -1.0 if cand == 'cook' else -2.0}

            def fake_generate(mdl, tok, text, cfg, t):
                seen_mnt.append(cfg['generation']['max_new_tokens'])
                return {
                    'rendered_prompt': text,
                    'effective_generation_config': {'max_new_tokens': cfg['generation']['max_new_tokens']},
                    'input_token_ids': [1, 2],
                    'generated_token_ids': [10, 11],
                    'generated_text': 'A. The cook',
                }

            modules = {
                'torch': torch, 'bitsandbytes': bnb, 'transformers': transformers,
                'huggingface_hub': hub, 'fcntl': fcntl, 'psutil': psutil,
            }
            out_root = root / 'results'
            argv = [
                'quantization/run.py', '--manifest', str(m1),
                '--manifests', str(m1), str(m2),
                '--max-new-tokens-cap', '24', '--skip-if-completed',
                '--output-root', str(out_root),
            ]
            with mock.patch.object(qrun, 'ROOT', root), mock.patch.dict(sys.modules, modules), \
                 mock.patch.dict(os.environ, {'TEST_IMAGE_ID': 'image-1', 'BAP_RUNTIME_CONFIG': str(root / 'none.json')}), \
                 mock.patch.object(qrun.importlib.metadata, 'version', return_value='test'), \
                 mock.patch.object(qrun, 'score_candidate', side_effect=fake_score) as scores, \
                 mock.patch.object(qrun, 'generate', side_effect=fake_generate) as gens, \
                 mock.patch.object(qrun, 'metrics', return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(sys, 'argv', argv):
                    qrun.main()

                self.assertEqual(from_pretrained.call_count, 1)
                self.assertTrue(all(mnt == 24 for mnt in seen_mnt))
                dirs = sorted(out_root.glob('*-inference'))
                self.assertEqual(len(dirs), 2)
                st1 = json.loads((dirs[0] / 'run.json').read_text(encoding='utf-8'))
                st2 = json.loads((dirs[1] / 'run.json').read_text(encoding='utf-8'))
                cfg1 = json.loads((dirs[0] / 'config.json').read_text(encoding='utf-8'))
                self.assertEqual(cfg1['generation']['max_new_tokens'], 48)
                self.assertEqual(st1['max_new_tokens_cap'], 24)
                self.assertFalse(st1['model_reused_in_memory'])
                self.assertTrue(st2['model_reused_in_memory'])
                self.assertGreaterEqual(st2['cross_seed_cache_hits'], 2)
                self.assertEqual(scores.call_count, 4)

                with mock.patch.object(sys, 'argv', argv):
                    qrun.main()
                self.assertEqual(from_pretrained.call_count, 1)


if __name__ == '__main__':
    unittest.main()
