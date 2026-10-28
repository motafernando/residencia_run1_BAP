import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import run


class RunnerResumeTests(unittest.TestCase):
    def test_interrupted_inference_resumes_without_recomputing_success(self):
        source = Path(run.__file__).parent
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('prepare.py', 'protocol.py', 'parser_v3.py', 'parser_v4.py', 'resume_journal.py', 'run.py'):
                shutil.copyfile(source / name, root / name)
            record = {'id': 'one', 'sentence': 'The nurse met the doctor.', 'pronoun': 'she',
                      'candidates': ['nurse', 'doctor'], 'candidate_text': {'nurse': 'nurse', 'doctor': 'doctor'}}
            (root / 'sample.json').write_text(json.dumps([record]), encoding='utf8')
            (root / 'inventory.json').write_text(json.dumps({'revision': 'test', 'occupations': []}), encoding='utf8')
            config = {'sample': 'sample.json', 'inventory': 'inventory.json', 'seed': 2026,
                      'prompt_variant': 'original', 'output_root': 'results', 'image_id_env': 'TEST_IMAGE_ID',
                      'model': {'repo_id': 'test', 'allowed_repos': ['test'], 'revision': 'a' * 40,
                                'tokenizer_revision': 'a' * 40, 'selection_status': 'test', 'dtype': 'bfloat16',
                                'device_map': 'auto', 'max_memory': {'0': '1GiB'}, 'local_files_only': True,
                                'quantization': None}, 'format': {'prompting': 'raw_completion'}, 'generation': {}}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(config), encoding='utf8')
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'config.json').write_text('{}', encoding='utf8')
            model = types.SimpleNamespace(eval=lambda: None, hf_device_map={'': 0})
            tokenizer = types.SimpleNamespace(chat_template='template')
            cuda = types.SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda _: None,
                get_device_name=lambda _: 'fake', get_device_capability=lambda _: (1, 0), get_arch_list=lambda: [],
                mem_get_info=lambda: (1, 2), max_memory_allocated=lambda: 1, max_memory_reserved=lambda: 1)
            torch = types.SimpleNamespace(cuda=cuda, manual_seed=lambda _: None, version=types.SimpleNamespace(cuda='fake'), bfloat16='fake')
            transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
                AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: model))
            hub = types.SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot))
            fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            psutil = types.SimpleNamespace(Process=lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=1)))
            score = lambda model, tokenizer, prefix, candidate, torch: {'sum_log_probability': 1 if candidate == 'nurse' else 0}
            response = {'generated_text': 'nurse'}
            argv = ['run.py', '--manifest', str(manifest)]
            modules = {'torch': torch, 'transformers': transformers, 'huggingface_hub': hub, 'fcntl': fcntl, 'psutil': psutil}
            with mock.patch.object(run, 'ROOT', root), mock.patch.dict(sys.modules, modules), \
                 mock.patch.dict(os.environ, {'TEST_IMAGE_ID': 'image-1'}), \
                 mock.patch.object(run.importlib.metadata, 'version', return_value='test'), \
                 mock.patch.object(run, 'score_candidate', side_effect=score) as scores, \
                 mock.patch.object(run, 'generate', side_effect=[KeyboardInterrupt(), response, response]), \
                 mock.patch.object(run, 'parse_response', return_value={'selection_prediction': 'nurse', 'rpa_prediction': 'nurse', 'rpa_choice_prediction': 'nurse'}), \
                 mock.patch.object(run, 'metrics', return_value={}), contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(sys, 'argv', argv), self.assertRaises(KeyboardInterrupt):
                    run.main()
                out = next((root / 'results').iterdir())
                self.assertEqual(len((out / 'responses.jsonl').read_text().splitlines()), 1)
                with mock.patch.object(sys, 'argv', argv + ['--resume-from', str(out)]):
                    run.main()
                self.assertEqual(scores.call_count, 2)
                self.assertEqual(len((out / 'responses.jsonl').read_text().splitlines()), 3)
                self.assertEqual(json.loads((out / 'run.json').read_text())['execution_errors'], 0)


if __name__ == '__main__':
    unittest.main()
