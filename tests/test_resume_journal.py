import json
import tempfile
import unittest
from pathlib import Path

from resume_journal import recover_journal, append_row, validate_resume


class JournalRecoveryTests(unittest.TestCase):
    def test_resume_refuses_changed_manifest_code_inputs_or_image(self):
        state = {'status': 'running', 'code_hashes': {'run.py': 'a'},
                 'input_hashes': {'sample': 'b'}, 'environment': {'image_id': 'sha256:c'}}
        cfg = {'seed': 2026}
        validate_resume(cfg, cfg, state, {'run.py': 'a'}, {'sample': 'b'}, 'sha256:c')
        for args in (({'seed': 4}, cfg, state, {'run.py': 'a'}, {'sample': 'b'}, 'sha256:c'),
                     (cfg, cfg, state, {'run.py': 'x'}, {'sample': 'b'}, 'sha256:c'),
                     (cfg, cfg, state, {'run.py': 'a'}, {'sample': 'x'}, 'sha256:c'),
                     (cfg, cfg, state, {'run.py': 'a'}, {'sample': 'b'}, 'sha256:x')):
            with self.assertRaises(ValueError):
                validate_resume(*args)

    def test_resume_preserves_complete_rows_and_partial_fragment(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'responses.jsonl'
            row = {'id': 'a', 'condition': 'intrinsic', 'status': 'ok'}
            path.write_bytes((json.dumps(row) + '\n{"id":').encode())
            result = recover_journal(path, {('a', 'intrinsic'), ('a', 'zero_shot')})
            self.assertEqual(result, {('a', 'intrinsic'): row})
            self.assertEqual(path.read_text(), json.dumps(row) + '\n')
            self.assertEqual((Path(temp) / 'responses-incomplete-1.bin').read_bytes(), b'{"id":')
            append_row(path, {'id': 'a', 'condition': 'zero_shot', 'status': 'ok'})
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_refuses_invalid_interior_and_unknown_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'responses.jsonl'
            path.write_bytes(b'bad\n{"id":"a","condition":"intrinsic","status":"ok"}\n')
            with self.assertRaises(ValueError):
                recover_journal(path, {('a', 'intrinsic')})
            path.write_text('{"id":"other","condition":"intrinsic","status":"ok"}\n')
            with self.assertRaises(ValueError):
                recover_journal(path, {('a', 'intrinsic')})

    def test_retries_error_without_erasing_history_and_rejects_duplicate_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'responses.jsonl'
            error = {'id': 'a', 'condition': 'intrinsic', 'status': 'error'}
            ok = dict(error, status='ok')
            append_row(path, error)
            self.assertEqual(recover_journal(path, {('a', 'intrinsic')})[('a', 'intrinsic')], error)
            append_row(path, ok)
            self.assertEqual(recover_journal(path, {('a', 'intrinsic')})[('a', 'intrinsic')], ok)
            append_row(path, ok)
            with self.assertRaises(ValueError):
                recover_journal(path, {('a', 'intrinsic')})


if __name__ == '__main__':
    unittest.main()
