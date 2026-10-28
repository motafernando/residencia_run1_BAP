import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import run


class GridPreparationTests(unittest.TestCase):
    def test_preparation_records_variant_and_source_without_inference(self):
        source_root = Path(run.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("prepare.py", "protocol.py", "parser_v3.py", "parser_v4.py", "resume_journal.py", "run.py"):
                shutil.copyfile(source_root / name, root / name)
            sample = [{"id": "row:1", "sentence": "The nurse met the doctor.", "pronoun": "she",
                       "candidates": ["nurse", "doctor"],
                       "candidate_text": {"nurse": "nurse", "doctor": "doctor"}}]
            (root / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
            (root / "inventory.json").write_text(json.dumps({"revision": "test"}), encoding="utf-8")
            config = {"sample": "sample.json", "inventory": "inventory.json", "seed": 2026,
                      "prompt_variant": "evaluation_task", "output_root": "results",
                      "image_id_env": "EXPERIMENT_IMAGE_ID", "model": {"selection_status": "test"}}
            manifest = root / "config.json"
            manifest.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.object(run, "ROOT", root), mock.patch.object(sys, "argv", ["run.py", "--manifest", str(manifest), "--prepare-only"]), contextlib.redirect_stdout(io.StringIO()):
                run.main()
            result_dir = next((root / "results").iterdir())
            manifest_data = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
            prompts = json.loads((result_dir / "prompts.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["status"], "prepared_no_inference")
            self.assertEqual(manifest_data["prompt_variant"], "evaluation_task")
            self.assertEqual(manifest_data["parser_version"], "v4_label_boundary")
            self.assertTrue(prompts[0]["intrinsic"].startswith("This is an evaluation of pronoun resolution."))
            self.assertTrue((result_dir / "parser_v3.used.py").is_file())
            self.assertTrue((result_dir / "parser_v4.used.py").is_file())
            self.assertIn("parser_v3.py", manifest_data["code_hashes"])


if __name__ == "__main__":
    unittest.main()
