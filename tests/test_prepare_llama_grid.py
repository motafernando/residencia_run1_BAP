import copy
import unittest

from prepare_llama_grid import build_configs


class LlamaGridConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "purpose": "Mistral", "seed": 2026, "sample": "data/prepared/all_records.json",
            "model": {"repo_id": "mistralai/Mistral-7B-Instruct-v0.3", "revision": "old",
                      "tokenizer_revision": "old", "dtype": "bfloat16", "quantization": None,
                      "max_memory": {"0": "13GiB", "cpu": "10GiB"}},
            "format": {"intrinsic": "raw_completion", "prompting": "tokenizer_chat_template"},
            "generation": {"do_sample": False, "max_new_tokens": 48, "num_beams": 1},
            "checkpoint_provenance": "Mistral", "output_root": "results",
        }
        self.sha = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"

    def test_configurations_keep_data_and_generation_and_fix_llama_revision(self):
        before = copy.deepcopy(self.base)
        configs = build_configs(self.base, self.sha)
        self.assertEqual(self.base, before)
        self.assertEqual(set(configs), {"original", "refined", "evaluation_task", "evaluation_gender_bias", "pilot_original"})
        for name, config in configs.items():
            self.assertEqual(config["model"]["repo_id"], "meta-llama/Meta-Llama-3-8B-Instruct")
            self.assertEqual(config["model"]["revision"], self.sha)
            self.assertEqual(config["model"]["tokenizer_revision"], self.sha)
            self.assertEqual(config["generation"], self.base["generation"])
            self.assertEqual(config["seed"], 2026)
            self.assertEqual(config["model"]["dtype"], "bfloat16")
            self.assertIsNone(config["model"]["quantization"])
            self.assertEqual(config["sample"], "data/prepared/smoke24.json" if name == "pilot_original" else self.base["sample"])
            self.assertEqual(config["prompt_variant"], "original" if name == "pilot_original" else name)

    def test_revision_must_be_complete_commit(self):
        with self.assertRaises(ValueError):
            build_configs(self.base, "main")


if __name__ == "__main__":
    unittest.main()
