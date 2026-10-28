import unittest
import tempfile
from pathlib import Path

from run_llama_after_grid import grid_gate, required_files, find_resumable_run, reusable_gpu_pilot, valid_stage_result, image_reference


class LlamaQueueTests(unittest.TestCase):
    def test_started_run_keeps_exact_historical_image(self):
        state = {'image_id': 'sha256:old', 'image_tag': 'bias-winobias-v4:0.1'}
        self.assertEqual(image_reference(state, True), 'sha256:old')
        self.assertEqual(image_reference(state, False), 'bias-winobias-v4:0.1')

    def test_stage_result_requires_selected_variant_and_expected_count(self):
        sample = {'status': 'inference_finished_needs_manual_review', 'records_written': 72,
                  'execution_errors': 0, 'prompt_variant': 'refined', 'model_device_map': {'': '0'}}
        self.assertTrue(valid_stage_result(sample, 'pilot', 'refined', require_full_gpu=True))
        self.assertFalse(valid_stage_result(sample, 'full', 'refined', require_full_gpu=True))
        self.assertFalse(valid_stage_result(sample, 'pilot', 'original', require_full_gpu=True))
        self.assertFalse(valid_stage_result(dict(sample, model_device_map={'layer': 'cpu'}), 'pilot', 'refined', require_full_gpu=True))

    def test_reuse_requires_same_experiment_and_completed_full_gpu_pilot(self):
        pilot = {'sample': 'pilot', 'output_root': 'pilot-output', 'purpose': 'pilot',
                 'model': {'revision': 'abc', 'device_map': {'': 0}}, 'seed': 2026}
        full = dict(pilot, sample='full', output_root='full-output', purpose='full')
        state = {'status': 'inference_finished_needs_manual_review', 'sample_size': 24,
                 'records_written': 72, 'execution_errors': 0, 'model_device_map': {'': '0'},
                 'environment': {'image_id': 'image'}}
        self.assertTrue(reusable_gpu_pilot(pilot, full, state, 'image'))
        self.assertFalse(reusable_gpu_pilot(pilot, dict(full, seed=2027), state, 'image'))
        self.assertFalse(reusable_gpu_pilot(pilot, full, dict(state, model_device_map={'layer': 'cpu'}), 'image'))
        self.assertFalse(reusable_gpu_pilot(pilot, full, state, 'different-image'))

    def test_only_one_existing_initialized_run_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            incomplete = p / 'one-inference'
            incomplete.mkdir()
            self.assertIsNone(find_resumable_run(p))
            (incomplete / 'run.json').write_text('{}')
            self.assertEqual(find_resumable_run(p), incomplete)
            other = p / 'two-inference'
            other.mkdir()
            (other / 'run.json').write_text('{}')
            with self.assertRaises(ValueError):
                find_resumable_run(p)

    def test_waits_until_grid_completes(self):
        self.assertEqual(grid_gate({"status": "running", "completed": []}), "wait")

    def test_refuses_failed_or_incomplete_grid(self):
        self.assertEqual(grid_gate({"status": "failed", "completed": []}), "blocked")
        self.assertEqual(grid_gate({"status": "completed_needs_review", "completed": []}), "blocked")

    def test_only_accepts_three_successful_complete_variants(self):
        rows = [{"stage": "full", "variant": v, "records_written": 9504, "execution_errors": 0}
                for v in ("refined", "evaluation_task", "evaluation_gender_bias")]
        self.assertEqual(grid_gate({"status": "completed_needs_review", "completed": rows}), "ready")
        rows[2]["execution_errors"] = 1
        self.assertEqual(grid_gate({"status": "completed_needs_review", "completed": rows}), "blocked")

    def test_file_selection_excludes_original_weight_copies(self):
        files = {"config.json": {}, "generation_config.json": {}, "model.safetensors.index.json": {},
                 "tokenizer.json": {}, "tokenizer_config.json": {}, "special_tokens_map.json": {},
                 **{f"model-{i:05d}-of-00004.safetensors": {} for i in range(1, 5)},
                 "original/consolidated.00.pth": {}, "original/tokenizer.model": {}}
        selected = required_files(files)
        self.assertEqual(len(selected), 10)
        self.assertFalse(any(name.startswith("original/") for name in selected))
        del files["model-00004-of-00004.safetensors"]
        with self.assertRaises(ValueError):
            required_files(files)


if __name__ == "__main__":
    unittest.main()
