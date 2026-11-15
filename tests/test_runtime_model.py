from pathlib import Path
import sys
import tempfile
import unittest

from runtime_model import (
    describe_model_device_map,
    ensure_chat_template,
    prepare_generation_config,
)

QUANT_DIR = str(Path(__file__).resolve().parent.parent / 'quantization')
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quantization_config import summarize_quantized_model


class FakeWeight:
    device = 'cuda:0'


class FakeEmbeddings:
    weight = FakeWeight()


class ModelWithAccelerateMap:
    hf_device_map = {'model.layers.0': 0, 'lm_head': 'cuda:0'}


class ModelWithDevice:
    device = 'cuda:0'


class ModelWithEmbeddingsOnly:
    def get_input_embeddings(self):
        return FakeEmbeddings()


class TestDescribeModelDeviceMap(unittest.TestCase):
    def test_uses_accelerate_device_map_when_available(self):
        self.assertEqual(
            describe_model_device_map(ModelWithAccelerateMap()),
            {'model.layers.0': '0', 'lm_head': 'cuda:0'},
        )

    def test_falls_back_to_model_device(self):
        self.assertEqual(describe_model_device_map(ModelWithDevice()), {'': 'cuda:0'})

    def test_falls_back_to_input_embedding_device(self):
        self.assertEqual(describe_model_device_map(ModelWithEmbeddingsOnly()), {'': 'cuda:0'})

    def test_ensure_chat_template_reads_jinja_fallback(self):
        class FakeTokenizer:
            chat_template = None

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'chat_template.jinja').write_text('{{ messages[0].content }}', encoding='utf-8')
            tok = FakeTokenizer()
            loaded = ensure_chat_template(tok, tmp)
            self.assertEqual(loaded, '{{ messages[0].content }}')
            self.assertEqual(tok.chat_template, '{{ messages[0].content }}')

    def test_prepare_generation_config_clears_sampling_flags_when_greedy(self):
        class FakeGenConfig:
            do_sample = True
            temperature = 0.6
            top_p = 0.95
            top_k = 20
            eos_token_id = None
            pad_token_id = None
            max_new_tokens = 256

            def update(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class FakeTokenizer:
            pad_token_id = None
            eos_token_id = 2

        gen = prepare_generation_config(
            FakeGenConfig(),
            {'do_sample': False, 'max_new_tokens': 48, 'num_beams': 1},
            FakeTokenizer(),
        )
        self.assertFalse(gen.do_sample)
        self.assertEqual(gen.max_new_tokens, 48)
        self.assertEqual(gen.pad_token_id, 2)
        self.assertEqual(gen.eos_token_id, 2)
        self.assertIsNone(gen.temperature)
        self.assertIsNone(gen.top_p)
        self.assertIsNone(gen.top_k)

    def test_summarize_quantized_model_handles_missing_hf_device_map(self):
        class FakeLinear8bit:
            pass

        class FakeBnb:
            class nn:
                Linear8bitLt = FakeLinear8bit
                Linear4bit = type('FakeLinear4bit', (), {})

        class FakeQuantCfg:
            def to_dict(self):
                return {'load_in_8bit': True}

        class FakeCfg:
            quantization_config = FakeQuantCfg()

        class FakeQuantModel:
            hf_device_map = None
            device = 'cuda:0'
            is_loaded_in_8bit = True
            config = FakeCfg()

            def named_modules(self):
                return [('layer0', FakeLinear8bit())]

        summary = summarize_quantized_model(
            FakeQuantModel(),
            FakeBnb(),
            {'method': 'bitsandbytes', 'load_in_8bit': True},
        )
        self.assertEqual(summary['mode'], 'int8')
        self.assertEqual(summary['module_count'], 1)


if __name__ == '__main__':
    unittest.main()
