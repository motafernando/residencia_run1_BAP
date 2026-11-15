import unittest

from runtime_model import describe_model_device_map


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


if __name__ == '__main__':
    unittest.main()
