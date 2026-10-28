import unittest
from auditar_comparacao_mistral import transitions


class PairedTransitionsTests(unittest.TestCase):
    def test_transitions_use_type2_gold_and_separate_counts(self):
        previous = [{'id': 'a', 'type': 2, 'gold': 'x', 'rpa_prediction': 'x'},
                    {'id': 'b', 'type': 2, 'gold': 'x', 'rpa_prediction': 'y'},
                    {'id': 'c', 'type': 2, 'gold': 'x', 'rpa_prediction': 'x'},
                    {'id': 'd', 'type': 1, 'gold': 'x', 'rpa_prediction': 'x'}]
        current = {'a': {'rpa_prediction': None}, 'b': {'rpa_prediction': 'x'},
                   'c': {'rpa_prediction': 'x'}, 'd': {'rpa_prediction': None}}
        self.assertEqual(transitions(previous, current), {'n': 3, 'correct_to_not': 1,
                         'not_to_correct': 1, 'both_correct': 1, 'both_not': 0})


if __name__ == '__main__':
    unittest.main()
