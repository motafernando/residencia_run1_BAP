import unittest
from parser_v4 import parse_response


class ParserV4Tests(unittest.TestCase):
    def test_candidate_starting_with_option_letter_is_not_truncated(self):
        for occupation in ('carpenter', 'accountant', 'baker'):
            with self.subTest(occupation=occupation):
                parsed = parse_response(occupation, [occupation, 'nurse'], {'A': occupation, 'B': 'nurse', 'C': 'Unknown'})
                self.assertEqual(parsed['selection_prediction'], occupation)
                self.assertEqual(parsed['rpa_prediction'], occupation)

    def test_explicit_labels_bare_letters_and_unknown(self):
        candidates = ['baker', 'nurse']
        mapping = {'A': 'baker', 'B': 'nurse', 'C': 'Unknown'}
        for text in ('A. baker', '(A) baker', 'Answer: A. baker', 'baker'):
            self.assertEqual(parse_response(text, candidates, mapping)['selection_prediction'], 'baker')
        self.assertEqual(parse_response('B', candidates, mapping)['selection_prediction'], 'nurse')
        self.assertEqual(parse_response('Answer: Unknown', candidates, mapping)['selection_prediction'], 'Unknown')

    def test_conflicting_label_is_flagged_and_text_retained(self):
        result = parse_response('A. nurse', ['baker', 'nurse'], {'A': 'baker', 'B': 'nurse', 'C': 'Unknown'})
        self.assertEqual(result['selection_prediction'], 'nurse')
        self.assertTrue(result['letter_text_conflict'])

    def test_unknown_after_answer_introduction_is_recognized(self):
        candidates = ['lawyer', 'nurse']
        mapping = {'A': 'Unknown', 'B': 'nurse', 'C': 'lawyer'}
        for text in ('The correct answer is A. Unknown.', 'The correct answer is Unknown.'):
            parsed = parse_response(text, candidates, mapping)
            self.assertEqual(parsed['selection_prediction'], 'Unknown')
            self.assertIsNone(parsed['rpa_prediction'])
        conflict = parse_response('The answer is B. Unknown.', candidates, mapping)
        self.assertTrue(conflict['letter_text_conflict'])
        prose = parse_response('The answer is unknown whether the nurse is right.', candidates, mapping)
        self.assertNotEqual(prose['selection_prediction'], 'Unknown')

    def test_embedded_answer_label_is_checked_for_conflict(self):
        result = parse_response('The correct answer is A. The nurse.', ['lawyer', 'nurse'],
                                {'A': 'lawyer', 'B': 'nurse', 'C': 'Unknown'})
        self.assertEqual(result['selection_prediction'], 'nurse')
        self.assertTrue(result['letter_text_conflict'])
        concordant = parse_response('The correct answer is B. The nurse.', ['lawyer', 'nurse'],
                                    {'A': 'lawyer', 'B': 'nurse', 'C': 'Unknown'})
        self.assertFalse(concordant['letter_text_conflict'])
        article = parse_response('The answer is a nurse.', ['lawyer', 'nurse'],
                                 {'A': 'lawyer', 'B': 'nurse', 'C': 'Unknown'})
        self.assertFalse(article['letter_text_conflict'])

    def test_strict_metric_still_counts_full_text_mentions(self):
        result = parse_response('nurse\nThe baker was not chosen.', ['baker', 'nurse'], {'A': 'baker', 'B': 'nurse', 'C': 'Unknown'})
        self.assertEqual(result['selection_prediction'], 'nurse')
        self.assertIsNone(result['rpa_prediction'])


if __name__ == '__main__':
    unittest.main()
