import unittest

from parser_v3 import parse_response


class ParseResponseTests(unittest.TestCase):
    def setUp(self):
        self.candidates = ["doctor", "nurse"]
        self.mapping = {"A": "doctor", "B": "nurse", "C": "Unknown"}

    def parse(self, text):
        return parse_response(text, self.candidates, self.mapping)

    def test_bare_letter_resolves_selection_but_not_strict_rpa(self):
        result = self.parse("B")
        self.assertEqual(result["selection_prediction"], "nurse")
        self.assertEqual(result["selection_source"], "letter_only")
        self.assertIsNone(result["rpa_prediction"])
        self.assertEqual(result["rpa_choice_prediction"], "nurse")

    def test_unknown_letter_resolves_as_unknown(self):
        result = self.parse("Answer: C.")
        self.assertEqual(result["selection_prediction"], "Unknown")
        self.assertEqual(result["status"], "selected_unknown")
        self.assertIsNone(result["rpa_prediction"])
        self.assertIsNone(result["rpa_choice_prediction"])

    def test_named_answer_retains_strict_rpa(self):
        result = self.parse("B. The nurse")
        self.assertEqual(result["selection_prediction"], "nurse")
        self.assertEqual(result["rpa_prediction"], "nurse")
        self.assertFalse(result["letter_text_conflict"])

    def test_contradictory_letter_is_flagged_and_text_retained(self):
        result = self.parse("A. The nurse")
        self.assertEqual(result["selection_prediction"], "nurse")
        self.assertTrue(result["letter_text_conflict"])

    def test_multiline_letter_is_not_guessed(self):
        result = self.parse("B\nBoth are possible")
        self.assertIsNone(result["selection_prediction"])

    def test_letter_not_in_mapping_is_not_guessed(self):
        result = parse_response("B", self.candidates, {"A": "doctor"})
        self.assertIsNone(result["selection_prediction"])

    def test_unknown_with_explanation_stays_unknown(self):
        result = self.parse("Unknown\nThe doctor or the nurse could be referenced.")
        self.assertEqual(result["selection_prediction"], "Unknown")
        self.assertIsNone(result["rpa_prediction"])


if __name__ == "__main__":
    unittest.main()
