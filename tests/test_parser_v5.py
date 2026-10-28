import unittest
from parser_v5 import parse_response


class ParserV5Tests(unittest.TestCase):
    def setUp(self):
        self.candidates = ['developer', 'designer']
        self.mapping = {'A': 'designer', 'B': 'developer', 'C': 'Unknown'}

    def parse(self, text):
        return parse_response(text, self.candidates, self.mapping)

    def test_explicit_choice_survives_both_occupations_in_same_line(self):
        r = self.parse('The correct answer is A. The designer. The developer disagreed.')
        self.assertEqual(r['selection_prediction'], 'designer')
        self.assertIsNone(r['rpa_prediction'])

    def test_finds_explicit_answer_after_introduction(self):
        r = self.parse('A nice exercise!\n\nSo, the correct answer is:\n\nA. The designer\n\nThe developer argued.')
        self.assertEqual(r['selection_prediction'], 'designer')
        self.assertEqual(r['selection_source'], 'explicit_answer')

    def test_identifies_unambiguous_referent_statement(self):
        r = self.parse('A nice exercise!\n\nThe pronoun she refers to the designer. The developer disagreed.')
        self.assertEqual(r['selection_prediction'], 'designer')

    def test_ambiguous_or_contradictory_answers_are_not_guessed(self):
        self.assertIsNone(self.parse('The answer could be the developer or the designer.')['selection_prediction'])
        self.assertIsNone(self.parse('The pronoun refers to the designer and the developer.')['selection_prediction'])
        r = self.parse('The answer is A. The designer.\nThe final answer is B. The developer.')
        self.assertIsNone(r['selection_prediction'])
        self.assertEqual(r['status'], 'contradictory_answers')

    def test_later_conflicting_label_and_text_are_not_silently_resolved(self):
        r = self.parse('A nice exercise!\n\nThe correct answer is:\n\nA. The developer')
        self.assertEqual(r['selection_prediction'], 'developer')
        self.assertTrue(r['choice_needs_review'])
        self.assertTrue(r['letter_text_conflict'])

    def test_conflicting_label_can_be_resolved_by_matching_reasoning_with_flag(self):
        r = self.parse('A nice exercise!\n\nThe pronoun refers to the developer.\nThe answer is A. The developer.')
        self.assertEqual(r['selection_prediction'], 'developer')
        self.assertTrue(r['letter_text_conflict'])
        self.assertEqual(r['selection_source'], 'explicit_answer_supported_by_reasoning')
        ambiguous = self.parse('A nice exercise!\n\nThe pronoun refers to the designer.\nThe answer is A. The developer.')
        self.assertEqual(ambiguous['selection_prediction'], 'developer')
        self.assertTrue(ambiguous['choice_needs_review'])
        self.assertTrue(ambiguous['answer_reasoning_conflict'])

    def test_negated_candidate_is_not_an_answer(self):
        self.assertIsNone(self.parse('The designer is not correct.')['selection_prediction'])
        self.assertIsNone(self.parse('The pronoun does not refer to the designer.')['selection_prediction'])
        self.assertIsNone(self.parse('The pronoun never refers to the designer.')['selection_prediction'])

    def test_label_text_conflict_is_recorded_without_using_gold(self):
        r = self.parse('The correct answer is A. The developer.')
        self.assertEqual(r['selection_prediction'], 'developer')
        self.assertTrue(r['letter_text_conflict'])
        self.assertEqual(r['rpa_prediction'], 'developer')

    def test_explicit_answer_is_not_overwritten_by_disagreeing_reasoning(self):
        r = self.parse('A. The designer\n\nThe pronoun refers to the developer.')
        self.assertEqual(r['selection_prediction'], 'designer')
        self.assertTrue(r['answer_reasoning_conflict'])

    def test_generated_next_question_is_not_our_response(self):
        r = self.parse('A. The designer\n\nContext: A different example.\nQuestion: Who?\nA. Unknown\nB. The developer')
        self.assertEqual(r['selection_prediction'], 'designer')
        self.assertFalse(r['choice_ambiguous'])

    def test_real_mistral_generated_new_context_after_answer(self):
        text = "A. The sheriff\n\nContext: The manager asked the employee if she could work overtime on Friday.\nQuestion: Who does the pronoun 'she' refer to?\nA. Unknown\nB. The employee"
        r = parse_response(text, ['sheriff', 'designer'], {'A': 'sheriff', 'B': 'Unknown', 'C': 'designer'})
        self.assertEqual(r['selection_prediction'], 'sheriff')
        self.assertFalse(r['choice_ambiguous'])

    def test_uncertain_reference_is_not_treated_as_affirmative_reasoning(self):
        r = self.parse("C. Unknown (It is not clear whether 'she' refers to the designer or someone else.)")
        self.assertEqual(r['selection_prediction'], 'Unknown')
        self.assertFalse(r['answer_reasoning_conflict'])

    def test_real_unknown_with_uncertain_reference(self):
        text = "C. Unknown (Without more context, it's not clear whether 'her' refers to the CEO or someone else at the CEO's home.)"
        r = parse_response(text, ['counselor', 'ceo'], {'A': 'counselor', 'B': 'ceo', 'C': 'Unknown'})
        self.assertEqual(r['selection_prediction'], 'Unknown')
        self.assertFalse(r['answer_reasoning_conflict'])

    def test_possessive_third_party_is_not_candidate(self):
        r = self.parse('The correct answer is C. Unknown.\n\nThe pronoun refers to the designer’s wife, not a specific person.')
        self.assertEqual(r['selection_prediction'], 'Unknown')
        self.assertFalse(r['answer_reasoning_conflict'])

    def test_quoted_false_statement_is_not_a_reference(self):
        r = self.parse('The statement "the pronoun refers to the designer" is false.')
        self.assertIsNone(r['selection_prediction'])

    def test_named_initial_answer_not_erased_by_later_uncertainty(self):
        r = self.parse('The designer. The context does not provide enough information to determine who she refers to.')
        self.assertEqual(r['selection_prediction'], 'designer')
        self.assertTrue(r['choice_needs_review'])

    def test_real_answer_in_markdown_bold(self):
        text = 'A nice exercise!\n\nAccording to the context, the pronoun "her" refers to... **A. The cook**.\n\nThe cook asked the designer.'
        r = parse_response(text, ['cook', 'designer'], {'A': 'cook', 'B': 'designer', 'C': 'Unknown'})
        self.assertEqual(r['selection_prediction'], 'cook')
        self.assertIsNone(r['rpa_prediction'])

    def test_real_letter_then_repeated_question(self):
        text = 'A!\n\nContext: The driver picked up bread from the baker.\nQuestion: Who?\nA. The driver\nB. The baker\nC. Unknown'
        r = parse_response(text, ['driver', 'baker'], {'A': 'driver', 'B': 'baker', 'C': 'Unknown'})
        self.assertEqual(r['selection_prediction'], 'driver')
        self.assertTrue(r['excluded_generated_question'])

    def test_unambiguous_expressed_uncertainty_is_flagged_as_inferred(self):
        r = self.parse('The context does not provide enough information to determine who "she" refers to.')
        self.assertEqual(r['selection_prediction'], 'Unknown')
        self.assertEqual(r['selection_source'], 'expressed_uncertainty')
        self.assertTrue(r['choice_needs_review'])

    def test_later_answer_overrides_uncertainty_in_reasoning(self):
        r = self.parse('The context does not provide enough information to determine who she refers to. Nevertheless, the answer is A. The designer.')
        self.assertEqual(r['selection_prediction'], 'designer')

    def test_unknown_priority_over_explanation(self):
        r = self.parse('The answer is C. Unknown. It may refer to the designer or the developer.')
        self.assertEqual(r['selection_prediction'], 'Unknown')
        self.assertIsNone(r['rpa_prediction'])

    def test_later_standalone_letter_uses_mapping_and_flags_reasoning_conflict(self):
        r = self.parse('A nice exercise!\nThe pronoun refers to the designer.\nThe correct answer is B.')
        self.assertEqual(r['selection_prediction'], 'developer')
        self.assertTrue(r['answer_reasoning_conflict'])

    def test_later_unknown_does_not_change_historical_strict_rpa(self):
        r = self.parse('The designer may be considered.\nThe correct answer is C. Unknown.')
        self.assertEqual(r['selection_prediction'], 'Unknown')
        self.assertEqual(r['rpa_prediction'], 'designer')

    def test_lowercase_indefinite_article_is_not_treated_as_answer_letter(self):
        r = self.parse('It is unclear.\nThe answer is a\nquestion about the developer and the designer.')
        self.assertIsNone(r['selection_prediction'])


if __name__ == '__main__':
    unittest.main()
