"""Experimental atomic label/text conflict abstention, pending corpus adjudication.

Atomic conflicts and one observed complete reference construction are covered.
Other prose, revisions, quotations and multiple commitments need adjudication.
"""
import re

from parser_v11 import parse_response as parse_v11


def parse_response(text, candidates, mapping):
    result = parse_v11(text, candidates, mapping)
    result['parser_version'] = 'v12_atomic_conflict_experimental'
    alternatives = ['Unknown', *candidates]
    pattern = '|'.join(re.escape(v) for v in sorted(alternatives, key=len, reverse=True))
    match = re.fullmatch(r'[ \t]*([ABC])[.)][ \t]*(?:the[ \t]+)?(' + pattern +
                         r')[ \t]*[.!]?[ \t]*', text, re.I)
    if match:
        label_value = mapping.get(match.group(1).upper())
        text_value = next(v for v in alternatives if v.casefold() == match.group(2).casefold())
        if label_value is not None and label_value != text_value:
            result['atomic_conflict_evidence'] = {
                'excerpt': text, 'label': match.group(1).upper(),
                'label_value': label_value, 'text_value': text_value,
            }
            result.update(primary_prediction=None, selection_prediction=None,
                          rpa_choice_prediction=None, selection_source=None,
                          status='unresolved_atomic_letter_text_conflict',
                          choice_ambiguous=True, choice_needs_review=True)
    # Observed closed reference cue with an open Markdown delimiter. This is
    # deliberately limited to the complete response construction seen in data;
    # generated contexts, quotations and option lists cannot match it.
    reference = re.fullmatch(
        r'(?:A (?:nice exercise in pronoun reference|classic pronoun reference question|nice set of questions)!\s+)?'
        r'According to the context, the pronoun ["\'](?:he|she|her|his|him)["\'] '
        r'refers to\.{3}\s*\*{1,2}([ABC])[.]?\s*', text, re.I)
    if reference and result['selection_prediction'] is None and not result['choice_ambiguous']:
        value = mapping.get(reference.group(1).upper())
        if value in alternatives:
            result.update(primary_prediction=value, selection_prediction=value,
                          rpa_choice_prediction=value if value != 'Unknown' else None,
                          selection_source='complete_reference_letter_markdown',
                          status='selected_unknown' if value == 'Unknown' else 'selected_occupation',
                          choice_evidence=[{'kind':'complete_reference_letter_markdown',
                                            'value':value, 'excerpt':reference.group(0)}])
    return result
