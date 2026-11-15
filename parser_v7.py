"""Experimental parser: distinguish echoed question options from selections.

The strict RPA prediction always comes from the unchanged v4 parser on the
original text. No answer key is used.
"""
import re

from parser_v4 import parse_response as parse_v4
from parser_v6 import parse_response as parse_v6


def parse_response(text, candidates, mapping):
    lines = text.splitlines(keepends=True)
    in_question = False
    masked = False
    filtered = []
    for line in lines:
        if re.match(r'^\s*Question\s*:', line, re.I):
            in_question = True
        elif in_question and re.match(r'^\s*(?:the\s+correct\s+)?(?:answer|choice|option)\s*(?:is|:)', line, re.I):
            in_question = False
        if in_question and re.match(r'^\s*\(?[ABC]\)?[.)](?:\s|$)', line):
            filtered.append(' ' * len(line.rstrip('\r\n')) + line[len(line.rstrip('\r\n')):])
            masked = True
        else:
            filtered.append(line)
    analysis = ''.join(filtered)
    result = parse_v6(analysis, candidates, mapping)
    strict = parse_v4(text, candidates, mapping)
    result['rpa_prediction'] = strict['rpa_prediction']
    result['parser_version'] = 'v7_question_options_experimental'
    result['excluded_question_options'] = masked
    if masked and not result['selection_prediction']:
        result['choice_needs_review'] = True
    # Inspect only the answer scope, excluding any later generated question.
    uncertainty_scope = analysis
    if result['excluded_generated_question']:
        q = re.search(r'^[ \t]*(?:Context|Question)[ \t]*:', analysis, re.I | re.M)
        if q:
            uncertainty_scope = analysis[:q.start()]
    uncertain = bool(re.search(r'\b(?:unclear|uncertain|unsure|ambiguous|ambiguity|not\s+sure|cannot\s+determine|can\s*not\s+determine|difficult\s+to\s+determine|not\s+enough\s+information|might|perhaps|possibly)\b', uncertainty_scope, re.I))
    result['expressed_uncertainty'] = uncertain
    if uncertain and result['selection_prediction'] is not None:
        result['choice_needs_review'] = True
    return result
