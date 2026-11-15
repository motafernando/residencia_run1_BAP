"""Experimental uncertainty audit; v7 selection and strict RPA stay unchanged."""
import re
from parser_v7 import parse_response as parse_v7


def parse_response(text, candidates, mapping):
    result = parse_v7(text, candidates, mapping)
    scope = text
    if result['excluded_generated_question']:
        question = re.search(r'^[ \t]*(?:Context|Question)[ \t]*:', text, re.I | re.M)
        if question:
            scope = text[:question.start()]
    if re.search(r"\bit(?:'s|’s| is)\s+not\s+clear\s+(?:who|whether|which)\b", scope, re.I):
        result['expressed_uncertainty'] = True
        if result['selection_prediction'] is not None:
            result['choice_needs_review'] = True
    result['parser_version'] = 'v8_uncertainty_experimental'
    return result
