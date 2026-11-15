"""Experimental review flag for negated selections; strict RPA is unchanged."""
from parser_v8 import parse_response as parse_v8


def parse_response(text, candidates, mapping):
    result = parse_v8(text, candidates, mapping)
    if result['status'] == 'negated_candidate':
        result['choice_needs_review'] = True
    result['parser_version'] = 'v9_negation_review_experimental'
    return result
