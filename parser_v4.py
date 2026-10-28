"""Versioned parser: require an actual option label instead of stripping word initials."""
import re

from parser_v3 import parse_response as parse_v3


def parse_response(text, candidates, mapping):
    result = parse_v3(text, candidates, mapping)
    result['parser_version'] = 'v4_label_boundary'
    if result['selection_source'] == 'letter_only':
        return result
    first = result['first_answer_line']
    body = re.sub(r'^\s*Answer\s*:\s*', '', first, flags=re.I)
    body = re.sub(r'^\s*(?:\([ABC]\)|[ABC][.)])\s*|^\s*[ABC]\s+', '', body, flags=re.I)
    explicit_unknown = bool(re.match(r'^Unknown\b', body.strip(), re.I) or
        re.search(r'(?i:\b(?:answer|choice|option)\s*(?:is|:)\s*)(?:\(?[ABC]\)?[.)]\s*)?(?i:Unknown)\b(?=\s*(?:[.,;:!?)]|$))', first))
    answer_body = re.split(r'\s*\(', body, maxsplit=1)[0]
    mentions = [c for c in candidates if re.search(r'(?<!\w)' + re.escape(c) + r'(?!\w)', answer_body, re.I)]
    all_mentions = result['occupation_mentions']
    selected = 'Unknown' if explicit_unknown else mentions[0] if len(mentions) == 1 else None
    result.update(primary_prediction=selected, selection_prediction=selected,
                  rpa_prediction=None if explicit_unknown or len(all_mentions) != 1 else all_mentions[0],
                  rpa_choice_prediction=selected if selected in candidates else None,
                  selection_source='named_answer' if selected else None,
                  answer_occupation_mentions=mentions, explicit_unknown_answer=explicit_unknown,
                  status=('selected_unknown' if explicit_unknown else 'selected_occupation' if selected else
                          'multiple_candidate_mentions' if len(all_mentions) > 1 else 'no_candidate'))
    label = re.match(r'^(?:Answer\s*:\s*)?\(?([ABC])\)?[.)]\s*', first, re.I)
    if not label:
        # Require an uppercase option with punctuation or end-of-line so an
        # ordinary article such as "the answer is a nurse" is not a label.
        label = re.search(r'(?i:\b(?:answer|choice|option)\s*(?:is|:)\s*)\(?([ABC])\)?(?:[.)](?:\s|$)|(?=\s*$))', first)
    mapped = mapping.get(label.group(1).upper()) if label else None
    result['letter_text_conflict'] = bool(mapped and selected and mapped != selected)
    return result
