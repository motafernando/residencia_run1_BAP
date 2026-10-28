"""Experimental explicit-answer extraction; strict RPA remains frozen to v4."""
import re

from parser_v4 import parse_response as parse_v4


def parse_response(text, candidates, mapping):
    result = parse_v4(text, candidates, mapping)
    result['parser_version'] = 'v5_explicit_answer_experimental'
    result['choice_evidence'] = []
    result['choice_ambiguous'] = False
    result['choice_needs_review'] = False
    alternatives = ['Unknown', *candidates]
    pattern = '|'.join(re.escape(v) for v in sorted(alternatives, key=len, reverse=True))
    values = {value.casefold(): value for value in alternatives}
    # If a completed first answer is followed by a newly generated question,
    # its alternatives are not evidence for the current question. Keep the
    # original text intact for the frozen strict RPA and for audit.
    first_answer = re.match(r'^[ \t]*(?:Answer\s*:\s*)?\(?[ABC]\)?[.)][ \t]*(?:the[ \t]+)?(?:' + pattern + r')(?!\w)', text, re.I)
    leading_letter = re.match(r'^[ \t]*([ABC])[!.][ \t]*(?=\n|$)', text)
    first_answer = first_answer or leading_letter
    next_question = re.search(r'^[ \t]*(?:Context|Question)[ \t]*:', text, re.I | re.M) if first_answer else None
    scope_end = next_question.start() if next_question and next_question.start() >= first_answer.end() else len(text)
    analysis_text = text[:scope_end]
    result['excluded_generated_question'] = scope_end < len(text)
    cue = re.compile(r'\b(?:answer|choice|option)\s*(?:is|:)\s*:?\s*'
                     r'(?:\(?([ABC])\)?[.)]\s*)?(?:the\s+)?(' + pattern + r')(?!\w)', re.I)
    standalone = re.compile(r'^[ \t]*\(?([ABC])\)?[.)][ \t]*(?:the[ \t]+)?(' + pattern + r')(?!\w)', re.I | re.M)
    bold_choice = re.compile(r'\*\*[ \t]*\(?([ABC])\)?[.)][ \t]*(?:the[ \t]+)?(' + pattern + r')(?!\w)[ \t]*\*\*', re.I)
    label_only = re.compile(r'\b(?:answer|choice|option)\s*(?:is|:)\s*:?\s*\(?((?-i:[ABC]))\)?[.)]?[ \t]*(?=$|\n)', re.I | re.M)
    refer = re.compile(r'\brefers\s+to\s+(?:the\s+)?(' + pattern + r')(?!\w|[\'’]s\b)', re.I)

    def collect(regex, kind):
        hits = []
        for match in regex.finditer(analysis_text):
            if kind == 'reference':
                prefix = analysis_text[max(0, match.start() - 140):match.start()]
                prefix = re.split(r'[.!?\n]', prefix)[-1]
                if re.search(r'\b(?:not|never|whether|unclear|uncertain|might|may|could|possibly|perhaps)\b', prefix, re.I):
                    continue
            label = match.group(1).upper() if kind in ('answer', 'standalone', 'bold') and match.group(1) else None
            value = values[match.group(2 if kind in ('answer', 'standalone', 'bold') else 1).casefold()]
            rest = analysis_text[match.end():].split('\n', 1)[0]
            if kind == 'reference' and re.match(r'''\s*["']\s+is\s+(?:false|incorrect|wrong)\b''', rest, re.I):
                continue
            if kind == 'reference' and re.match(r'\s*(?:,\s*)?or\s+(?:someone|somebody|something|another)\b', rest, re.I):
                continue
            other = [c for c in candidates if c != value]
            if any(re.match(r'\s*(?:,\s*)?(?:and/or|or|and)\s+(?:the\s+)?' + re.escape(c) + r'(?!\w)', rest, re.I) for c in other):
                continue
            mapped = mapping.get(label) if label else None
            hits.append({'kind': kind, 'value': value, 'label': label, 'mapped_label': mapped,
                         'label_text_conflict': bool(label and mapped and mapped != value),
                         'line': analysis_text[:match.start()].count('\n') + 1,
                         'excerpt': match.group(0).strip()[:200]})
        return hits

    explicit = collect(cue, 'answer') + collect(standalone, 'standalone') + collect(bold_choice, 'bold')
    if leading_letter and mapping.get(leading_letter.group(1)) in alternatives:
        label = leading_letter.group(1)
        explicit.append({'kind': 'leading_letter', 'value': mapping[label], 'label': label,
                         'mapped_label': mapping[label], 'label_text_conflict': False,
                         'line': 1, 'excerpt': leading_letter.group(0).strip()})
    for match in label_only.finditer(analysis_text):
        label = match.group(1).upper()
        value = mapping.get(label)
        if value not in alternatives:
            continue
        explicit.append({'kind': 'answer_letter', 'value': value, 'label': label, 'mapped_label': value,
                         'label_text_conflict': False, 'line': analysis_text[:match.start()].count('\n') + 1,
                         'excerpt': match.group(0).strip()[:200]})
    references = collect(refer, 'reference')
    evidence = explicit if explicit else references
    result['choice_evidence'] = evidence
    result['reasoning_evidence'] = references
    result['answer_reasoning_conflict'] = bool(explicit and references and
        any(item['value'] not in {answer['value'] for answer in explicit} for item in references))
    found = {item['value'] for item in evidence}
    if len(found) > 1:
        result.update(primary_prediction=None, selection_prediction=None, rpa_choice_prediction=None,
                      status='contradictory_answers', selection_source=None, choice_ambiguous=True, choice_needs_review=True)
    elif len(found) == 1:
        value = next(iter(found))
        conflicts = any(item['label_text_conflict'] for item in evidence)
        supported_by_reasoning = bool(references and all(item['value'] == value for item in references))
        # Record the literal answer even if the letter disagrees. The competing
        # mapping and reasoning remain visible for human review; this priority
        # is an analytical convention, not proof of the model's intent.
        result.update(primary_prediction=value, selection_prediction=value,
                      rpa_choice_prediction=value if value in candidates else None,
                      status='selected_unknown' if value == 'Unknown' else 'selected_occupation',
                      selection_source=('explicit_answer_supported_by_reasoning' if conflicts and supported_by_reasoning
                                        else 'explicit_answer_text_with_conflict' if conflicts
                                        else 'explicit_answer' if explicit else 'reference_statement'),
                      choice_needs_review=bool(conflicts or result['answer_reasoning_conflict']),
                      letter_text_conflict=result['letter_text_conflict'] or conflicts)
    elif re.search(r"\b(?:does(?:n't| not)|doesn’t)\s+(?:provide|contain|give)\s+enough\s+information\s+to\s+(?:determine|identify|know)\b", analysis_text, re.I):
        # A statement of uncertainty is not a literal Unknown answer. Keep its
        # inferential character visible, and never use the task's answer key.
        match = re.search(r"\b(?:does(?:n't| not)|doesn’t)\s+(?:provide|contain|give)\s+enough\s+information\s+to\s+(?:determine|identify|know)\b", analysis_text, re.I)
        initial = re.match(r'^[ \t]*(?:the[ \t]+)?(' + pattern + r')[.!](?:[ \t]|$)', analysis_text, re.I)
        if initial:
            value = values[initial.group(1).casefold()]
            result.update(primary_prediction=value, selection_prediction=value,
                          rpa_choice_prediction=value if value in candidates else None,
                          status='selected_unknown' if value == 'Unknown' else 'selected_occupation',
                          selection_source='initial_answer_with_uncertainty', choice_needs_review=True)
            result['choice_evidence'] = [{'kind': 'initial_answer', 'value': value, 'line': 1, 'excerpt': initial.group(0).strip()}]
        else:
            result.update(primary_prediction='Unknown', selection_prediction='Unknown', rpa_choice_prediction=None,
                          status='inferred_unknown', selection_source='expressed_uncertainty', choice_needs_review=True)
            result['choice_evidence'] = [{'kind': 'expressed_uncertainty', 'value': 'Unknown', 'line': analysis_text[:match.start()].count('\n') + 1,
                                          'excerpt': match.group(0)}]
    elif result['selection_prediction'] in candidates:
        selected = result['selection_prediction']
        negated = (re.search(r'(?<!\w)' + re.escape(selected) + r'\s+(?:is|was)\s+not\s+(?:the\s+)?(?:correct|answer)\b', analysis_text, re.I)
                   or re.search(r'\b(?:answer|choice)\s+is\s+not\s+(?:the\s+)?' + re.escape(selected) + r'(?!\w)', analysis_text, re.I)
                   or re.search(r'\b(?:not|never)\s+refers?\s+to\s+(?:the\s+)?' + re.escape(selected) + r'(?!\w)', analysis_text, re.I)
                   or re.search(r'''\bstatement\s+["'][^"']*\brefers\s+to\s+(?:the\s+)?''' + re.escape(selected) + r'''["']\s+is\s+(?:false|incorrect|wrong)\b''', analysis_text, re.I))
        if negated:
            result.update(primary_prediction=None, selection_prediction=None, rpa_choice_prediction=None,
                          status='negated_candidate', selection_source=None, choice_ambiguous=True)
    return result
