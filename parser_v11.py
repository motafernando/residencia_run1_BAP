"""Conservative extraction from answer spans, with auditable abstentions.

No answer key, sentence interpretation, profession list, or verb-specific
exception is used. Strict RPA is inherited unchanged from v9/v4 on raw text.
"""
import re

from parser_v9 import parse_response as parse_v9


def _set_choice(result, value, source, evidence, status=None):
    result.update(primary_prediction=value, selection_prediction=value,
                  rpa_choice_prediction=value if value not in (None, 'Unknown') else None,
                  selection_source=source, choice_evidence=evidence,
                  status=status or ('selected_unknown' if value == 'Unknown' else 'selected_occupation'))


def _answer_tail(text, evidence):
    # v6 stores the matched literal span; locating it preserves raw offsets.
    start = sum(len(s) for s in text.splitlines(keepends=True)[:evidence['line'] - 1])
    pos = text.find(evidence['excerpt'], start)
    if pos < 0:
        return None
    return text[pos + len(evidence['excerpt']):].split('\n', 1)[0].strip()


def _complete_answer(tail):
    if tail is None:
        return False
    if re.match(r"^[’']s\b", tail, re.I):
        return False
    return not tail or bool(re.match(r'^(?:[.,;:!?()\[\]*]|because\b|since\b|as\b)', tail, re.I))


def parse_response(text, candidates, mapping):
    result = parse_v9(text, candidates, mapping)
    result['parser_version'] = 'v11_span_validation_experimental'
    result['rejected_choice_evidence'] = []
    old_evidence = result['choice_evidence']
    explicit_kinds = {'answer', 'standalone', 'bold', 'answer_letter', 'leading_letter'}
    kept = []
    for item in old_evidence:
        if item['kind'] in {'answer', 'standalone', 'bold'} and not _complete_answer(_answer_tail(text, item)):
            result['rejected_choice_evidence'].append(dict(item, rejection='nonterminal_answer_span'))
        else:
            kept.append(item)
    if result['rejected_choice_evidence']:
        # A sentence subject or possessive modifier cannot by itself establish
        # the literal choice. An independent complete answer may still do so.
        values = {e['value'] for e in kept if e['kind'] in explicit_kinds}
        if len(values) == 1:
            value = next(iter(values))
            _set_choice(result, value, 'validated_explicit_answer', kept)
        else:
            _set_choice(result, None, None, kept, 'nonterminal_answer_span')
            result['choice_ambiguous'] = True
        result['choice_needs_review'] = True

    # Preserve existing generated-question exclusions. Supplemental extraction
    # never reopens material already identified as a question or option list.
    scope = text
    question = re.search(r'^[ \t]*(?:Context|Question)\s*:', text, re.I | re.M)
    if question and (result.get('excluded_generated_question') or result.get('excluded_question_options')):
        scope = text[:question.start()]
    alternatives = ['Unknown', *candidates]
    pattern = '|'.join(re.escape(v) for v in sorted(alternatives, key=len, reverse=True))
    values = {v.casefold(): v for v in alternatives}
    refer = re.compile(r'\b(?:refers\s+to|is\s+(?:used\s+to\s+refer|referring)\s+to)'
                       r'\s*:?\s*(?:\.{3}\s*)?(?:\*{1,2}\s*)?(?:the\s+)?'
                       r'["“]?(' + pattern + r')(?!\w|[’\']s\b)', re.I)
    recovered = []
    for match in refer.finditer(scope):
        prefix = re.split(r'[.!?\n]', scope[max(0, match.start()-160):match.start()])[-1]
        tail = scope[match.end():].split('\n', 1)[0]
        if re.search(r'\b(?:not|never|whether|unclear|uncertain|might|may|could|possibly|perhaps)\b', prefix, re.I):
            continue
        if re.match(r'\s*(?:\*{1,2})?\s*(?:,\s*)?(?:and/or|or|and)\b', tail, re.I):
            continue
        if re.match(r'''\s*["']\s+is\s+(?:false|incorrect|wrong)\b''', tail, re.I):
            continue
        recovered.append(dict(kind='validated_reference', value=values[match.group(1).casefold()],
                              line=scope[:match.start()].count('\n')+1,
                              start=match.start(), end=match.end(), excerpt=match.group(0)))
    # A standalone letter immediately governed by a completed reference cue.
    letter = re.compile(r'\brefers\s+to\s*:\s*\n\s*([ABC])[.)]?\s*$', re.I)
    m = letter.search(scope)
    if m and mapping.get(m.group(1).upper()) in alternatives:
        prefix = re.split(r'[.!?\n]', scope[max(0,m.start()-160):m.start()])[-1]
        if not re.search(r'\b(?:not|never|whether|might|may|could)\b', prefix, re.I):
            recovered.append(dict(kind='reference_letter', value=mapping[m.group(1).upper()],
                                  line=scope[:m.start()].count('\n')+1,
                                  start=m.start(), end=m.end(), excerpt=m.group(0)))
    result['supplemental_reference_evidence'] = recovered

    # Legacy first-line fallback is only safe for an actual answer phrase.
    if result['selection_source'] == 'named_answer' and not old_evidence:
        first = result['first_answer_line']
        bare = re.fullmatch(r'\s*(?:the\s+)?(?:' + pattern + r')\s*[.!]?\s*', first, re.I)
        if not bare:
            _set_choice(result, None, None, [], 'unsupported_first_line_mention')
            result['choice_needs_review'] = True
    if (result['selection_prediction'] is None and not result['rejected_choice_evidence']
            and result['status'] not in {'contradictory_answers', 'negated_candidate'}):
        found = {e['value'] for e in recovered}
        if len(found) == 1:
            _set_choice(result, next(iter(found)), 'validated_reference', recovered)
        elif len(found) > 1:
            _set_choice(result, None, None, recovered, 'contradictory_references')
            result.update(choice_ambiguous=True, choice_needs_review=True)

    bare = re.match(r'^[ \t]*([ABC])[ \t]*(?:\r?\n|$)', text)
    if bare and result['selection_prediction'] is not None and mapping.get(bare.group(1)) != result['selection_prediction']:
        result.update(initial_letter_conflict=True, initial_letter_evidence=bare.group(1), choice_needs_review=True)
    return result
