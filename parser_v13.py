"""Experimental structural extraction, not a semantic adjudication lookup.

Candidate vocabulary comes only from the call. No sentence, ID, gold, profession
list or adjudication table is consulted. v12 remains untouched; strict RPA is
computed on the raw response. Promotion requires the Marechal gate.
"""
import re

from parser_v12 import parse_response as parse_v12


def _choice(result, value, evidence, status=None):
    result.update(primary_prediction=value, selection_prediction=value,
                  rpa_choice_prediction=value if value not in (None, 'Unknown') else None,
                  selection_source='v13_structural_reference' if value is not None else None,
                  status=status or ('selected_unknown' if value == 'Unknown' else 'selected_occupation'),
                  choice_evidence=evidence)


def _visible(text):
    """Mask explicit quotation surfaces, preserving character offsets.

    Inline pronoun quotation is grammatical evidence and is preserved. Markdown
    emphasis is spacing, not content. This does not claim to resolve every
    quotation, hypothetical, or generated-question boundary.
    """
    lines = []
    fenced = False
    for line in text.splitlines(keepends=True):
        delimiter = bool(re.match(r'^\s*(?:```|~~~)', line))
        excluded = fenced or delimiter or bool(re.match(r'^\s*(?:>|Example\s*:|Quotation\s*:)', line, re.I))
        excluded = excluded or bool(re.fullmatch(r'''\s*(["']).*\1\s*[.!]?\s*''', line.rstrip('\r\n')))
        if delimiter:
            fenced = not fenced
        lines.append(re.sub(r'[^\r\n]', ' ', line) if excluded else line)
    return re.sub(r'[*`_]', ' ', ''.join(lines))


def parse_response(text, candidates, mapping):
    raw_result = parse_v12(text, candidates, mapping)
    visible = _visible(text)
    # Reparse only when explicit quote surfaces were masked. Emphasis-only
    # normalization is used by the structural collector, not the legacy parser.
    masked = ''.join(' ' if a != b and a not in '*`_' else a
                     for a, b in zip(text, visible))
    result = parse_v12(masked, candidates, mapping) if masked != text else raw_result
    result['rpa_prediction'] = raw_result['rpa_prediction']
    result['parser_version'] = 'v13_structural_experimental'
    result['v13_reference_evidence'] = []
    result['v13_rejected_reference_evidence'] = []
    result['v13_masked_quote_surface'] = masked != text

    alternatives = ['Unknown', *candidates]
    values = {v.casefold(): v for v in alternatives}
    vocab = '|'.join(re.escape(v) for v in sorted(alternatives, key=len, reverse=True))
    # The grammatical subject must be a pronoun, not a profession in a story.
    # Parenthetical asides are transparent only between the reference predicate
    # and an explicit label/candidate. Their words are never used as a choice.
    subject = r'(?:the\s+pronoun\s+)?["\'“‘]?(?:he|she|her|his|him)["\'”’]?'
    aside = r'(?:\s*(?:\.{2,}|…|\([^()\n]*\)))\s*'
    reference = re.compile(
        r'(?<!\w)' + subject + r'\s+(?P<predicate>refers\s+to|is\s+referring\s+to|is\s+used\s+to\s+refer\s+to)'
        r'\s*:?\s*(?:' + aside + r')*'
        r'(?:(?P<label>[ABC])[.)]\s*)?(?:the\s+)?["“]?'
        r'(?P<value>' + vocab + r')(?!\w|[’\']s\b)', re.I)
    recovered = []
    for match in reference.finditer(visible):
        prefix = re.split(r'[.!?\n]', visible[:match.start()])[-1]
        tail = visible[match.end():].split('\n', 1)[0]
        rejection = None
        if re.search(r'\b(?:example|quote|quotation|hypothetical|if|whether|not|never|may|might|could|perhaps|possibly)\b', prefix, re.I):
            rejection = 'quoted_hypothetical_or_nonassertive_prefix'
        elif re.search(r'\b(?:Context|Question)\s*:', prefix, re.I):
            rejection = 'context_or_question_echo'
        elif re.match(r'\s*["”]?\s*(?:,\s*)?(?:and/or|or|and)\b', tail, re.I):
            rejection = 'nonexclusive_reference'
        elif re.match(r'\s*["”\']\s+is\s+(?:false|incorrect|wrong)\b', tail, re.I):
            rejection = 'rejected_quoted_proposition'
        elif tail and not re.match(r'\s*(?:$|["”.,;:!?()\[\]]|because\b|since\b|as\b)', tail, re.I):
            rejection = 'nonterminal_reference_span'
        item = dict(kind='structural_reference', value=values[match['value'].casefold()],
                    label=match['label'].upper() if match['label'] else None,
                    start=match.start(), end=match.end(), excerpt=text[match.start():match.end()])
        item['mapped_label'] = mapping.get(item['label'])
        item['label_text_conflict'] = bool(item['mapped_label'] is not None and item['mapped_label'] != item['value'])
        if rejection:
            result['v13_rejected_reference_evidence'].append(dict(item, rejection=rejection))
        else:
            recovered.append(item)
    result['v13_reference_evidence'] = recovered

    # A legacy reference match cannot outvote a structural rejection of that
    # same candidate. An independent explicit answer is kept; rejection alone
    # never supplies the other candidate or Unknown.
    current = result['selection_prediction']
    if (result['selection_source'] in {'reference_statement', 'validated_reference'}
            and any(e['value'] == current for e in result['v13_rejected_reference_evidence'])
            and not any(e['value'] == current for e in recovered)):
        _choice(result, None, [], 'rejected_reference_scope_or_span')
        result['choice_needs_review'] = True

    # Conflicts in a complete answer span survive explanations, Markdown and
    # parenthetical continuations. Do not use a majority vote over mentions.
    explicit_conflicts = [e for e in result['choice_evidence']
                          if e.get('label_text_conflict') and e['kind'] in
                          {'answer', 'standalone', 'bold'}]
    conflicts = explicit_conflicts + [e for e in recovered if e['label_text_conflict']]
    if conflicts:
        result['v13_conflict_evidence'] = conflicts
        _choice(result, None, result['choice_evidence'] + recovered,
                'unresolved_label_text_conflict')
        result.update(choice_ambiguous=True, choice_needs_review=True, letter_text_conflict=True)
        return result

    if recovered:
        found = {e['value'] for e in recovered}
        current = result['selection_prediction']
        if current is not None:
            found.add(current)
        if len(found) > 1:
            _choice(result, None, result['choice_evidence'] + recovered,
                    'unresolved_explicit_reference_conflict')
            result.update(choice_ambiguous=True, choice_needs_review=True,
                          answer_reasoning_conflict=True)
        elif current is None and not result['choice_ambiguous']:
            _choice(result, next(iter(found)), recovered)
    return result
