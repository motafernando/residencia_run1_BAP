"""Experimental scope-only repair of v13; no semantic adjudication lookup.

v13 remains unchanged. Only explicitly paired single-line Markdown strikes and
explicit generated-question boundaries are masked, preserving source offsets.
No sentence, gold, profession inventory or case ID is consulted.
"""
import re

from parser_v13 import _visible, parse_response as parse_v13


def _blank(text):
    return re.sub(r'[^\r\n]', ' ', text)


def parse_response(text, candidates, mapping):
    raw = parse_v13(text, candidates, mapping)
    evidence = []
    # Do not interpret unpaired, escaped or fence-sized tildes as retraction.
    strike = re.compile(r'(?<![\\~])~~(?!~)[^\r\n]+?(?<![\\~])~~(?!~)')

    def mask_strike(match):
        evidence.append(dict(kind='paired_markdown_strike', start=match.start(),
                             end=match.end(), excerpt=match.group(0)))
        return _blank(match.group(0))

    scoped = strike.sub(mask_strike, text)
    # Re-evaluate inherited scope after removing explicit strikes: a retracted
    # leading answer must not cause all later answers to be excluded.
    inherited = parse_v13(scoped, candidates, mapping)
    visible = _visible(scoped)
    cutoff = None
    if inherited.get('excluded_generated_question'):
        match = re.search(r'^[ \t]*(?:Context|Question)[ \t]*:', visible, re.I | re.M)
        if match:
            cutoff = match.start()
    if cutoff is not None:
        evidence.append(dict(kind='inherited_generated_question', start=cutoff,
                             end=len(text), excerpt=text[cutoff:]))
        scoped = scoped[:cutoff] + _blank(scoped[cutoff:])
    else:
        # A leading question may have an actual answer afterwards. Resume only
        # at an explicit answer cue; an empty cue does not manufacture a choice.
        in_question = False
        offset = 0
        lines = []
        for line, surface in zip(scoped.splitlines(keepends=True), visible.splitlines(keepends=True)):
            if re.match(r'^[ \t]*Question[ \t]*:', surface, re.I):
                in_question = True
            elif in_question and re.match(
                    r'^[ \t]*(?:the\s+correct\s+)?(?:answer|choice|option)\s*(?:is|:)',
                    surface, re.I):
                in_question = False
            if in_question:
                evidence.append(dict(kind='explicit_question_block', start=offset,
                                     end=offset+len(line), excerpt=text[offset:offset+len(line)]))
                lines.append(_blank(line))
            else:
                lines.append(line)
            offset += len(line)
        scoped = ''.join(lines)
    result = parse_v13(scoped, candidates, mapping) if scoped != text else raw
    result['rpa_prediction'] = raw['rpa_prediction']
    result['parser_version'] = 'v14_explicit_scope_experimental'
    result['v14_masked_scope_evidence'] = evidence
    result['v14_scope_changed'] = scoped != text
    return result
