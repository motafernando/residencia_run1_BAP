"""Optional exact-context reuse of assisted decisions, separate from v14 rules.

No registry is loaded implicitly. A lookup hit is reuse, not parser capability.
All original v14 fields survive in rule_output; strict audit and flags are never
rewritten. IDs are returned as provenance only, never used to select a decision.
"""
import copy
import hashlib
import json
from pathlib import Path

from parser_v14 import parse_response

SCHEMA = 'assisted-context-v1'


def package(text, sentence, candidates, option_map):
    return dict(text=text, sentence=sentence, candidates=list(candidates), option_map=dict(option_map))


def context_hash(text, sentence, candidates, option_map):
    payload = package(text, sentence, candidates, option_map)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _validate(entry, key):
    if entry['schema_version'] != SCHEMA or entry['context_sha256'] != key:
        raise ValueError('Registry schema or context key mismatch')
    raw = entry['package']
    if set(raw) != {'text', 'sentence', 'candidates', 'option_map'} or context_hash(**raw) != key:
        raise ValueError('Registry context integrity mismatch')
    choice = entry['adjudication']['choice']
    if choice is not None and choice not in ['Unknown', *raw['candidates']]:
        raise ValueError('Adjudication is outside the local alternatives')
    if entry['adjudication'].get('gold_used') is not False:
        raise ValueError('Registry requires an explicit gold-free adjudication')


def load_registry(path, expected_sha256=None):
    """Load a versioned JSONL; reject duplicates, tampering and invalid choices."""
    data = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Registry file SHA256 mismatch')
    registry = {}
    for line in data.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        key = entry['context_sha256']
        _validate(entry, key)
        if key in registry:
            raise ValueError('Duplicate context hash in registry')
        registry[key] = entry
    return registry


def parse(text, sentence, candidates, mapping, registry=None):
    """Apply v14 then optionally reuse one exact complete context decision.

    selection_prediction/primary_prediction become the final layered choice.
    status, choice_evidence, rpa_prediction, rpa_choice_prediction and every
    rule flag retain their v14 meanings. Consult selection_provenance for the
    final choice source; consult rule_output for a completely untouched result.
    """
    baseline = parse_response(text, candidates, mapping)
    result = copy.deepcopy(baseline)
    result['rule_output'] = copy.deepcopy(baseline)
    result['rule_selection'] = baseline['selection_prediction']
    result['adjudication_applied'] = False
    key = context_hash(text, sentence, candidates, mapping)
    provenance = dict(layer='rules_only', parser_version=baseline.get('parser_version'),
                      context_sha256=key, case_id=None,
                      status='disabled' if registry is None else 'hash_miss')
    result['selection_provenance'] = provenance
    if registry is None or key not in registry:
        return result
    entry = registry[key]
    # Do not trust a digest alone, including caller-created in-memory mappings.
    if entry.get('package') != package(text, sentence, candidates, mapping):
        provenance['status'] = 'registry_package_mismatch'
        return result
    _validate(entry, key)
    choice = entry['adjudication']['choice']
    result['selection_prediction'] = choice
    result['primary_prediction'] = choice
    result['selection_source'] = 'assisted_exact_context_registry'
    result['adjudication_applied'] = True
    result['adjudication'] = copy.deepcopy(entry['adjudication'])
    result['selection_provenance'] = dict(
        layer='assisted_adjudication_exact_context', status='exact_package_match',
        schema_version=entry['schema_version'], context_sha256=key,
        case_id=entry['case_id'], source=copy.deepcopy(entry['source']),
        adjudication_source=entry['adjudication'].get('source'),
        evidence=copy.deepcopy(entry['adjudication'].get('evidence')),
        baseline_parser_version=baseline.get('parser_version'))
    return result
