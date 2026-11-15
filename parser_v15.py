"""Consolidated v15 entry point for rules, assisted context, and surface gate.

The deterministic parser remains v14. Exact-context assisted decisions are
loaded only when the caller explicitly requests the frozen registry. Context
reuse is provenance, not a claim that the lookup generalizes. No gold, item ID,
gender or task type is accepted by this API.
"""

from __future__ import annotations

import re
from pathlib import Path

import adjudicacao_lateral as lateral


PARSER_VERSION = "v15_consolidated_layered"
ROOT = Path(__file__).resolve().parent
FROZEN_REGISTRY_PATH = (
    ROOT
    / "analises"
    / "revisao-parser-integral-20261109"
    / "camada-lateral"
    / "registro-assistido-contextual-v1.jsonl"
)
FROZEN_REGISTRY_SHA256 = "825f79a5cb0d6012637e359f98b784af6d9aba2d8bd5799f142e99b3fc654a2a"

_RISK_BOOLEANS = (
    "choice_needs_review",
    "letter_text_conflict",
    "initial_letter_conflict",
    "choice_ambiguous",
    "answer_reasoning_conflict",
    "excluded_generated_question",
    "expressed_uncertainty",
    "excluded_question_options",
    "v14_scope_changed",
)


def load_frozen_registry():
    """Explicitly load the frozen 9,435-context, gold-free registry."""
    return lateral.load_registry(FROZEN_REGISTRY_PATH, expected_sha256=FROZEN_REGISTRY_SHA256)


def _risk_signals(parsed):
    found = [key for key in _RISK_BOOLEANS if parsed.get(key)]
    selected = parsed.get("selection_prediction")
    if selected is None:
        found.append("selecao_nula")
    if parsed.get("rejected_choice_evidence"):
        found.append("evidencia_de_escolha_rejeitada")
    evidence = parsed.get("choice_evidence") or []
    if not evidence:
        found.append("sem_evidencia_de_escolha")
    elif any(item.get("value") != selected for item in evidence):
        found.append("evidencia_nao_coincide_com_selecao")
    return found


def _surface_class(text, parsed):
    if parsed.get("selection_prediction") == "Unknown":
        return "unknown_explicito"
    if parsed.get("selection_source") == "reference_statement":
        return "referencia_direta"
    if re.fullmatch(r"[\s*`\"\[({]*[ABC][\s.*`\"\])}:;!]*", text or "", re.I):
        return "rotulo_isolado_mapeado"
    if re.search(
        r"\b(?:not|no|neither|cannot|can.t|doesn.t|isn.t|wasn.t|rather\s+than)\b",
        text or "",
        re.I,
    ):
        return "ocupacao_explicita_com_negacao_na_explicacao"
    return "ocupacao_explicita"


def parse_response(text, candidates, mapping, *, sentence=None, registry=None):
    """Return one layered result while keeping rule and provenance separate.

    `registry` is never loaded implicitly. Exact package equality is required
    for an assisted decision. A miss falls back to v14 and receives a surface
    gate, which is a routing decision rather than human semantic ground truth.
    """
    result = lateral.parse(text, sentence, candidates, mapping, registry=registry)
    rule = result["rule_output"]
    risk = _risk_signals(rule)
    if registry is not None and not isinstance(sentence, str):
        risk.append("sentenca_ausente_para_lookup_contextual")
    surface_class = _surface_class(text, rule) if not risk else None
    if result["adjudication_applied"]:
        tier = "assisted_exact_context"
        review_level = "triangulated_assisted"
    elif not risk:
        tier = "surface_consistent_rules"
        review_level = "surface_only"
    else:
        tier = "rules_needs_review"
        review_level = "unreviewed"
    result["parser_version"] = PARSER_VERSION
    result["v15_processing"] = {
        "tier": tier,
        "surface_class": surface_class,
        "surface_status": "consistente_superficial" if not risk else "revisao_aprofundada",
        "semantic_review_level": review_level,
        "risk_signals": risk,
        "registry_enabled": registry is not None,
        "gold_used": False,
    }
    return result


def parse_record(record, *, registry=None):
    """Adapt a saved row while ignoring gold and evaluation metadata."""
    return parse_response(
        record.get("generated_text", ""),
        record["candidates"],
        record["option_map"],
        sentence=record.get("sentence"),
        registry=registry,
    )
