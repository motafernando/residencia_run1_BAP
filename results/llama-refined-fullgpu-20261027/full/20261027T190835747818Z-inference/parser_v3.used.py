"""Versioned response parsing with a letter-only selection fallback.

The original strict RPA interpretation remains available independently from
the answer-selection interpretation. Raw responses are never modified.
"""

import re

from protocol import parse_response as parse_response_v2


def parse_response(text, candidates, mapping):
    safe_mapping = {letter: mapping.get(letter) for letter in "ABC"}
    result = parse_response_v2(text, candidates, safe_mapping)
    result["parser_version"] = "v3_letter_fallback"
    result["selection_source"] = "named_answer" if result["selection_prediction"] else None

    bare_letter = re.fullmatch(r"\s*(?:Answer\s*:\s*)?\(?([ABC])\)?[.\s]*", text, re.I)
    letter_choice = safe_mapping.get(bare_letter.group(1).upper()) if bare_letter else None
    if letter_choice == "Unknown" or letter_choice in candidates:
        result["selection_prediction"] = letter_choice
        result["primary_prediction"] = letter_choice
        result["selection_source"] = "letter_only"
        result["status"] = "selected_unknown" if letter_choice == "Unknown" else "selected_occupation"

    selection = result["selection_prediction"]
    result["rpa_choice_prediction"] = selection if selection in candidates else None
    first_line = result["first_answer_line"]
    label = re.match(r"^(?:Answer\s*:\s*)?\(?([ABC])\)?[.)]\s*", first_line, re.I)
    mapped = safe_mapping.get(label.group(1).upper()) if label else None
    result["letter_text_conflict"] = bool(mapped and selection and mapped != selection)
    return result
