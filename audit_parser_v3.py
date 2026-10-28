"""Compare parsing policies offline without modifying a saved inference run."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from parser_v3 import parse_response as parse_v3
from protocol import metrics, parse_response as parse_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    response_path = args.run_dir / "responses.jsonl"
    if args.output.resolve() == response_path.resolve() or args.output.exists():
        raise FileExistsError("Output must be a new file, separate from saved responses")
    source_hash = hashlib.sha256(response_path.read_bytes()).hexdigest()
    rows_v2, rows_v3 = [], []
    summaries = {c: Counter() for c in ("zero_shot", "three_shot")}
    with response_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            old, new = dict(row), dict(row)
            if row["condition"] != "intrinsic" and row.get("status") == "ok":
                c = row["condition"]
                a = parse_v2(row.get("generated_text", ""), row["candidates"], row.get("option_map", {}))
                b = parse_v3(row.get("generated_text", ""), row["candidates"], row.get("option_map", {}))
                summaries[c]["rows"] += 1
                summaries[c]["letter_only_recovered"] += b["selection_source"] == "letter_only"
                summaries[c]["letter_text_conflict"] += b["letter_text_conflict"]
                summaries[c]["selection_changed"] += a["selection_prediction"] != b["selection_prediction"]
                summaries[c]["strict_rpa_prediction_changed"] += a["rpa_prediction"] != b["rpa_prediction"]
                if row["type"] == 2:
                    summaries[c]["type2_total"] += 1
                    summaries[c]["type2_strict_correct"] += b["rpa_prediction"] == row["gold"]
                    summaries[c]["type2_choice_correct"] += b["rpa_choice_prediction"] == row["gold"]
                for target, parsed in ((old, a), (new, b)):
                    target["parsing"] = parsed
                    target["prediction"] = parsed["selection_prediction"]
                    target["selection_prediction"] = parsed["selection_prediction"]
                    target["rpa_prediction"] = parsed["rpa_prediction"]
            rows_v2.append(old)
            rows_v3.append(new)
    occupations = sorted({occupation for row in rows_v3 for occupation in row.get("candidates", [])})
    metrics_v2 = metrics(rows_v2, occupations)
    metrics_v3 = metrics(rows_v3, occupations)
    output = {
        "source_responses": str(response_path),
        "source_responses_sha256": source_hash,
        "description": "Offline parser comparison; existing raw data and historical metrics unchanged.",
        "strict_rpa_policy": "The v2 criterion remains the primary RPA; letter-only answers do not name the referent.",
        "choice_rpa_policy": "Additional local diagnostic based on selected answer, including mapped bare letters.",
        "conditions": {},
    }
    for condition, counts in summaries.items():
        output["conditions"][condition] = {
            **dict(counts),
            "a_sb_v2_by_type": {k: v["a_sb_40"] for k, v in metrics_v2["conditions"][condition]["bias_by_type"].items()},
            "a_sb_v3_by_type": {k: v["a_sb_40"] for k, v in metrics_v3["conditions"][condition]["bias_by_type"].items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
