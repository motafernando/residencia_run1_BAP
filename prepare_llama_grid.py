"""Create versioned Llama grid manifests from the Mistral reference settings."""

import argparse
import copy
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
VARIANTS = ("original", "refined", "evaluation_task", "evaluation_gender_bias")
LLAMA_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"


def build_configs(base, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("A full, reviewed, lowercase 40-character checkpoint SHA is required")
    configs = {}
    for name in (*VARIANTS, "pilot_original"):
        config = copy.deepcopy(base)
        variant = "original" if name == "pilot_original" else name
        config["purpose"] = f"WinoBias Llama 3 8B Instruct prompt grid: {name}"
        config["prompt_variant"] = variant
        config["model"]["repo_id"] = LLAMA_REPO
        config["model"]["revision"] = revision
        config["model"]["tokenizer_revision"] = revision
        config["model"]["selection_status"] = "official_pre_article_candidate_author_revision_unconfirmed"
        config["checkpoint_provenance"] = (
            "Official pre-publication revision candidate; author-used revision unconfirmed. "
            "Weights and tokenizer from same reviewed snapshot; BF16 without quantization."
        )
        if name == "pilot_original":
            config["sample"] = "data/prepared/smoke24.json"
        configs[name] = config
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    base = json.loads((ROOT / "configs" / "mistral-full-3168.json").read_text(encoding="utf-8"))
    configs = build_configs(base, args.revision)
    targets = {name: ROOT / "configs" / f"llama-grid-{name}.json" for name in configs}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to replace existing manifests: " + ", ".join(existing))
    for name, target in targets.items():
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(configs[name], handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
