"""
Few-shot copying check (plan T5.1 b): prior work saw few-shot examples copied verbatim. For each generated reason,
the highest similarity (difflib ratio on lower-cased text) to any example sentence of the prompt; a reason counts as
copied at >= 0.9. Reasons whose item is identical to an example item (same feature, finding, direction, strength)
are reported apart, since a near-identical sentence is then expected.

    python explain/fewshot_copy.py --gens <run>/generations.jsonl --prompt hybrid_fs_v1
"""
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
COPY_RATIO = 0.9
KEY = ("feature_id", "finding", "direction", "strength")


def copy_stats(gens, spec):
    sentences = [r["text"].lower() for e in spec["examples"] for r in e["output"]["reasons"]]
    example_items = {tuple(i[k] for k in KEY) for e in spec["examples"] for i in e["evidence"]}
    out = {"reasons": 0, "copied": 0, "same_item_reasons": 0, "copied_same_item": 0}
    for g in gens:
        items = {i["feature_id"]: i for i in g["llm_input"]["evidence"]}
        reasons = (g.get("final_output") or {}).get("reasons") or []
        for r in reasons:
            best = max(SequenceMatcher(None, r.get("text", "").lower(), s).ratio() for s in sentences)
            item = items.get(r.get("feature_id"))
            same = item is not None and tuple(item.get(k) for k in KEY) in example_items
            key = "same_item_reasons" if same else "reasons"
            out[key] += 1
            out["copied_same_item" if same else "copied"] += best >= COPY_RATIO
    out["copy_rate"] = out["copied"] / out["reasons"] if out["reasons"] else None
    return out


def main(opts):
    spec = json.loads((HERE / "prompts" / f"{opts.prompt}.json").read_text(encoding="utf-8"))
    gens = [json.loads(line) for line in open(opts.gens, encoding="utf-8")]
    stats = copy_stats(gens, spec)
    Path(opts.gens).with_name("fewshot_copy.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--prompt", default="hybrid_fs_v1")
    main(ap.parse_args())
