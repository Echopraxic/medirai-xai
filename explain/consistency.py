"""
Consistency of explanations (plan T6.3), complementing explain/counterfactuals.py (sensitivity, determinism).

  similar inputs   cases whose number-free inputs give an item the same (feature_id, finding, relative_to_benign,
                   direction, strength) should get the same claim. For each such item signature seen in at least
                   two cases: mean pairwise token Jaccard of the claim texts, and the share of pairs that are
                   identical. Compared with random pairs of claims citing the same feature_id with a DIFFERENT
                   signature, which should be less similar if the text follows the evidence.
  paraphrase       the same inputs run with a paraphrased prompt (e.g. hybrid_v1 vs hybrid_v1p): per case and
                   feature_id, same direction cited, and token Jaccard of the two claims.

    python explain/consistency.py similar explain/output/<run>/<model>__hybrid_v1/generations.jsonl
    python explain/consistency.py paraphrase <generations hybrid_v1> <generations hybrid_v1p>
"""
import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from counterfactuals import claims, jaccard

MAX_PAIRS_PER_GROUP = 50


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def similar(path, seed=0):
    rng = random.Random(seed)
    groups, by_feature = defaultdict(list), defaultdict(list)
    for g in load(path):
        cl = claims(g)
        for e in g["llm_input"]["evidence"]:
            c = cl.get(e["feature_id"])
            if not c:
                continue
            sig = (e["feature_id"], e["finding"], e["relative_to_benign"], e["direction"], e["strength"],
                   g["llm_input"]["prediction"])
            groups[sig].append(c.get("text", ""))
            by_feature[e["feature_id"]].append((sig, c.get("text", "")))
    same_sims, same_identical = [], []
    for texts in groups.values():
        pairs = list(itertools.combinations(texts, 2))
        for a, b in rng.sample(pairs, min(MAX_PAIRS_PER_GROUP, len(pairs))):
            same_sims.append(jaccard(a, b))
            same_identical.append(a == b)
    diff_sims = []
    for items in by_feature.values():
        for _ in range(min(MAX_PAIRS_PER_GROUP, len(items))):
            (s1, t1), (s2, t2) = rng.sample(items, 2) if len(items) > 1 else (items[0], items[0])
            if s1 != s2:
                diff_sims.append(jaccard(t1, t2))
    report = {"generations": str(path), "n_signatures_with_2plus": sum(len(t) > 1 for t in groups.values()),
              "same_signature": {"n_pairs": len(same_sims), "mean_jaccard": float(np.mean(same_sims)) if same_sims else None,
                                 "identical_rate": float(np.mean(same_identical)) if same_identical else None},
              "different_signature_same_feature": {"n_pairs": len(diff_sims),
                                                   "mean_jaccard": float(np.mean(diff_sims)) if diff_sims else None}}
    return report


def paraphrase(path_a, path_b):
    a = {g.get("row_id", g["isic_id"]): g for g in load(path_a)}
    b = {g.get("row_id", g["isic_id"]): g for g in load(path_b)}
    same_dir, sims, missing = [], [], 0
    for rid in a.keys() & b.keys():
        ca, cb = claims(a[rid]), claims(b[rid])
        for f, c in ca.items():
            if f not in cb:
                missing += 1
                continue
            same_dir.append(c.get("direction") == cb[f].get("direction"))
            sims.append(jaccard(c.get("text"), cb[f].get("text")))
    return {"a": str(path_a), "b": str(path_b), "n_claims": len(sims), "claims_missing_in_b": missing,
            "same_direction_rate": float(np.mean(same_dir)) if same_dir else None,
            "mean_jaccard": float(np.mean(sims)) if sims else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("similar")
    s.add_argument("generations")
    p = sub.add_parser("paraphrase")
    p.add_argument("a")
    p.add_argument("b")
    opts = ap.parse_args()
    if opts.cmd == "similar":
        rep = similar(opts.generations)
        Path(opts.generations).with_name("consistency_similar.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    else:
        rep = paraphrase(opts.a, opts.b)
        Path(opts.b).with_name("consistency_paraphrase.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
