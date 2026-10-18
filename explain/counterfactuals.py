"""
Counterfactual sensitivity and stability of explanations (plan T6.2) + repeat-run determinism (T6.3).

`make` perturbs the TreeSHAP payload of a stratified test sample, one evidence item at a time:
  flip    reverse the item's direction (supports <-> argues_against); everything else unchanged
  swap    replace the item with an item of a concept group the case does not have, taken from another case
  drop    remove the item (cases with at least two items)
  repeat  an unchanged copy of the case (identical input -> identical output expected)
Each variant is a row with its own `row_id` (`<isic_id>#<variant>`) for explain/llm_generate.py; validation rows
are copied unchanged so the generator derives the same entropy threshold.

`score` pairs every variant's generation with the base generation of the same case:
  sensitivity  the perturbed claim changed as it should:
               flip -> the claim for that feature_id has the new direction and says so;
               swap -> the new feature_id is cited and the old one is not;
               drop -> the dropped feature_id is no longer cited
  stability    every untouched item keeps its feature_id and direction, and its sentence stays close to the base
               sentence (token Jaccard; reported as the share >= STABLE_JACCARD and the mean)
  determinism  repeat output identical to base output (exact raw text)

    python explain/counterfactuals.py make --inputs trees/output_v1.0/explanation_inputs.jsonl \
        --out explain/output/counterfactual_inputs.jsonl --per_stratum 75
    python explain/llm_generate.py --inputs explain/output/counterfactual_inputs.jsonl ... --out explain/output/<run>
    python explain/counterfactuals.py score explain/output/<run>/<model>__<prompt>/generations.jsonl
"""
import argparse
import copy
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import verifier

STABLE_JACCARD = 0.6
SEED = 0


def stratum(row, threshold):
    pl = row["llm_input"]
    correct = (pl["prediction"] == "malignant") == (row["label"] == 1)
    flag = pl["uncertainty"].get("deferred")  # uncertainty-table flag when attached (llm_input.is_uncertain)
    uncertain = bool(flag) if flag is not None else pl["uncertainty"]["entropy_of_expected"] > threshold
    return f"{'correct' if correct else 'wrong'}|{'uncertain' if uncertain else 'confident'}"


def make(opts):
    rows = [json.loads(line) for line in open(opts.inputs, encoding="utf-8")]
    val = [r for r in rows if r["split"] == "val"]
    threshold = float(np.quantile([r["llm_input"]["uncertainty"]["entropy_of_expected"] for r in val], 0.75))
    test = [r for r in rows if r["split"] == "test"]
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in test:
        by[stratum(r, threshold)].append(r)
    sample = [r for s in sorted(by) for r in rng.sample(by[s], min(opts.per_stratum, len(by[s])))]
    donors = [it for r in test for it in r["llm_input"]["top_concepts"]]

    out = list(val)
    for r in sample:
        items = r["llm_input"]["top_concepts"]
        k = rng.randrange(len(items))
        variants = {"base": None, "repeat": None, "flip": k, "swap": k}
        if len(items) >= 2:
            variants["drop"] = k
        for name, idx in variants.items():
            v = copy.deepcopy(r)
            v["row_id"], v["variant"] = f"{r['isic_id']}#{name}", name
            its = v["llm_input"]["top_concepts"]
            if name == "flip":
                its[idx]["direction"] = "argues_against" if its[idx]["direction"] == "supports" else "supports"
                its[idx]["grouped_shap"] = -its[idx]["grouped_shap"]
            elif name == "swap":
                have = {it["group"] for it in its}
                choices = [d for d in donors if d["group"] not in have]
                its[idx] = copy.deepcopy(rng.choice(choices))
            elif name == "drop":
                its.pop(idx)
            v["perturbed_index"] = idx
            v["perturbed_feature"] = r["llm_input"]["top_concepts"][idx]["feature_id"] if idx is not None else None
            out.append(v)
    Path(opts.out).parent.mkdir(parents=True, exist_ok=True)
    with open(opts.out, "w", encoding="utf-8") as fh:
        for v in out:
            fh.write(json.dumps(v) + "\n")
    print(f"{len(sample)} cases x variants -> {len(out) - len(val)} test rows (+{len(val)} val rows) in {opts.out}")


def tokens(text):
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def jaccard(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def output(gen):
    """final_output is a dict (reasons-only prompts) or the model's raw JSON text (full-output prompts)."""
    out = gen["final_output"]
    if isinstance(out, str):
        out, ok = verifier.parse(out)
        out = out if ok and isinstance(out, dict) else {}
    return out


def claims(gen):
    return {r.get("feature_id"): r for r in (output(gen).get("reasons") or []) if isinstance(r, dict)}


def says_against(text):
    return bool(verifier._has(text, verifier.AGAINST_WORDS))


def score_case(base, var):
    """-> dict(sensitive: bool|None, stable_ok: [bool], stable_sim: [float], deterministic: bool|None)"""
    b, v = claims(base), claims(var)
    variant = var["variant"]
    pin = var["llm_input"]["evidence"]
    res = {"variant": variant, "sensitive": None, "stable_ok": [], "stable_sim": [], "deterministic": None}
    if variant == "repeat":
        res["deterministic"] = base["raw_output"] == var["raw_output"]
        return res
    base_ids = [e["feature_id"] for e in base["llm_input"]["evidence"]]
    var_ids = [e["feature_id"] for e in pin]
    if variant == "flip":
        changed = [f for f, eb, ev in zip(base_ids, base["llm_input"]["evidence"], pin) if eb["direction"] != ev["direction"]]
        f = changed[0] if changed else None
        want = next((e["direction"] for e in pin if e["feature_id"] == f), None)
        c = v.get(f)
        res["sensitive"] = bool(c) and c.get("direction") == want and says_against(c.get("text")) == (want == "argues_against")
        untouched = [x for x in var_ids if x != f]
    elif variant == "swap":
        new = [x for x in var_ids if x not in base_ids]
        old = [x for x in base_ids if x not in var_ids]
        res["sensitive"] = bool(new) and all(n in v for n in new) and not any(o in v for o in old)
        untouched = [x for x in var_ids if x in base_ids]
    else:  # drop
        gone = [x for x in base_ids if x not in var_ids]
        res["sensitive"] = not any(g in v for g in gone)
        untouched = var_ids
    dirs = {e["feature_id"]: e["direction"] for e in pin}
    for f in untouched:
        ok = f in v and f in b and v[f].get("direction") == dirs[f]
        sim = jaccard(b.get(f, {}).get("text"), v.get(f, {}).get("text")) if ok else 0.0
        res["stable_ok"].append(ok and sim >= STABLE_JACCARD)
        res["stable_sim"].append(sim)
    return res


def score(opts):
    gens = [json.loads(line) for line in open(opts.generations, encoding="utf-8")]
    by_case = defaultdict(dict)
    for g in gens:
        by_case[g["isic_id"]][g.get("variant", "base")] = g
    results = [score_case(c["base"], g) for c in by_case.values() if "base" in c
               for name, g in c.items() if name != "base"]
    report = {"generations": opts.generations, "n_cases": sum("base" in c for c in by_case.values())}
    for variant in ("flip", "swap", "drop"):
        rs = [r for r in results if r["variant"] == variant]
        stab = [x for r in rs for x in r["stable_ok"]]
        sims = [x for r in rs for x in r["stable_sim"]]
        report[variant] = {"n": len(rs), "sensitivity": float(np.mean([r["sensitive"] for r in rs])) if rs else None,
                           "stability": float(np.mean(stab)) if stab else None,
                           "mean_untouched_jaccard": float(np.mean(sims)) if sims else None}
    rep = [r["deterministic"] for r in results if r["variant"] == "repeat"]
    report["determinism"] = {"n": len(rep), "identical_rate": float(np.mean(rep)) if rep else None}
    out = Path(opts.generations).with_name("counterfactual_scores.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--inputs", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--per_stratum", type=int, default=75)
    s = sub.add_parser("score")
    s.add_argument("generations")
    opts = ap.parse_args()
    make(opts) if opts.cmd == "make" else score(opts)
