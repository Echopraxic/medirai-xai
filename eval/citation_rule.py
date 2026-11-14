"""
C2 citation rule R1 vs the current rule (v1.1) and variant R2, split-v2 test, final model (plan 2026-11-04, rule fixed
before this was run). Inputs: explanation_inputs{_v1,_r1,_r2}.jsonl and shap_values.parquet in the trees folder.

Measures per rule: status shares of cited reasons; cases whose cited group set differs from v1.1; cited-feature share
(cited feature's contribution / the group's largest contribution in the group's direction; median and share < 0.5);
|grouped SHAP| of the cited groups as a share of the original top-3 groups' |grouped SHAP|.
Adopt R1 if: failed share 0; cited group set unchanged in >= 95% of test cases; median cited-feature share >= 0.5.
(The fourth check, template verifier pass rate not lower, is run with explain/llm_generate.py --backend template.)

    python eval/citation_rule.py --trees trees/output_v1.1_kdv8f_mel_s3   -> <trees>/citation_rule.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(trees, suffix):
    rows = [json.loads(line) for line in open(trees / f"explanation_inputs{suffix}.jsonl", encoding="utf-8")]
    return {r["isic_id"]: r for r in rows if r["split"] == "test"}


def measure(cases, base, sv, status):
    st, groups_changed, shares, mass = [], 0, [], []
    for iid, c in cases.items():
        reasons = c["llm_input"]["top_concepts"]
        b = base[iid]["llm_input"]["top_concepts"]
        st += [status.get(r["feature_id"], "untested") for r in reasons]
        groups_changed += {r["group"] for r in reasons} != {r["group"] for r in b}
        sign = 1.0 if c["llm_input"]["prediction"] == "malignant" else -1.0
        row = sv.loc[iid]
        for r in reasons:
            members = [f for f in sv.columns if f[0] == r["group"]]
            contrib = row[members] * sign * (1 if r["direction"] == "supports" else -1)
            shares.append(float(contrib[r["feature_id"]] / contrib.max()) if contrib.max() > 0 else np.nan)
        orig = sum(abs(x["grouped_shap"]) for x in b)
        mass.append(sum(abs(x["grouped_shap"]) for x in reasons) / orig if orig else 1.0)
    st = pd.Series(st)
    shares = pd.Series(shares).dropna()
    return {"n_cases": len(cases), "n_reasons": int(len(st)),
            "status_share": {k: round(float(v), 4) for k, v in st.value_counts(normalize=True).items()},
            "validated_or_partial": round(float(st.isin(["validated", "partial"]).mean()), 4),
            "cases_with_validated_reason": round(float(np.mean([any(status.get(r["feature_id"]) == "validated"
                                                                    for r in c["llm_input"]["top_concepts"])
                                                                for c in cases.values()])), 4),
            "group_set_unchanged": round(1 - groups_changed / len(cases), 4),
            "cited_feature_share_median": round(float(shares.median()), 4),
            "cited_feature_share_below_0.5": round(float((shares < 0.5).mean()), 4),
            "cited_group_shap_mass_vs_original": round(float(np.mean(mass)), 4),
            "cases_with_fewer_than_3_reasons": int(sum(len(c["llm_input"]["top_concepts"]) < 3 for c in cases.values()))}


def main(opts):
    trees = Path(opts.trees)
    status = dict(pd.read_csv(ROOT / opts.phrase_bank)[["feature_id", "status"]].itertuples(index=False))
    sv = pd.read_parquet(trees / "shap_values.parquet").set_index("isic_id")
    base = load(trees, "_v1")  # the v1.1 citations, kept as explanation_inputs_v1.jsonl when R1 was adopted
    res = {name: measure(load(trees, suf), base, sv, status) for name, suf in (("v1.1", "_v1"), ("R1", "_r1"), ("R2", "_r2"))}
    r1 = res["R1"]
    checks = {"failed_share_0": r1["status_share"].get("failed", 0.0) == 0.0,
              "group_set_unchanged>=0.95": r1["group_set_unchanged"] >= 0.95,
              "cited_feature_share_median>=0.5": r1["cited_feature_share_median"] >= 0.5}
    res["adoption R1 (3 of 4 checks; verifier pass rate checked separately)"] = {"checks": checks, "pass": all(checks.values())}
    (trees / "citation_rule.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", required=True)
    ap.add_argument("--phrase_bank", default="concepts/output/phrase_bank_v3.csv")
    main(ap.parse_args())
