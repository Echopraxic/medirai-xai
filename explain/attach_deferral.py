"""Attach the per-image deferral flag from the uncertainty table (eval/uncertainty_table.py) to explanation inputs.

The table's `deferred` is high entropy-of-expected (val 75th percentile) OR disagreement between the deterministic
and variational heads. With it attached, llm_input / template / counterfactuals use it instead of the entropy
threshold alone (S1-4). The same table is what S1 gets, so "uncertain" means the same thing in both streams.

    python explain/attach_deferral.py --inputs trees/output_v1.0_detv2/explanation_inputs.jsonl \
        --table eval/output/<run>/uncertainty/uncertainty_table.csv --out trees/output_v1.0_detv2/explanation_inputs_deferral.jsonl
"""
import argparse
import json

import pandas as pd


def attach(rows, table):
    flags = table.set_index("isic_id")["deferred"].astype(bool).to_dict()
    reasons = table.set_index("isic_id")["defer_reason"].fillna("").to_dict()
    out = []
    for r in rows:
        if r["isic_id"] not in flags:
            raise KeyError(f"{r['isic_id']} missing from the uncertainty table")
        r["llm_input"]["uncertainty"]["deferred"] = flags[r["isic_id"]]
        r["llm_input"]["uncertainty"]["defer_reason"] = reasons[r["isic_id"]]
        out.append(r)
    return out


def main(opts):
    rows = [json.loads(line) for line in open(opts.inputs, encoding="utf-8")]
    out = attach(rows, pd.read_csv(opts.table))
    with open(opts.out, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    test = [r for r in out if r["split"] == "test"]
    print(f"{len(out)} inputs; test deferred {sum(r['llm_input']['uncertainty']['deferred'] for r in test)}/{len(test)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
