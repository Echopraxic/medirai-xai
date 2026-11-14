"""
End-to-end run of the explanation module with a real LLM (plan T8.1, D4): one Explainer.explain call per case, as the
product would call it, on test cases of the explained model. Reports how often the LLM text fails the verifier and is
replaced by the template (fallback), the verifier pass rate of what is returned, and seconds per case.

    python explain/run_module.py --trees trees/output_v1.1_kdv6_mel_s1 --features features/output/feature_table_v1.1.parquet \
        --phrase_bank concepts/output/phrase_bank_v3.csv --model Qwen/Qwen2.5-7B-Instruct --n 200 --out explain/output/module_run
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import llm_input as li  # noqa: E402
from llm_generate import VLLMBackend  # noqa: E402
from module import Explainer  # noqa: E402


def main(opts):
    backend = VLLMBackend(opts.model, True, li.REASONS_SCHEMA) if opts.model else None
    ex = Explainer.from_dirs(opts.trees, opts.features, backend=backend, prompt=opts.prompt, phrase_bank=opts.phrase_bank)
    table = pd.read_parquet(ROOT / opts.features).set_index("isic_id")
    tp = pd.read_csv(ROOT / opts.trees / "tree_predictions.csv")
    cases = tp[(tp.split == "test") & tp.mask_quality.notna()].sample(frac=1, random_state=0).head(opts.n)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "module_outputs.jsonl", "w", encoding="utf-8") as fh:
        for r in cases.itertuples():
            t0 = time.time()
            res = ex.explain(table.loc[r.isic_id].to_dict(),
                             {"p_malignant": r.resnet_p_malignant, "entropy_of_expected": r.entropy_of_expected},
                             r.mask_quality)
            sec = time.time() - t0
            rows.append({"fallback": bool(res["fallback_used"]), "pass": bool(res["verification"]["pass_all"]), "seconds": sec})
            fh.write(json.dumps({"isic_id": r.isic_id, "seconds": round(sec, 3), "text": res["text"],
                                 "fallback_used": res["fallback_used"], "provenance": res["provenance"]}) + "\n")
    d = pd.DataFrame(rows)
    summary = {"n": len(d), "model": opts.model or "template", "prompt": opts.prompt, "phrase_bank": opts.phrase_bank,
               "fallback_rate": float(d.fallback.mean()), "returned_pass_rate": float(d["pass"].mean()),
               "seconds_median": float(d.seconds.median()), "seconds_p95": float(d.seconds.quantile(0.95))}
    (out / "module_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--phrase_bank", default="concepts/output/phrase_bank_v3.csv")
    ap.add_argument("--model", default="", help="vLLM model id; empty = template backend (CPU smoke test)")
    ap.add_argument("--prompt", default="hybrid_v1")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=str(HERE / "output/module_run"))
    main(ap.parse_args())
