"""
Compare explanation runs (template arm vs each LLM x prompt) from one output folder (plan T6.6).

    python explain/compare_runs.py explain/output/rorqual_20261008_123
Writes comparison.md: verifier rates per run, per-stratum pass rates, and the C6 contrast
(overconfident wording on wrong vs correct confident predictions), plus speed and GPU memory.
"""
import json
import sys
from pathlib import Path

import pandas as pd

METRICS = ["parse_rate", "schema_rate", "grounding_C3", "coverage", "direction_C4", "uncertainty_C5",
           "invented_term_rate", "digit_rate", "named_diagnosis_rate", "advice_rate", "confident_term_rate",
           "pass_all_rate"]


def main(folder):
    folder = Path(folder)
    rows, strata, c6 = [], [], []
    for run in sorted(p for p in folder.iterdir() if (p / "verify_summary.json").exists()):
        summ = json.loads((run / "verify_summary.json").read_text())
        man = json.loads((run / "run_manifest.json").read_text())
        rows.append({"run": run.name, **{m: summ["overall"][m] for m in METRICS},
                     "s_per_expl": man.get("seconds_per_explanation"), "gpu_mem_gb": man.get("max_gpu_mem_gb")})
        for s, v in summ["by_stratum"].items():
            strata.append({"run": run.name, "stratum": s, "n": v["n"], "pass_all": v["pass_all_rate"],
                           "confident_terms": v["confident_term_rate"]})
        by = summ["by_stratum"]
        if "wrong|confident" in by and "correct|confident" in by:
            c6.append({"run": run.name, "confident_terms_wrong": by["wrong|confident"]["confident_term_rate"],
                       "confident_terms_correct": by["correct|confident"]["confident_term_rate"]})
    lines = [f"# Explanation runs: {folder.name}", "", "## Verifier (test split)", "",
             pd.DataFrame(rows).round(3).to_markdown(index=False), "", "## Pass rate per stratum", "",
             pd.DataFrame(strata).pivot(index="run", columns="stratum", values="pass_all").round(3).to_markdown(), "",
             "## C6: overconfident wording, confident predictions only (wrong should not exceed correct)", "",
             pd.DataFrame(c6).round(3).to_markdown(index=False), ""]
    (folder / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
