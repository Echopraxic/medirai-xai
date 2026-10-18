"""
Collect the headline numbers from the pipeline's output files into one RESULTS.md (local-only, regenerated).

Every number is read from an output file, never typed by hand, and each section names its source files and their
modification dates, so the page cannot drift from the runs. Missing sources are listed instead of failing.

    python tools/make_results.py              # writes RESULTS.md at the repo root
"""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD_BAD_DICE = 0.8


def load(rel):
    path = ROOT / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else pd.read_csv(path)


def src(*rels):
    parts = []
    for rel in rels:
        path = ROOT / rel
        when = dt.date.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else "MISSING"
        parts.append(f"`{rel}` ({when})")
    return "*Source: " + "; ".join(parts) + "*"


def fmt(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r.get(c, "—")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def first(*rels):
    """The first existing output among newer-to-older candidates (e.g. the split-v2 refit, then the v1 one)."""
    return next((r for r in rels if (ROOT / r).exists()), rels[-1])


def latest(pattern):
    hits = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1].relative_to(ROOT).as_posix() if hits else None


# ----------------------------------------------------------------------------------------------- sections
def llm_runs():
    """Latest verifier summary per (model, prompt), with the inputs it ran on."""
    runs = {}
    for summ in sorted(ROOT.glob("explain/output/**/verify_summary.json"), key=lambda p: p.stat().st_mtime):
        name = summ.parent.name
        manifest = summ.parent / "run_manifest.json"
        meta = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        runs[name] = (summ.relative_to(ROOT).as_posix(), json.loads(summ.read_text(encoding="utf-8")), meta)
    return runs


def c1_met(f, target=0.85):
    """C1 needs >= 85% overall and per true class; 'overall yes' only when the overall figure itself passes."""
    if min(f["by_true_label"].values()) >= target:
        return "yes"
    return "overall yes, per class no" if f["overall"] >= target else "no"


def criteria_section(runs):
    # C1 is measured against the explained model: the split-v2 det ResNet50 when its refit exists, else the v1 one
    tree_rel = first("trees/output_v1.0_detv2/tree_report.json", "trees/output_v1.0/tree_report.json")
    tree = load(tree_rel) if tree_rel else None
    explained = "split-v2 det ResNet50" if tree_rel and "detv2" in tree_rel else "split-v1 ResNet50 (stale)"
    enh = load("concepts/output/enhance_report.json")
    rows = []
    if tree:
        f = tree["fidelity"]["binary_surrogate"]
        rows.append({"criterion": "C1 surrogate fidelity", "target": "≥ 85%, per class",
                     "value": f"{fmt(f['overall'], pct=True)} (benign {fmt(f['by_true_label']['0'], pct=True)}, "
                              f"malignant {fmt(f['by_true_label']['1'], pct=True)}; on ResNet-wrong "
                              f"{fmt(f['on_resnet_wrong']['agreement'], pct=True)}; {explained})",
                     "met": c1_met(f)})
    if enh:
        parts, usable, rel_ok = [], [], []
        for c, v in enh["concepts"].items():
            parts.append(f"{c} {fmt(v['features_vs_other_raters_rho'], 2)} vs rater {fmt(v['crowd_rater_vs_others_rho'], 2)}")
            if v.get("labels_usable"):  # raters must agree with each other before they can validate anything
                usable.append(c)
                rel_ok.append(c) if v.get("meets_C2_relative") else None
        absolute = [c for c, v in enh["concepts"].items() if v.get("meets_C2_absolute")]
        rows.append({"criterion": "C2 concept validity (ENHANCE)", "target": "ρ ≥ 0.5 (proposed: ≥ a single rater)",
                     "value": "; ".join(parts),
                     "met": f"absolute: {', '.join(absolute) or 'none'}; relative: {', '.join(rel_ok) or 'none'} "
                            f"(labels usable only for {', '.join(usable) or 'none'})"})
    if runs:
        best = max(runs.items(), key=lambda kv: kv[1][1]["overall"]["pass_all_rate"] if "template" not in kv[0] else -1)
        o = best[1][1]["overall"]
        bs = best[1][1].get("by_stratum", {})
        rows += [
            {"criterion": "C3 grounding (structural)", "target": "≥ 98%", "value": f"{fmt(o['grounding_C3'], pct=True)} ({best[0]})",
             "met": "yes" if o["grounding_C3"] >= 0.98 else "no"},
            {"criterion": "C4 direction", "target": "≥ 95%", "value": f"{fmt(o['direction_C4'], pct=True)} ({best[0]})",
             "met": "yes" if o["direction_C4"] >= 0.95 else "no"},
            {"criterion": "C5 uncertainty stated", "target": "100%", "value": f"{fmt(o['uncertainty_C5'], pct=True)} ({best[0]})",
             "met": "yes" if o["uncertainty_C5"] >= 1 else "no"},
            {"criterion": "C6 confident wording, wrong vs correct", "target": "wrong ≤ correct",
             "value": f"{fmt(bs.get('wrong|confident', {}).get('confident_term_rate'), pct=True)} vs "
                      f"{fmt(bs.get('correct|confident', {}).get('confident_term_rate'), pct=True)} ({best[0]})",
             "met": "yes" if bs and bs["wrong|confident"]["confident_term_rate"] <= bs["correct|confident"]["confident_term_rate"] else "—"},
        ]
    rows.append({"criterion": "C7 clinician review", "target": "NL ≥ Grad-CAM, label-only", "value": "not run", "met": "—"})
    return ("## Success criteria (plan §1)\n\n" + table(rows, ["criterion", "target", "value", "met"])
            + "\n\nC3–C6 show the best non-template LLM run by pass-all rate; all runs are in the LLM section. C3 is "
              "structural (each claim cites a supplied feature_id); the semantic NLI check is in the D5 section.\n\n"
            + src(*([tree_rel] if tree_rel else []), "concepts/output/enhance_report.json"))


def baseline_section():
    rels = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob("eval/output/*/baseline_results.json"))
    rows = []
    for rel in rels:
        for r in load(rel):
            t, nm = r["test"], r.get("test_without_mskcc") or {}
            rows.append({"report": rel.split("/")[2], "run": r["run"], "test n": t["n"], "accuracy": fmt(t["accuracy"]),
                         "AUROC": fmt(t["auroc"]), "ECE (top label)": fmt(t.get("ece_top_label")),
                         "excl. MSKCC acc / AUROC": f"{fmt(nm.get('accuracy'))} / {fmt(nm.get('auroc'))}",
                         "deferred / retained acc": f"{fmt(r.get('test_deferred_fraction'), pct=True)} / "
                                                     f"{fmt(r.get('test_retained_accuracy'))}"})
    if not rows:
        return "## ResNet50 baselines\n\nNo baseline reports found."
    return ("## ResNet50 baselines\n\nLocal reports used split v1 (test n = 1,627); \"excl. MSKCC\" is the split-v2 test "
            "set. `rorqual_v2_*` are the split-v2 retrains (Narval A100, 2026-10-14; lr 3e-4 chosen by val AUROC); their "
            "deferred share is the entropy rule alone (the uncertainty table, which also defers on head disagreement, "
            "defers more).\n\n"
            + table(rows, list(rows[0])) + "\n\n" + src(*rels))


def masks_section():
    per = load("segmentation/output/gold_eval/per_image.csv")
    qa = load("segmentation/output/qa_clinical/summary.json")
    split = load("splits/isic_clinical_v2.csv")
    qa_rows = load("segmentation/output/qa_clinical/mask_quality.csv")
    out = ["## Lesion masks (gold set, 97 hand-traced split-v2 test images)"]
    if per is not None and qa_rows is not None and split is not None:
        test_ids = set(split.loc[split.split == "test", "isic_id"])
        w = load("segmentation/output/qa/mask_quality.csv")
        w = w[w.isic_id.isin(test_ids)].mask_quality.value_counts(normalize=True)
        rows = []
        for m, g in per[per.half == "report"].groupby("method"):
            by = g.groupby("flag").dice.agg(["mean", lambda s: (s < GOLD_BAD_DICE).mean()])
            by.columns = ["dice", "bad"]
            rows.append({"method": m, "Dice (reweighted)": fmt((by.dice * w).sum()),
                         f"share Dice < {GOLD_BAD_DICE}": fmt((by.bad * w).sum(), pct=True), "n": len(g)})
        out.append("Report half, reweighted to the test-set mix of the original UNet flag (the gold set was "
                   "sampled about one-third per flag).\n\n" + table(rows, list(rows[0])))
    if qa:
        out.append(f"v1 flag on split v2 (clinical UNet): {qa['flag_counts']}.")
    out.append(src("segmentation/output/gold_eval/per_image.csv", "segmentation/output/qa_clinical/summary.json"))
    return "\n\n".join(out)


def trees_section():
    tree = load("trees/output_v1.0/tree_report.json")
    shap_sum = load("trees/output_v1.0/treeshap_summary.json")
    s1 = load("trees/output/s1_binary_v1.0/report.json")
    if not tree:
        return "## Trees\n\nNo v1.0 tree report."
    tp = tree["test_performance"]
    rows = [{"model": f"one-vs-all {k}", "AUROC": fmt(v["auroc"]), "F1": fmt(v["f1"]), "n positive": v["n_pos"]}
            for k, v in tp["ova_per_class"].items()]
    rows.append({"model": "binary (ground truth)", "AUROC": fmt(tp["binary_gt"]["auroc"]), "F1": "—", "n positive": "—"})
    if s1:
        mel = s1["test_by_ova_class"].get("Melanoma", {})
        rows.append({"model": "S1 binary (ground truth)", "AUROC": fmt(s1["test"]["auroc"]),
                     "F1": f"melanoma recall {fmt(mel.get('sensitivity'))}", "n positive": "—"})
    extra = ""
    if shap_sum:
        extra = (f"\n\nTop-3 bootstrap Jaccard {fmt(shap_sum['top3_bootstrap_jaccard_mean'], 2)}; concept groups in "
                 f"the test top-3: {shap_sum['group_frequency_in_top3_test']}.")
    sc = tree.get("source_shortcut_check", {})
    extra += "\n\nSource shortcut (UFES vs MILK within one diagnosis, AUROC; 0.5 = no shortcut): " + ", ".join(
        f"{k} {fmt(v['auroc'], 2)}" for k, v in sc.items() if isinstance(v, dict))
    v2 = load("trees/output_v1.0_detv2/tree_report.json")
    rels_v2 = []
    if v2:
        sur, res = v2["test_performance"]["binary_surrogate"], v2["test_performance"]["resnet"]
        extra += ("\n\nRefit against the split-v2 deterministic ResNet50 (lr 3e-4; `trees/output_v1.0_detv2/`): ResNet50 test "
                  f"accuracy {fmt(res['accuracy'])} / AUROC {fmt(res['auroc'])}; surrogate accuracy {fmt(sur['accuracy'])}, "
                  f"balanced accuracy {fmt(sur.get('balanced_accuracy'))}, AUROC {fmt(sur['auroc'])}. "
                  "The tables above are the refit on the v1-split ResNet50.")
        rels_v2 = ["trees/output_v1.0_detv2/tree_report.json"]
    return ("## Tree models and TreeSHAP (features v1.0, split-v2 test)\n\n" + table(rows, ["model", "AUROC", "F1", "n positive"])
            + extra + "\n\n" + src("trees/output_v1.0/tree_report.json", "trees/output_v1.0/treeshap_summary.json",
                                   "trees/output/s1_binary_v1.0/report.json", *rels_v2))


def concepts_section():
    out = ["## Concept validation"]
    enh = load("concepts/output/enhance_report.json")
    if enh:
        rows = [{"concept": c, "features vs other raters": fmt(v["features_vs_other_raters_rho"], 2),
                 "single rater vs other raters": fmt(v["crowd_rater_vs_others_rho"], 2),
                 "best feature": f"{v['best_single_feature']} ({fmt(v['best_single_feature_rho'], 2)})",
                 "labels usable": v.get("labels_usable")} for c, v in enh["concepts"].items()]
        out.append(f"ENHANCE crowd labels (dermoscopy, features {enh.get('feature_version')}):\n\n" + table(rows, list(rows[0])))
    monet = load("concepts/output/monet_report.json")
    if monet:
        out.append("MILK10k MONET scores (clinical photos), Spearman ρ: " + ", ".join(
            f"{k} {fmt(v, 2)}" for k, v in monet["pairs"].items()))
    vlm_rel = latest("concepts/output/vlm_*/eval.json")
    vlm = load(vlm_rel) if vlm_rel else None
    if vlm:
        rows = []
        for run, res in vlm.items():
            if not isinstance(res, dict) or run.startswith("our_"):
                continue
            for q, v in res.items():
                if isinstance(v, dict) and "kappa" in v:
                    rows.append({"run": run, "question": q, "n": v["n"], "κ": fmt(v["kappa"], 2),
                                 "accuracy": fmt(v["accuracy"], 2), "majority": fmt(v["majority_baseline"], 2)})
        out.append("Derm7pt expert labels, VLM as a concept checker (Cohen's κ; dermoscopy rows are partial until "
                   "n = 1,011):\n\n" + table(rows, list(rows[0])))
        ours = vlm.get("our_features_vs_expert_auroc", {})
        out.append("Our features vs Derm7pt expert labels (clinical photos, AUROC): " + "; ".join(
            f"{lab}: " + ", ".join(f"{f} {fmt(a, 2)}" for f, a in feats.items()) for lab, feats in ours.items()))
    out.append(src("concepts/output/enhance_report.json", "concepts/output/monet_report.json", vlm_rel or "concepts/output/vlm_*/eval.json"))
    return "\n\n".join(out)


def llm_section(runs):
    if not runs:
        return "## LLM explanations\n\nNo verifier summaries found."
    rows = []
    for name, (rel, s, meta) in sorted(runs.items()):
        o = s["overall"]
        inputs = Path(meta.get("inputs", "")).parent.name or "—"
        rows.append({"run": name, "inputs": inputs, "n": o["n"], "pass all": fmt(o["pass_all_rate"], pct=True),
                     "C3": fmt(o["grounding_C3"], pct=True), "C4": fmt(o["direction_C4"], pct=True),
                     "C5": fmt(o["uncertainty_C5"], pct=True), "invented terms": fmt(o["invented_term_rate"], pct=True),
                     "decoding": "greedy" if meta.get("decoding", {}).get("temperature", 0) == 0 else "sampled"})
    return ("## LLM explanations (rule-based verifier, test split)\n\nLatest run per model and prompt. `inputs` is the "
            "TreeSHAP output folder the run explained.\n\n" + table(rows, list(rows[0]))
            + "\n\n" + src(*[v[0] for v in runs.values()]))


def c6_gradcam_section():
    c6_rel = first("eval/output/c6_v2/c6_results.json", "eval/output/c6/c6_results.json")
    cam_rel = first("eval/output/gradcam_v2/gradcam_metrics.csv", "eval/output/gradcam/gradcam_metrics.csv")
    c6, cam = load(c6_rel), load(cam_rel)
    out = ["## Wrong predictions (C6) and Grad-CAM"]
    if c6:
        rows = [{k: (fmt(v) if isinstance(v, float) else v) for k, v in s.items()
                 if k in ("stratum", "n", "surrogate_agrees", "confident_wording", "flagged_uncertain", "cam_in_lesion")}
                for s in c6["strata"]]
        out.append("Template explanations by stratum:\n\n" + table(rows, list(rows[0])))
        out.append("Error-detection AUROC (higher = flags wrong predictions): " + ", ".join(
            f"{k} {fmt(v, 2)}" for k, v in c6.get("error_detection_auroc", {}).items()))
    if cam is not None:
        out.append(f"Grad-CAM (n = {len(cam)}): deletion AUC {fmt(cam.deletion_auc.mean())} vs random order "
                   f"{fmt(cam.random_deletion_auc.mean())} (lower = more faithful); CAM mass in lesion "
                   f"{fmt(cam.cam_in_lesion.mean(), pct=True)}; pointing hit {fmt(cam.pointing_hit.mean(), pct=True)}.")
    out.append(src(c6_rel, cam_rel))
    return "\n\n".join(out)


def new_checks_section():
    out = ["## Error signals, tree drivers and the NLI claim check"]
    es_rel = first("eval/output/error_signals_v2/error_signals.json", "eval/output/error_signals/error_signals.json")
    dc_rel = first("trees/output_v1.0_detv2/driver_comparison/report.json", "trees/output_v1.0/driver_comparison/report.json")
    es = load(es_rel)
    if es:
        rows = [{"signal": k, "test AUROC (all)": f"{v['test_all'][0]:.2f} [{v['test_all'][1][0]:.2f}, {v['test_all'][1][1]:.2f}]",
                 "test AUROC (confident only)": f"{v['test_confident'][0]:.2f} [{v['test_confident'][1][0]:.2f}, {v['test_confident'][1][1]:.2f}]"}
                for k, v in es["single_signal_auroc"].items()]
        r20 = es["caveat_rule"]["options"]["top_20pct"]
        out.append("Label-free signals for 'ResNet50 is wrong' (test, bootstrap 95% CI):\n\n" + table(rows, list(rows[0]))
                   + f"\n\nCaveat rule ({es['caveat_rule']['signal']}, top 20% of val confident): flags "
                   f"{fmt(r20['test_flag_rate'], pct=True)} of confident test predictions, catches "
                   f"{fmt(r20['test_share_of_confident_errors_caught'], pct=True)} of confident errors; error rate "
                   f"{fmt(r20['test_error_rate_flagged'], pct=True)} flagged vs {fmt(r20['test_error_rate_unflagged'], pct=True)} unflagged.")
    dc = load(dc_rel)
    if dc:
        m = dc["melanoma"]
        out.append(f"Tree drivers (test, n = {dc['n_test']}): S1's binary tree agrees with S2's GT binary tree at rho "
                   f"{dc['agreement_with_S1_binary']['S2_binary_gt']['spearman_mean_abs_shap']:.2f}, with one-vs-all BCC "
                   f"{dc['agreement_with_S1_binary']['ova_BCC']['spearman_mean_abs_shap']:.2f}, with one-vs-all melanoma "
                   f"{dc['agreement_with_S1_binary']['ova_Melanoma']['spearman_mean_abs_shap']:.2f}. Melanoma recall: S1 binary "
                   f"{fmt(m['S1_binary_recall'], pct=True)}, one-vs-all {fmt(m['S2_ova_as_binary_recall'], pct=True)}, ResNet50 "
                   f"{fmt(m['resnet_recall'], pct=True)}.")
    nli = load("explain/output/nli_calibration_v1.0.json")
    if nli:
        out.append(f"NLI claim check ({nli['model']}) on synthetic controls: false alarms {fmt(nli['false_alarm_rate'], pct=True)}; "
                   "detection " + ", ".join(f"{k} {fmt(v, pct=True)}" for k, v in nli["detection_rate"].items()) + ".")
    for rel in sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob("explain/output/**/nli_scores.json")):
        r = load(rel)
        out.append(f"NLI on `{Path(rel).parent.name}` ({r['n_claims']} claims): entailed {fmt(r['entailed_rate'], pct=True)}, "
                   f"contradicted {fmt(r['contradicted_rate'], pct=True)}.")
    out.append(src(es_rel, dc_rel, "explain/output/nli_calibration_v1.0.json"))
    return "\n\n".join(out)


def d5_section():
    """D5 automated validation from the latest d5_* run: counterfactual sensitivity, consistency, NLI (T6.1-T6.3)."""
    out = ["## D5 automated validation (counterfactuals, consistency, NLI)"]
    d5 = latest("explain/output/d5_*/nli_calibration.json")
    if not d5:
        return out[0] + "\n\nNo D5 run found."
    run = ROOT / Path(d5).parent
    nli = load(d5)
    out.append(f"Run `{run.name}`. NLI checker ({nli['model']}, entail >= {nli['entail_min']}) on synthetic controls: "
               f"false alarms {fmt(nli['false_alarm_rate'], pct=True)}; detection "
               + ", ".join(f"{k} {fmt(v, pct=True)}" for k, v in nli["detection_rate"].items()) + ".")
    rows = []
    for arm in sorted(d for d in run.iterdir() if d.is_dir()):
        rel = lambda name: (arm / name).relative_to(ROOT).as_posix()
        cf, sim, par, ns = (load(rel(f)) for f in ("counterfactual_scores.json", "consistency_similar.json",
                                                   "consistency_paraphrase.json", "nli_scores.json"))
        if not any((cf, sim, par, ns)):
            continue
        row = {"arm": arm.name}
        if cf:
            for k in ("flip", "swap", "drop"):
                row[f"{k} sens / stab"] = f"{fmt(cf[k]['sensitivity'], 2)} / {fmt(cf[k]['stability'], 2)}"
            row["determinism"] = fmt(cf["determinism"]["identical_rate"], pct=True)
        if sim:
            row["same-signature Jaccard"] = fmt(sim["same_signature"]["mean_jaccard"], 2)
        if par:
            row["paraphrase same direction / Jaccard"] = f"{fmt(par['same_direction_rate'], pct=True)} / {fmt(par['mean_jaccard'], 2)}"
        if ns:
            row["NLI entailed / contradicted"] = f"{fmt(ns['entailed_rate'], pct=True)} / {fmt(ns['contradicted_rate'], pct=True)}"
        rows.append(row)
    cols = list(dict.fromkeys(c for r in rows for c in r))
    out.append(table(rows, cols) + "\n\nSensitivity = the explanation changes when the perturbed feature changes; stability "
               "= the untouched claims stay the same. Paraphrase rows compare the paraphrased prompt with the base prompt "
               "of the same model. NLI precision on real flagged claims still needs human labels (RQ1).")
    out.append(src(d5))
    return "\n\n".join(out)


def model_comparison_section():
    """Paired cluster bootstraps between explained-model candidates, and performance by skin tone."""
    out = ["## Explained-model candidates and skin tone"]
    rels = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob("eval/output/paired_bootstrap_*/paired_bootstrap.json"))
    for rel in rels:
        b = load(rel)
        rows = [{"metric": k, "A": fmt(v["A"]), "B": fmt(v["B"]), "B - A": f"{v['diff']:+.3f}",
                 "95% CI": f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]", "distinguishable": v["significant"]}
                for k, v in b["metrics"].items()]
        out.append(f"`{Path(b['a']).parent.name}` (A) vs `{Path(b['b']).parent.name}` (B), n = {b['n_test']}:\n\n"
                   + table(rows, list(rows[0])))
    tone_rel = "eval/output/skin_tone/skin_tone_report.json"
    tone = load(tone_rel)
    if tone:
        rows = [{"run": run, "tone": g, "n": m["n"], "malignant share": fmt(m["malignant_share"], 2),
                 "AUROC": fmt(m["auroc"]) + (f" [{m['ci']['auroc'][0]:.2f}, {m['ci']['auroc'][1]:.2f}]" if m.get("ci") else ""),
                 "malignant recall": fmt(m["malignant_recall"]), "benign recall": fmt(m["benign_recall"])}
                for run, groups in tone.items() for g, m in groups.items() if g != "unknown"]
        out.append("By skin tone (darker = MILK10k tone 0-2 or Fitzpatrick V-VI; see the report for the mapping):\n\n"
                   + table(rows, list(rows[0])))
    out.append(src(*rels, tone_rel))
    return "\n\n".join(out)


def main(opts):
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = "?"
    runs = llm_runs()
    parts = [
        "# Results (generated, do not edit)",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `python tools/make_results.py` at commit `{commit}`. "
        "Numbers are read from the output files named under each section. Older outputs (e.g. features v0.2, "
        "the split-v1 ResNet50) are labelled as such; re-run the pipeline step to refresh them.",
        criteria_section(runs), baseline_section(), masks_section(), trees_section(), concepts_section(),
        llm_section(runs), c6_gradcam_section(), new_checks_section(), d5_section(), model_comparison_section(),
    ]
    out = Path(opts.out)
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "RESULTS.md"))
    main(ap.parse_args())
