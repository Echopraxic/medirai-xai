"""
Net-benefit review of the four explanation features (decision 2026-10-21: implement all, then review each and all
together). Selection data only: split v2 val (full explanations) and MIDAS dev_val phone photos (scope flag and
similar-case panel; no handcrafted features there). Criteria were fixed before looking at results:

  faithfulness filter  + cited directions that match how the feature moves the ResNet's logit (linear readout)
                       - cases left with fewer than three reasons; weaker reasons cited
  concept share        + the band separates ResNet errors / surrogate disagreement; - useless if one band dominates
  scope flag           + wrong predictions that gain a caution; - correct predictions that get a warning
  similar-case panel   + wrong predictions whose neighbours contradict them; - wrong predictions whose neighbours
                       agree (reinforcement), correct predictions whose neighbours contradict them (confusion)
  all together         false reassurance: confident wrong predictions with no caution and no contradicting panel;
                       template verifier pass rate; explanation length

    python interpret/review_explanation_features.py --run rorqual_v2_lr3e-4_20261014/resnet50_det/run \
        --trees trees/output_v1.0_detv2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain"))
sys.path.insert(0, str(ROOT / "interpret"))
import llm_input as li  # noqa: E402
import verifier  # noqa: E402
from explanation_features import ExplanationFeatures  # noqa: E402
from template import render_llm_input  # noqa: E402


def rows_of(path, split="val"):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return {r["isic_id"]: r for r in rows if r["split"] == split}, li.entropy_threshold_from_val(rows)


def emb_table(*files):
    parts = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        key = "isic_id" if "isic_id" in z.files else "image_id"
        parts.append(pd.DataFrame(z["emb"].astype(np.float32), index=z[key]))
    return pd.concat(parts)


def caution(inp):
    """Any statement that tells the reader not to take the prediction at face value."""
    return bool(inp["uncertain"] or not inp["surrogate_agrees"] or inp.get("independent_check_weak")
                or inp.get("in_scope") is False)


def main(opts):
    run_tag = opts.run.split("/")[0]
    fx = ExplanationFeatures(opts.run)
    trees = ROOT / opts.trees
    base, thr = rows_of(trees / "explanation_inputs.jsonl")
    filt, _ = rows_of(trees / "explanation_inputs_filtered.jsonl")
    labelled, _ = rows_of(trees / "explanation_inputs_labelled.jsonl")
    feats = pd.read_parquet(ROOT / "features/output/feature_table_v1.0.parquet").set_index("isic_id")
    monet = pd.read_parquet(ROOT / "concepts/output/monet_concepts_v1.parquet").set_index("isic_id")
    run = ROOT / "uncertaintyNet-main/output" / opts.run
    pred = pd.read_csv(run / "predictions.csv").set_index("isic_id")
    dino = emb_table(ROOT / "classifier/output/embeddings/dinov2.npz", ROOT / "classifier/output/embeddings/dinov2__midas.npz")
    resnet = emb_table(run / "embeddings.npz",
                       ROOT / "eval/output/interpretability" / run_tag / "midas_dev_val_embeddings.npz")
    res = {}

    # ---------------- per-case table on internal val
    recs = []
    for i, r in base.items():
        p = float(pred.loc[i, "p_malignant"])
        y = int(r["label"])
        values = {**feats.loc[i].to_dict(), **monet.loc[i].to_dict()}
        share, band = fx.concept_share(values, p)
        ins, _ = fx.in_scope(dino.loc[i].to_numpy())
        sim = fx.similar_cases(resnet.loc[i].to_numpy(), space="resnet")
        effect = fx.feature_effect(values)
        sgn = 1 if p >= 0.5 else -1
        def faithful(reasons):
            ok = [(np.sign(effect[c["feature_id"]] * sgn) > 0) == (c["direction"] == "supports") for c in reasons
                  if c["feature_id"] in effect]
            return ok
        recs.append({"id": i, "label": y, "pred": int(p >= 0.5), "share": share, "band": band, "in_scope": ins,
                     "nb_contradict": np.mean([c["label"] != int(p >= 0.5) for c in sim]) > 0.5,
                     "base": r, "filt": filt[i], "lab": labelled[i], "faith_base": faithful(r["llm_input"]["top_concepts"]),
                     "faith_filt": faithful(filt[i]["llm_input"]["top_concepts"])})
    d = pd.DataFrame(recs)
    d["wrong"] = d.label != d.pred
    not_cited = fx.not_cited

    # ---------------- faithfulness filter
    def cited_unencoded(col):
        return float(np.mean([any(c["feature_id"] in not_cited for c in r["llm_input"]["top_concepts"]) for r in d[col]]))
    res["faithfulness_filter"] = {
        "not_cited_features": sorted(not_cited),
        "cases_citing_unencoded_feature": {"base": cited_unencoded("base"), "filtered": cited_unencoded("filt")},
        "direction_agrees_with_resnet": {"base": float(np.mean(np.concatenate(d.faith_base.map(np.array).to_list()))),
                                         "filtered": float(np.mean(np.concatenate(d.faith_filt.map(np.array).to_list())))},
        "cases_with_fewer_than_3_reasons": float(np.mean([len(r["llm_input"]["top_concepts"]) < 3 for r in d.filt])),
        "share_of_strong_reasons": {
            k: float(np.mean([c["strength"] == "strong" for r in d[k] for c in r["llm_input"]["top_concepts"]]))
            for k in ("base", "filt")},
    }

    # ---------------- concept share
    sur_dis = d.base.map(lambda r: not r["llm_input"]["surrogate_agrees"])
    res["concept_share"] = {
        "band_distribution": d.band.value_counts(normalize=True).round(3).to_dict(),
        "resnet_error_rate_by_band": d.groupby("band").wrong.mean().round(3).to_dict(),
        "surrogate_disagreement_by_band": sur_dis.groupby(d.band).mean().round(3).to_dict(),
        "auroc_low_share_for_resnet_error": float(roc_auc_score(d.wrong, 1 - d.share)),
        "auroc_low_share_for_surrogate_disagreement": float(roc_auc_score(sur_dis, 1 - d.share)),
    }

    # ---------------- MIDAS dev_val (scope + panel only)
    m = np.load(ROOT / "eval/output/interpretability" / run_tag / "midas_dev_val_embeddings.npz", allow_pickle=True)
    mid = pd.DataFrame({"id": m["image_id"], "label": m["label"].astype(int), "p": m["p"]})
    mid["pred"] = (mid.p >= 0.5).astype(int)
    mid["wrong"] = mid.label != mid.pred
    mid["uncertain"] = -(np.clip(mid.p, 1e-7, 1) * np.log(np.clip(mid.p, 1e-7, 1))
                         + np.clip(1 - mid.p, 1e-7, 1) * np.log(np.clip(1 - mid.p, 1e-7, 1))) > thr
    mid["in_scope"] = [fx.in_scope(dino.loc[i].to_numpy())[0] for i in mid.id]
    mid["nb_contradict"] = [np.mean([c["label"] != pr for c in fx.similar_cases(resnet.loc[i].to_numpy(), space="resnet")]) > 0.5
                            for i, pr in zip(mid.id, mid.pred)]

    # ---------------- scope flag
    def scope_stats(frame, base_caution):
        with_scope = base_caution | ~frame.in_scope
        w, c = frame.wrong.to_numpy(), ~frame.wrong.to_numpy()
        return {"flagged_out_of_scope": float((~frame.in_scope).mean()),
                "wrong_with_caution": {"without_scope": float(base_caution[w].mean()), "with_scope": float(with_scope[w].mean())},
                "correct_with_caution": {"without_scope": float(base_caution[c].mean()), "with_scope": float(with_scope[c].mean())}}
    d_base_caution = d.base.map(lambda r: caution(li.build(r["llm_input"], r["mask_quality"], thr))).to_numpy()
    res["scope_flag"] = {"internal_val": scope_stats(d, d_base_caution),
                         "midas_dev_val": scope_stats(mid, mid.uncertain.to_numpy())}

    # ---------------- similar-case panel
    def panel_stats(frame, gated):
        shown = frame.in_scope.to_numpy() if gated else np.ones(len(frame), bool)
        w, c, nb = frame.wrong.to_numpy(), ~frame.wrong.to_numpy(), frame.nb_contradict.to_numpy()
        return {"shown": float(shown.mean()),
                "wrong_helpful_contradiction": float((shown & w & nb).sum() / max(w.sum(), 1)),
                "wrong_reinforced": float((shown & w & ~nb).sum() / max(w.sum(), 1)),
                "correct_confusing_contradiction": float((shown & c & nb).sum() / max(c.sum(), 1))}
    res["similar_case_panel"] = {s: {"ungated": panel_stats(f, False), "scope_gated": panel_stats(f, True)}
                                 for s, f in (("internal_val", d), ("midas_dev_val", mid))}

    # ---------------- all together (internal val, full explanations, template backend)
    def explanation(row, all_on):
        """all_on = the configuration decided after this review (2026-10-21): unencoded features labelled (not
        dropped), scope flag, concept-share sentence only in the 'little' band, no similar-case panel."""
        r = row.lab if all_on else row.base
        payload = dict(r["llm_input"])
        if all_on:
            payload["in_scope"] = bool(row.in_scope)
            if row.band == "little":
                payload["named_concept_share"] = row.band
        inp = li.build(payload, r["mask_quality"], thr)
        return inp, render_llm_input(inp)
    out = {}
    for name, on in (("current", False), ("decided_configuration", True)):
        built = [explanation(row, on) for row in d.itertuples()]
        cau = np.array([caution(i) for i, _ in built])
        panel_contra = np.zeros(len(d), bool)   # no panel in either configuration (decision 2026-10-21)
        conf_wrong = (d.wrong & ~np.array([i["uncertain"] for i, _ in built])).to_numpy()
        out[name] = {
            "false_reassurance_confident_wrong": float((conf_wrong & ~cau & ~panel_contra).sum() / max(conf_wrong.sum(), 1)),
            "verifier_pass_rate": float(np.mean([verifier.verify(i, o)["pass_all"] for i, o in built])),
            "cases_with_a_labelled_reason": float(np.mean([any(not e.get("model_encoded", True) for e in i["evidence"])
                                                           for i, _ in built])),
            "reasons_labelled": float(np.mean([not e.get("model_encoded", True) for i, _ in built for e in i["evidence"]])),
            "mean_words": float(np.mean([len(" ".join([o["summary"], o["uncertainty_note"]]
                                                      + [x["text"] for x in o["reasons"]]).split()) for _, o in built])),
        }
    res["all_together"] = out
    dest = ROOT / "eval/output/interpretability" / run_tag / "feature_review.json"
    dest.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rorqual_v2_lr3e-4_20261014/resnet50_det/run")
    ap.add_argument("--trees", default="trees/output_v1.0_detv2")
    main(ap.parse_args())
