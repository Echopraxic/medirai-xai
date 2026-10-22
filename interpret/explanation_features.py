"""
Explanation features from the interpretability prototypes (local classifier plan §10; decision 2026-10-21: implement
all four, then review each and all together for net benefit). Fitted on split v2 TRAIN only, per explained model.

  faithfulness filter  features the ResNet's embedding does not encode (probe R² < NOT_ENCODED_R2 on internal val,
                       interpret/prototypes.py concepts) are never cited as a reason (their SHAP still counts)
  concept share        ridge from named concepts (handcrafted + MONET, z-scored) to the ResNet's logit; per case,
                       share = |concept part| / (|concept part| + |residual|) -> "most" / "about half" / "little"
  scope flag           Mahalanobis distance in DINOv2-L space (PCA 128, Ledoit-Wolf), cut = internal val p95:
                       out-of-scope images get a warning (a scope warning, not an error filter)
  similar cases        k nearest training cases (cosine), in-scope images only. Review 2026-10-21: in the ResNet's own
                       space they reinforce 80% of wrong predictions (confirmation bias), so the product never shows
                       them; the DINOv2-space version (independent view) is kept for the clinician study arm only.
Decisions after the review (2026-10-21): unencoded features are LABELLED, not dropped; the concept-share sentence
is shown only in the "little" band.

    python interpret/explanation_features.py fit --run rorqual_v2_lr3e-4_20261014/resnet50_det/run
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
NOT_ENCODED_R2 = 0.3
SHARE_BANDS = [(2 / 3, "most"), (1 / 3, "about half"), (0.0, "little")]


def artifact_dir(run):
    return ROOT / "eval/output/interpretability" / run.split("/")[0] / "artifacts"


# ------------------------------------------------------------------------------------------------ fit
def fit(opts):
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    sys.path.insert(0, str(ROOT / "trees"))
    from train_trees import feature_columns
    out = artifact_dir(opts.run)
    out.mkdir(parents=True, exist_ok=True)
    folder = UNC / "output" / opts.run
    pred = pd.read_csv(folder / "predictions.csv")
    tr_ids = set(pred.loc[pred.split == "train", "isic_id"])
    va_ids = set(pred.loc[pred.split == "val", "isic_id"])

    # faithfulness filter
    probes = json.loads((ROOT / "eval/output/interpretability" / opts.run.split("/")[0] / "concepts.json").read_text())
    r2 = probes["probe_r2_all"]
    not_cited = sorted(k for k, v in r2.items() if v < NOT_ENCODED_R2 and not k.startswith("M_"))
    (out / "not_cited.json").write_text(json.dumps({"threshold_r2": NOT_ENCODED_R2, "features": not_cited,
                                                    "probe_r2": r2}, indent=1))

    # concept share (handcrafted + MONET -> ResNet logit) and a handcrafted-only model for direction checks
    feats = pd.read_parquet(ROOT / "features/output/feature_table_v1.0.parquet")
    hand = feature_columns(feats)
    monet = pd.read_parquet(ROOT / "concepts/output/monet_concepts_v1.parquet")
    mcols = [c for c in monet.columns if c.startswith("M_") and not c.startswith("M_artifact_")]
    d = pred.merge(feats[["isic_id"] + hand], on="isic_id").merge(monet[["isic_id"] + mcols], on="isic_id").dropna(subset=hand)
    logit = np.log(np.clip(d.p_malignant, 1e-6, 1 - 1e-6) / np.clip(1 - d.p_malignant, 1e-6, 1))
    models = {}
    for name, cols in (("share", hand + mcols), ("direction", hand)):
        tr = d.isic_id.isin(tr_ids).to_numpy()
        mu, sd = d.loc[tr, cols].mean(), d.loc[tr, cols].std().replace(0, 1)
        z = (d[cols] - mu) / sd
        r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(z[tr], logit[tr])
        models[name] = {"columns": cols, "mean": mu.tolist(), "std": sd.tolist(), "coef": r.coef_.tolist(),
                        "intercept": float(r.intercept_), "alpha": float(r.alpha_)}
    (out / "concept_model.json").write_text(json.dumps(models, indent=1))

    # scope flag in DINOv2 space
    z = np.load(ROOT / "classifier/output/embeddings/dinov2.npz", allow_pickle=True)
    emb = pd.DataFrame(z["emb"].astype(np.float32), index=z["isic_id"])
    e_tr, e_va = emb.loc[emb.index.isin(tr_ids)].to_numpy(), emb.loc[emb.index.isin(va_ids)].to_numpy()
    pca = PCA(128, random_state=0).fit(e_tr)
    lw = LedoitWolf().fit(pca.transform(e_tr))
    cut = float(np.quantile(lw.mahalanobis(pca.transform(e_va)), 0.95))
    np.savez_compressed(out / "scope_model.npz", pca_mean=pca.mean_, pca_components=pca.components_,
                        location=lw.location_, precision=lw.precision_, cut=cut)

    # similar-case bank (training cases, ResNet embedding)
    z = np.load(folder / "embeddings.npz", allow_pickle=True)
    keep = np.isin(z["isic_id"], list(tr_ids))
    e = z["emb"][keep].astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    lab = pred.set_index("isic_id").loc[z["isic_id"][keep]]
    np.savez_compressed(out / "retrieval_bank.npz", isic_id=z["isic_id"][keep], emb=e.astype(np.float16),
                        label=lab.label.to_numpy(), diagnosis=lab.diagnosis_3.fillna("").to_numpy())
    d2 = np.load(ROOT / "classifier/output/embeddings/dinov2.npz", allow_pickle=True)
    keep2 = np.isin(d2["isic_id"], list(tr_ids))
    e2 = d2["emb"][keep2].astype(np.float32)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    lab2 = pred.set_index("isic_id").loc[d2["isic_id"][keep2]]
    np.savez_compressed(out / "retrieval_bank_dinov2.npz", isic_id=d2["isic_id"][keep2], emb=e2.astype(np.float16),
                        label=lab2.label.to_numpy(), diagnosis=lab2.diagnosis_3.fillna("").to_numpy())
    print(f"artifacts -> {out}; not cited: {not_cited}; scope cut {cut:.1f}")


# ------------------------------------------------------------------------------------------------ runtime
class ExplanationFeatures:
    """Per-case functions used by the explanation pipeline (all inputs are per-case vectors / dicts)."""
    def __init__(self, run):
        a = artifact_dir(run)
        self.not_cited = set(json.loads((a / "not_cited.json").read_text())["features"])
        self.models = json.loads((a / "concept_model.json").read_text())
        s = np.load(a / "scope_model.npz")
        self.scope = {k: s[k] for k in s.files}
        self.banks = {}
        for space, f in (("resnet", "retrieval_bank.npz"), ("dinov2", "retrieval_bank_dinov2.npz")):
            if (a / f).exists():
                b = np.load(a / f, allow_pickle=True)
                self.banks[space] = {k: b[k] for k in b.files}
                self.banks[space]["emb"] = self.banks[space]["emb"].astype(np.float32)

    def _linear(self, name, values):
        m = self.models[name]
        z = (np.array([values[c] for c in m["columns"]], dtype=float) - np.array(m["mean"])) / np.array(m["std"])
        return np.array(m["coef"]) * z, m["intercept"]

    def concept_share(self, values, p_malignant):
        """values: handcrafted + MONET concept values of one case -> (share in [0, 1], band)."""
        contrib, b = self._linear("share", values)
        resnet_logit = float(np.log(np.clip(p_malignant, 1e-6, 1 - 1e-6) / np.clip(1 - p_malignant, 1e-6, 1)))
        concept = float(contrib.sum())
        residual = resnet_logit - (b + concept)
        share = abs(concept) / (abs(concept) + abs(residual) + 1e-9)
        return share, next(name for cut, name in SHARE_BANDS if share >= cut)

    def feature_effect(self, values):
        """Handcrafted-feature contributions to the ResNet's logit (linear readout), for direction checks."""
        contrib, _ = self._linear("direction", values)
        return dict(zip(self.models["direction"]["columns"], contrib))

    def in_scope(self, dinov2_emb):
        s = self.scope
        z = (np.asarray(dinov2_emb, dtype=np.float64) - s["pca_mean"]) @ s["pca_components"].T - s["location"]
        dist = float(z @ s["precision"] @ z)
        return dist <= float(s["cut"]), dist

    def similar_cases(self, emb, k=3, space="dinov2"):
        """k nearest training cases in `space` (dinov2: independent of the explained model; resnet: its own space)."""
        bank = self.banks[space]
        e = np.asarray(emb, dtype=np.float32)
        sims = bank["emb"] @ (e / np.linalg.norm(e))
        top = np.argsort(-sims)[:k]
        return [{"isic_id": str(bank["isic_id"][i]), "label": int(bank["label"][i]),
                 "diagnosis": str(bank["diagnosis"][i]), "similarity": float(sims[i])} for i in top]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fit"])
    ap.add_argument("--run", default="rorqual_v2_lr3e-4_20261014/resnet50_det/run")
    fit(ap.parse_args())
