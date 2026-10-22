"""
Interpretability prototypes for the classifier sprint (local plan §5, I2/I3/I5/I7), run on the interim explained model
and re-run on the frozen one. Fitted on split v2 train; evaluated on the SELECTION data only (split v2 val + MIDAS
dev_val phone photos); no locked or test split is read.

  embed     ResNet penultimate embeddings + p(malignant) for MIDAS dev_val (split v2 embeddings come from
            eval/export_predictions.py)
  ood       I7  Mahalanobis distance to the training embeddings (PCA 128, Ledoit-Wolf; --space resnet or a frozen
                foundation model: dinov2, monet, panderm): does it detect phone photos,
                and does "defer if entropy high OR distance high" protect accuracy better than entropy alone?
  retrieval I5  k nearest biopsied training cases (cosine): neighbour label agreement, and whether neighbours that
                disagree with the prediction flag errors
  concepts  I3  linear probes embedding -> named concepts (handcrafted features, MONET scores): what the network encodes
            I2  concept bottleneck share: how much of the ResNet's logit a linear model on named concepts reproduces

    python interpret/prototypes.py embed --run rorqual_v2_lr3e-4_20261014/resnet50_det/run
    python interpret/prototypes.py ood|retrieval|concepts --run rorqual_v2_lr3e-4_20261014/resnet50_det/run
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
OUT = ROOT / "eval/output/interpretability"


def entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ------------------------------------------------------------------------------------------------ data
def load(run):
    """Split v2 train/val + MIDAS dev_val: embedding, label, p(malignant) (no TTA, consistent with the embeddings)."""
    folder = UNC / "output" / run
    z = np.load(folder / "embeddings.npz", allow_pickle=True)
    pred = pd.read_csv(folder / "predictions.csv").set_index("isic_id")
    ids = z["isic_id"]
    v2 = pd.DataFrame({"id": ids, "set": pred.loc[ids, "split"].map({"train": "train", "val": "internal_val",
                                                                      "test": "test"}).to_numpy(),
                       "label": pred.loc[ids, "label"].to_numpy(), "p": pred.loc[ids, "p_malignant"].to_numpy(),
                       "diagnosis": pred.loc[ids, "diagnosis_3"].fillna("").to_numpy()})
    emb = z["emb"].astype(np.float32)
    keep = v2.set.isin(["train", "internal_val"]).to_numpy()   # the test split is never read here
    v2, emb = v2[keep].reset_index(drop=True), emb[keep]
    m = np.load(OUT / run.split("/")[0] / "midas_dev_val_embeddings.npz", allow_pickle=True)
    mid = pd.DataFrame({"id": m["image_id"], "set": "midas_dev_val", "label": m["label"], "p": m["p"],
                        "diagnosis": m["diagnosis"]})
    df = pd.concat([v2, mid], ignore_index=True)
    df["pred"] = (df.p >= 0.5).astype(int)
    df["error"] = (df.pred != df.label).astype(int)
    df["entropy"] = entropy(df.p.to_numpy())
    return df, np.concatenate([emb, m["emb"].astype(np.float32)])


def embed(opts):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(UNC))
    sys.path.insert(0, str(ROOT / "eval"))
    import dataset
    import utils
    from export_predictions import build_network, load_best
    folder = UNC / "output" / opts.run
    config = utils.read_config(folder)
    net = build_network(config, folder)
    load_best(net, folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()
    feats = {}
    pool = next(m for n, m in net.named_modules() if n.endswith("avgpool"))
    pool.register_forward_hook(lambda m, i, o: feats.__setitem__("x", o.flatten(1)))
    _, tf = dataset.medirv2_transforms(config["Network"]["Basic Setup"]["input_size"])
    s = pd.read_csv(ROOT / "splits/midas_v1.csv")
    s = s[(s.split == "dev_val") & s.modality.str.startswith("phone")].reset_index(drop=True)
    embs, ps = [], []
    with torch.no_grad():
        for i in range(0, len(s), 32):
            x = torch.stack([tf(Image.open(ROOT / "data/midas/files_768" / f"{k}.jpg").convert("RGB"))
                             for k in s.image_id[i:i + 32]]).to(device)
            ps.append(F.softmax(net(x).float(), -1)[:, 1].cpu().numpy())
            embs.append(feats["x"].float().cpu().numpy())
    out = OUT / opts.run.split("/")[0]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "midas_dev_val_embeddings.npz", image_id=s.image_id.to_numpy(), label=s.label.astype(int).to_numpy(),
                        diagnosis=s.midas_path.to_numpy(), p=np.concatenate(ps), emb=np.concatenate(embs).astype(np.float16))
    print(f"{len(s)} MIDAS dev_val images embedded -> {out}")


# ------------------------------------------------------------------------------------------------ I7
def foundation_space(df, model):
    """Frozen foundation embeddings (classifier/embed_probe.py) aligned to df's rows: an outside view of the image that
    the fine-tuned ResNet cannot collapse onto familiar regions."""
    emb_dir = ROOT / "classifier/output/embeddings"
    parts = []
    for f in (f"{model}.npz", f"{model}__midas.npz"):
        z = np.load(emb_dir / f, allow_pickle=True)
        parts.append(pd.DataFrame(z["emb"].astype(np.float32), index=z["isic_id"]))
    table = pd.concat(parts)
    return table.loc[df.id].to_numpy()


def ood(opts):
    df, emb = load(opts.run)
    if opts.space != "resnet":
        emb = foundation_space(df, opts.space)
    tr = (df.set == "train").to_numpy()
    pca = PCA(128, random_state=0).fit(emb[tr])
    zt = pca.transform(emb)
    lw = LedoitWolf().fit(zt[tr])
    df["maha"] = lw.mahalanobis(zt)
    iv, md = (df.set == "internal_val").to_numpy(), (df.set == "midas_dev_val").to_numpy()
    res = {"shift_detection_auroc_midas_vs_internal_val": float(roc_auc_score(md[iv | md], df.maha[iv | md]))}
    ent_cut = float(df.entropy[iv].quantile(0.75))            # the current deferral operating point (val p75)
    maha_cut = float(df.maha[iv].quantile(0.95))              # out-of-scope: beyond 95% of internal val
    for name, mask in (("internal_val", iv), ("midas_dev_val", md)):
        d = df[mask]
        defer_e = d.entropy > ent_cut
        defer_both = defer_e | (d.maha > maha_cut)
        res[name] = {
            "n": int(len(d)), "accuracy": float(1 - d.error.mean()),
            "error_auroc_entropy": float(roc_auc_score(d.error, d.entropy)),
            "error_auroc_mahalanobis": float(roc_auc_score(d.error, d.maha)),
            "share_out_of_scope": float((d.maha > maha_cut).mean()),
            "entropy_rule": {"deferred": float(defer_e.mean()), "retained_accuracy": float(1 - d.error[~defer_e].mean())},
            "entropy_or_ood_rule": {"deferred": float(defer_both.mean()),
                                    "retained_accuracy": float(1 - d.error[~defer_both].mean())}}
    res["cuts"] = {"entropy_val_p75": ent_cut, "mahalanobis_val_p95": maha_cut}
    res["space"] = opts.space
    save(opts, f"ood_{opts.space}", res)


# ------------------------------------------------------------------------------------------------ I5
def retrieval(opts, k=5):
    df, emb = load(opts.run)
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    tr = (df.set == "train").to_numpy()
    bank, bank_label = e[tr], df.label[tr].to_numpy()
    res = {"k": k}
    for name in ("internal_val", "midas_dev_val"):
        m = (df.set == name).to_numpy()
        sims = e[m] @ bank.T
        nn = np.argsort(-sims, axis=1)[:, :k]
        nb_mal = bank_label[nn].mean(1)
        d = df[m].reset_index(drop=True)
        disagree = np.where(d.pred == 1, 1 - nb_mal, nb_mal)   # share of neighbours whose label contradicts the call
        res[name] = {"precision_at_k_same_label": float((bank_label[nn] == d.label.to_numpy()[:, None]).mean()),
                     "neighbour_vote_auroc": float(roc_auc_score(d.label, nb_mal)),
                     "error_auroc_neighbour_disagreement": float(roc_auc_score(d.error, disagree)),
                     "error_auroc_entropy": float(roc_auc_score(d.error, d.entropy)),
                     "error_auroc_low_top1_similarity": float(roc_auc_score(d.error, -sims.max(1))),
                     "mean_top1_cosine": float(sims.max(1).mean())}
    save(opts, "retrieval", res)


# ------------------------------------------------------------------------------------------------ I3 / I2
def concepts(opts):
    df, emb = load(opts.run)
    feats = pd.read_parquet(ROOT / "features/output/feature_table_v1.0.parquet")
    sys.path.insert(0, str(ROOT / "trees"))
    from train_trees import feature_columns
    hand = feature_columns(feats)
    monet = pd.read_parquet(ROOT / "concepts/output/monet_concepts_v1.parquet").rename(columns={"isic_id": "id"})
    mcols = [c for c in monet.columns if c.startswith("M_") and not c.startswith("M_artifact_")]
    d = df[df.set.isin(["train", "internal_val"])].rename_axis("row").reset_index().merge(
        feats[["isic_id"] + hand].rename(columns={"isic_id": "id"}), on="id").merge(monet[["id"] + mcols], on="id")
    d = d.dropna(subset=hand).reset_index(drop=True)       # MIDAS has no handcrafted features (no reliable masks)
    rows = d.row.to_numpy()                                # positions of these images in `emb`
    E = StandardScaler().fit(emb[rows][(d.set == "train").to_numpy()]).transform(emb[rows])
    tr, va = (d.set == "train").to_numpy(), (d.set == "internal_val").to_numpy()
    # I3: which named concepts are linearly decodable from the ResNet's embedding (R² on internal val)
    probes = {}
    for c in hand + mcols:
        y = d[c].to_numpy()
        m = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(E[tr], y[tr])
        probes[c] = float(r2_score(y[va], m.predict(E[va])))
    # I2: concept bottleneck share: linear readout of the ResNet's decision from named concepts only
    logit = np.log(np.clip(d.p, 1e-6, 1 - 1e-6) / np.clip(1 - d.p, 1e-6, 1)).to_numpy()
    share = {}
    for name, cols in (("handcrafted", hand), ("monet", mcols), ("handcrafted+monet", hand + mcols)):
        C = StandardScaler().fit(d.loc[tr, cols]).transform(d[cols])
        r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(C[tr], logit[tr])
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(C[tr], d.pred[tr])
        share[name] = {"logit_r2_internal_val": float(r2_score(logit[va], r.predict(C[va]))),
                       "agreement_with_resnet_call": float((clf.predict(C[va]) == d.pred[va]).mean())}
    res = {"n_train": int(tr.sum()), "n_val": int(va.sum()),
           "probe_r2_top": dict(sorted(probes.items(), key=lambda kv: -kv[1])[:12]),
           "probe_r2_bottom": dict(sorted(probes.items(), key=lambda kv: kv[1])[:8]),
           "probe_r2_monet": {c: probes[c] for c in mcols},
           "probe_r2_all": probes,
           "concept_bottleneck": share}
    save(opts, "concepts", res)


def save(opts, name, res):
    out = OUT / opts.run.split("/")[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "ood", "retrieval", "concepts"])
    ap.add_argument("--run", default="rorqual_v2_lr3e-4_20261014/resnet50_det/run")
    ap.add_argument("--space", default="resnet", choices=["resnet", "dinov2", "monet", "panderm"],
                    help="ood: embedding space for the Mahalanobis distance")
    opts = ap.parse_args()
    {"embed": embed, "ood": ood, "retrieval": retrieval, "concepts": concepts}[opts.step](opts)
