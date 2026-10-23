"""
Frozen foundation-model embeddings + a linear probe for benign/malignant (classifier improvement, 2026-10-15).

  embed   one forward pass per image (split v2 train/val/test) -> classifier/output/embeddings/<name>.npz
  probe   L2-regularized logistic regression on standardized embeddings, C chosen on val AUROC; writes
          uncertaintyNet-main/output/probe_<name>/run/predictions.csv (+ predictions_meta.json) with the same columns
          as eval/export_predictions.py, so eval/calibrate_ensemble.py / paired_bootstrap / baseline_report take it.

Backbones (public weights, images never leave this machine; licences to confirm under M-8 before any deployment):
  dinov2   facebook/dinov2-large (Apache-2.0), CLS token
  monet    chanwkim/monet (MONET dermatology CLIP, Kim et al. 2024), CLIP image embedding

    python classifier/embed_probe.py embed --model dinov2 --dataset-path uncertaintyNet-main/datasets_768
    python classifier/embed_probe.py probe --model dinov2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uncertaintyNet-main"))
import dataset  # noqa: E402

MODELS = {"dinov2": "facebook/dinov2-large", "monet": "chanwkim/monet",
          "panderm": "PanDerm ViT-L/16 (Yan et al. 2025), panderm_ll_data6_checkpoint-499.pth, CC-BY-NC-ND: research only"}
PANDERM_CKPT = Path.home() / ".cache/panderm/panderm_ll_data6_checkpoint-499.pth"


def load_panderm():
    """PanDerm's encoder (CAE-style ViT-L/16) in timm's VisionTransformer: layer-scale gamma_1/2 -> ls1/ls2.gamma,
    split q/v biases -> one qkv bias (k bias 0). Strict load, so any unmapped key fails. No third-party code is run."""
    import timm
    sd = torch.load(PANDERM_CKPT, map_location="cpu", weights_only=True)
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    out = {}
    for k, v in enc.items():
        if k.endswith("attn.q_bias"):
            pre = k[: -len("q_bias")]
            out[pre + "qkv.bias"] = torch.cat([v, torch.zeros_like(v), enc[pre + "v_bias"]])
        elif k.endswith("attn.v_bias"):
            continue
        else:
            out[k.replace("gamma_1", "ls1.gamma").replace("gamma_2", "ls2.gamma")] = v
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, init_values=1e-5,
                              global_pool="avg", fc_norm=False)
    model.load_state_dict(out, strict=True)
    return model
EMB_DIR = ROOT / "classifier" / "output" / "embeddings"
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]


def load_backbone(name, device):
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    repo = MODELS[name]
    if name == "panderm":
        from transformers import CLIPImageProcessor
        model = load_panderm()
        proc = CLIPImageProcessor(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
                                  image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        model.eval().to(device)
        return (lambda px: model(px)), proc  # mean of the normalized patch tokens
    if name == "monet":
        model, proc = CLIPModel.from_pretrained(repo), CLIPProcessor.from_pretrained(repo)
        def fn(px):  # transformers >= 5 returns an output object, older versions the tensor
            out = model.get_image_features(pixel_values=px)
            return out if torch.is_tensor(out) else out.pooler_output
    else:
        model, proc = AutoModel.from_pretrained(repo), AutoImageProcessor.from_pretrained(repo)
        fn = lambda px: model(pixel_values=px).last_hidden_state[:, 0]  # noqa: E731
    model.eval().to(device)
    image_proc = proc.image_processor if hasattr(proc, "image_processor") else proc
    return fn, image_proc


@torch.no_grad()
def embed(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opts.split_csv:  # any image set, e.g. MIDAS: <image_dir>/<id>.jpg for the non-excluded rows of the csv
        data_dir = Path(opts.image_dir)
        df = pd.read_csv(opts.split_csv)
        df = df[df.split != "excluded"].rename(columns={opts.id_column: "isic_id"}).reset_index(drop=True)
    else:
        data_dir = Path(opts.dataset_path) / "ISIC_clinical"
        df = dataset.load_split(str(ROOT / "splits/isic_clinical_v2.csv"), str(data_dir))
    fn, proc = load_backbone(opts.model, device)
    feats = []
    for i in range(0, len(df), opts.batch_size):
        ids = df.isic_id.iloc[i:i + opts.batch_size]
        imgs = [Image.open(data_dir / f"{x}.jpg").convert("RGB") for x in ids]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            feats.append(fn(px).float().cpu().numpy())
        if (i // opts.batch_size) % 20 == 0:
            print(f"{i + len(ids)}/{len(df)}", flush=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{opts.model}__{opts.tag}.npz" if opts.tag else f"{opts.model}.npz"
    np.savez_compressed(EMB_DIR / name, isic_id=df.isic_id.values, emb=np.concatenate(feats).astype(np.float16),
                        repo=MODELS[opts.model])
    print(f"wrote {EMB_DIR / name}")


def probe(opts):
    z = np.load(EMB_DIR / f"{opts.model}.npz", allow_pickle=True)
    emb = pd.DataFrame(z["emb"].astype(np.float32), index=z["isic_id"])
    df = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    df = df[df.split.isin(["train", "val", "test"])].set_index("isic_id").loc[emb.index]
    tr, va = (df.split == "train").values, (df.split == "val").values
    y = df.label.astype(int).values
    scaler = StandardScaler().fit(emb.values[tr])
    X = scaler.transform(emb.values)
    best = None
    for c in C_GRID:
        m = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(X[tr], y[tr])
        auc = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
        print(f"C={c:g} val AUROC {auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, c, m)
    auc, c, m = best
    p = m.predict_proba(X)[:, 1].clip(1e-7, 1 - 1e-7)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    out = ROOT / "uncertaintyNet-main" / "output" / f"probe_{opts.model}" / "run"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"isic_id": df.index, "label": y, "pred": (p >= 0.5).astype(int), "p_malignant": p,
                  "logit_0": np.log(1 - p), "logit_1": np.log(p), "max_softmax": np.maximum(p, 1 - p),
                  "entropy_of_expected": ent, "expected_entropy": ent, "mutual_information": 0.0,
                  "mc_std_p_malignant": 0.0, "split": df.split.values, "diagnosis_3": df.diagnosis_3.values}
                 ).to_csv(out / "predictions.csv", index=False)
    te = (df.split == "test").values
    meta = {"run": f"probe_{opts.model}", "best_epoch": None, "network_type": f"linear_probe:{MODELS[opts.model]}",
            "mc_samples": 1, "seed": 0, "C": c, "val_auroc": auc, "test_auroc": roc_auc_score(y[te], p[te]), "n": len(df)}
    (out / "predictions_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "probe"])
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--dataset-path", default=str(ROOT / "uncertaintyNet-main/datasets_768"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--split_csv", default="", help="embed: another image set (e.g. splits/midas_v1.csv)")
    ap.add_argument("--id_column", default="isic_id")
    ap.add_argument("--image_dir", default="")
    ap.add_argument("--tag", default="", help="embed: output <model>__<tag>.npz")
    opts = ap.parse_args()
    embed(opts) if opts.step == "embed" else probe(opts)
