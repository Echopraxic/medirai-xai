"""
Export per-image predictions of an uncertaintyNet run, keyed by isic_id, for every split.

Output (in the run folder unless --out is given):
  predictions.csv  isic_id, split, label, pred, p_malignant, logit_0/1 (deterministic) or mean MC logits,
                   max_softmax, entropy_of_expected, expected_entropy, mutual_information, mc_std_p_malignant
  embeddings.npz   isic_id + 2048-d ResNet avgpool embedding (float16), for optional embedding features

Variational runs draw --mc-samples posterior samples (VLL samples only the last layer, so this is cheap).

    python eval/export_predictions.py --run s2_local/resnet50_det/run_<date> \
        --dataset-path uncertaintyNet-main/datasets_768
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
sys.path.insert(0, str(UNC))

import dataset  # noqa: E402
import models.model as model  # noqa: E402
import models.model_utils as mutils  # noqa: E402
import utils  # noqa: E402


def build_network(config, run_folder):
    """Same reconstruction as test.py: deterministic base, wrapped for (partial_)bayesian runs."""
    net_config = config["Network"]["Basic Setup"]
    t = config["Training"]
    net = model.get_model(dict(net_config, network_type="deterministic"))
    if "bayesian" in net_config["network_type"]:
        net = mutils.build_variational_model(net, net_config["network_type"], str(run_folder),
                                             t["Bayesian Parameters"], t["Partial Bayesian Parameters"],
                                             t["Layer Bayesian Parameters"], save_init=False)
    return net


def load_best(net, run_folder):
    ckpt = run_folder / "checkpoint_best_val.pth.tar"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(state["network_state_dict"], strict=True)
    return state.get("epoch")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


@torch.no_grad()
def run(opts):
    run_folder = UNC / "output" / opts.run
    config = utils.read_config(run_folder)
    net_config = config["Network"]["Basic Setup"]
    bayesian = "bayesian" in net_config["network_type"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = build_network(config, run_folder)
    epoch = load_best(net, run_folder)
    if bayesian:
        net.num_posterior_samples = opts.mc_samples
    net.to(device).eval()

    embeddings = {}
    pool = next(m for name, m in net.named_modules() if name.endswith("avgpool"))
    pool.register_forward_hook(lambda m, i, o: embeddings.__setitem__("x", o.flatten(1)))

    split_file = config["Training"]["Dataset"].get("split_file") or None
    df = dataset.load_split(split_file)
    data_dir = Path(opts.dataset_path) / "ISIC_clinical"
    _, test_tf = dataset.medirv2_transforms(net_config["input_size"])
    loader = DataLoader(dataset.ISICDatasetV2(df, str(data_dir), transform=test_tf),
                        batch_size=opts.batch_size, shuffle=False)

    rows, embs = [], []
    torch.manual_seed(opts.seed)
    for inputs, labels, ids in loader:
        inputs = inputs.to(device)
        if bayesian:
            logits, _ = net(inputs)                      # [S, B, C]
        else:
            logits = net(inputs).unsqueeze(0)            # [1, B, C]
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        mean_p = probs.mean(0)
        eoe = utils.entropy_of_expected(probs)
        ee = utils.expected_entropy(probs)
        mean_logits = logits.float().mean(0).cpu().numpy()
        for j, isic_id in enumerate(ids):
            rows.append({
                "isic_id": isic_id, "label": int(labels[j]), "pred": int(mean_p[j].argmax()),
                "p_malignant": float(mean_p[j, 1]), "logit_0": float(mean_logits[j, 0]),
                "logit_1": float(mean_logits[j, 1]), "max_softmax": float(mean_p[j].max()),
                "entropy_of_expected": float(eoe[j]), "expected_entropy": float(ee[j]),
                "mutual_information": float(eoe[j] - ee[j]),
                "mc_std_p_malignant": float(probs[:, j, 1].std()) if bayesian else 0.0,
            })
        embs.append(embeddings["x"].half().cpu().numpy())

    out = pd.DataFrame(rows).merge(df[["isic_id", "split", "diagnosis_3"]], on="isic_id", how="left")
    out_dir = Path(opts.out) if opts.out else run_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "predictions.csv", index=False)
    np.savez_compressed(out_dir / "embeddings.npz", isic_id=out["isic_id"].values, emb=np.concatenate(embs))
    meta = {"run": opts.run, "best_epoch": epoch, "network_type": net_config["network_type"],
            "mc_samples": opts.mc_samples if bayesian else 1, "seed": opts.seed,
            "checkpoint_sha256_16": file_sha(run_folder / "checkpoint_best_val.pth.tar"),
            "dataset_path": str(opts.dataset_path), "n": len(out)}
    pd.Series(meta).to_json(out_dir / "predictions_meta.json", indent=1)
    test = out[out.split == "test"]
    print(f"{len(out)} images exported; test accuracy {(test.pred == test.label).mean():.4f} -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder relative to uncertaintyNet-main/output")
    ap.add_argument("--dataset-path", required=True, help="folder containing ISIC_clinical/")
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    run(ap.parse_args())
