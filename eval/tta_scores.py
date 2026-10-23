"""
Per-image 8-view TTA scores of one run (prereg 2026-10-23, sections 2-6). Writes <run>/tta_scores.npz with, per set,
the image ids, the binary logit margin (logit_1 - logit_0, uncalibrated) and p(melanoma) when the run has the head.
Calibration, seed ensembles, the deferral rule and the per-domain policy are all fit from these files
(eval/sprint2_analysis.py), so every model is run on the GPU once.

Sets: v2_val, v2_test, midas_dev_val, midas_dev_test (as eval/final_model_report.py) and dsd (splits/f17k_dsd_v1.csv:
Fitzpatrick17k FST 4-6; its `dsd` column marks the FST 5-6 selection set). No locked set is read.

    python eval/tta_scores.py --run <run under uncertaintyNet-main/output> --isic_dir <dir> --midas_dir <dir> --dsd_dir <dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from final_model_report import UNC, sets, tta_logits  # noqa: E402
from selection_score import load_net  # noqa: E402


def main(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = UNC / "output" / opts.run
    net, tf, kind = load_net(run)
    net.to(device).eval()
    data = sets(opts.isic_dir, opts.midas_dir)
    dsd = pd.read_csv(ROOT / "splits/f17k_dsd_v1.csv")
    data["dsd"] = {"ids": dsd.isic_id.tolist(), "dir": opts.dsd_dir}
    out = {}
    for name, d in data.items():
        margin, pmel = tta_logits(net, kind, tf, d["ids"], d["dir"], device)
        out[f"{name}_ids"], out[f"{name}_margin"] = np.array(d["ids"]), margin
        if pmel is not None:
            out[f"{name}_pmel"] = pmel
        print(f"{name}: {len(margin)} images", flush=True)
    np.savez_compressed(run / "tta_scores.npz", **out)
    print(f"wrote {run / 'tta_scores.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--isic_dir", required=True)
    ap.add_argument("--midas_dir", required=True)
    ap.add_argument("--dsd_dir", required=True)
    main(ap.parse_args())
