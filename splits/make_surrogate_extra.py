"""
Images the final model (kdv8f_mel s3) never trained on, as extra training rows for the C1 surrogate (plan decision
2026-11-04, arm E): SD-198 tumour-class images (split v8s rows with source SD-198) + Mendeley light-skin melanoma
originals (split v7 rows MDO_*, label 1, melanoma). Train rows for the tree only; never val/test.

Kept only if the image is not a near-duplicate (dHash Hamming <= DUP_HAMMING, eval/external_test.py) of any image in
split v8f (the final model's train, val, test and MIDAS dev rows: a duplicate of a training image is not out-of-sample).
v8s/v7 already dropped duplicates of evaluation, locked and dev-set images. A Mendeley melanoma that duplicates an
SD-198 image (same DermQuest atlas) is dropped and the SD-198 copy kept. Run on Narval next to the image cache.

    sbatch --account=def-omar12 --cpus-per-task=16 --mem=16G --time=1:00:00 --wrap       "module load python/3.12; source .venv/bin/activate; python splits/make_surrogate_extra.py --image_dir $SCRATCH/medirai/datasets_768_v4/ISIC_clinical"
        -> splits/surrogate_extra_v1.csv (split v8 columns, split = "train")
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "eval")]
from external_test import DUP_HAMMING, dhash  # noqa: E402
from make_split_v7 import bits, min_distance  # noqa: E402


def _hash(path):
    return dhash(Image.open(path))


def hashes(ids, img_dir):
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
        return bits(pool.map(_hash, [img_dir / f"{i}.jpg" for i in ids], chunksize=64))


def main(opts):
    img = Path(opts.image_dir)
    v8s = pd.read_csv(HERE / "isic_v8s_v6_plus_sd198.csv")
    v7 = pd.read_csv(HERE / "isic_v7_v6_plus_mendeley.csv")
    v8f = pd.read_csv(HERE / "isic_v8f_v6_plus_f17k.csv")
    sd = v8s[v8s.source == "SD-198"]
    md = v7[v7.isic_id.str.startswith("MDO_") & (v7.label == 1) & v7.diagnosis_3.str.startswith("Melanoma")]
    seen = v8f[v8f.split != "excluded"]
    print(f"SD-198 {len(sd)}, Mendeley melanoma originals {len(md)}, v8f images {len(seen)}", flush=True)
    h_seen = hashes(seen.isic_id, img)
    h_sd, h_md = hashes(sd.isic_id, img), hashes(md.isic_id, img)
    sd_keep = min_distance(h_sd, h_seen) > DUP_HAMMING
    md_keep = (min_distance(h_md, h_seen) > DUP_HAMMING) & (min_distance(h_md, h_sd) > DUP_HAMMING)
    out = pd.concat([sd[sd_keep], md[md_keep]], ignore_index=True).assign(split="train")
    out.to_csv(HERE / "surrogate_extra_v1.csv", index=False)
    print(f"kept SD-198 {int(sd_keep.sum())}/{len(sd)}, Mendeley melanoma {int(md_keep.sum())}/{len(md)}; "
          f"melanomas {int(out.diagnosis_3.str.startswith('Melanoma').sum())}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    main(ap.parse_args())
