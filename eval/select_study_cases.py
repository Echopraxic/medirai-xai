"""
Draft clinician-study case set (plan T7.2/T7.3): ~40 test cases stratified by ResNet50 correct/wrong x
confident/uncertain x true benign/malignant (5 per cell, mask_quality 'ok' preferred), with per-case
materials for each arm: image, label-only card, Grad-CAM overlay, template (no-LLM) explanation.

Draft only: case count, strata and arms are to be agreed with the supervisor/MedirAI, and no clinician
session may run before the ethics determination (A-1). Images are public ISIC images.

    python eval/select_study_cases.py --c6 eval/output/c6 --gradcam eval/output/gradcam \
        --images uncertaintyNet-main/datasets_768/ISIC_clinical --out eval/output/study_cases_draft
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

PER_CELL = 5
SEED = 42


def main(opts):
    c6 = Path(opts.c6)
    df = pd.read_csv(c6 / "c6_per_image.csv")
    texts = {}
    with open(c6 / "template_explanations_test.jsonl", encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            texts[t["isic_id"]] = t
    df["label"] = df.isic_id.map(lambda i: texts[i]["label"])
    df["cell"] = df.stratum + "/" + np.where(df.label == 1, "malignant", "benign")
    df["prefer"] = (df.mask_quality == "ok").astype(int)
    rng = np.random.default_rng(SEED)
    df["r"] = rng.random(len(df))
    picked = (df.sort_values(["prefer", "r"], ascending=[False, True]).groupby("cell").head(PER_CELL)
              .sort_values("r").reset_index(drop=True))
    picked.insert(0, "case_id", [f"C{i + 1:02d}" for i in range(len(picked))])

    cams = np.load(Path(opts.gradcam) / "cams.npz")
    cam_by_id = dict(zip(cams["isic_id"], cams["cam"]))
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in picked.itertuples():
        d = out / r.case_id
        d.mkdir(exist_ok=True)
        src = Path(opts.images) / f"{r.isic_id}.jpg"
        shutil.copy(src, d / "image.jpg")
        cam = cam_by_id[r.isic_id].astype(np.float32)
        raw = np.asarray(Image.open(src).convert("RGB").resize(cam.shape[::-1])) / 255.0
        Image.fromarray(show_cam_on_image(raw.astype(np.float32), cam, use_rgb=True)).save(d / "gradcam.png")
        t = texts[r.isic_id]
        (d / "label_only.json").write_text(json.dumps({"prediction": t["prediction"],
                                                       "probability_malignant": t["probability_malignant"]}, indent=1))
        exp = t["explanation"]
        lines = [exp["summary"]] + ([exp["uncertainty_note"]] if exp["uncertainty_note"] else [])
        lines += [f"{i + 1}. {x['text']}" for i, x in enumerate(exp["reasons"])] + [exp["limitations"]]
        (d / "explanation.txt").write_text("\n".join(lines), encoding="utf-8")
    # the answer key (stratum, truth) is kept apart from the materials shown to clinicians
    picked[["case_id", "isic_id", "cell", "label", "source", "mask_quality", "entropy", "surrogate_agrees"]].to_csv(
        out / "answer_key.csv", index=False)
    print(picked.groupby("cell").size().to_string())
    print(f"{len(picked)} cases -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c6", required=True)
    ap.add_argument("--gradcam", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
