"""
Named-concept scores from MONET (dermatology CLIP, chanwkim/monet) for every split-v2 image, as candidate tree
features (C1 surrogate fidelity) that keep a clinical name.

Score = cosine(image, mean concept-prompt embedding) - cosine(image, mean reference-prompt embedding), the
concept-minus-reference contrast MONET uses. Image embeddings come from classifier/embed_probe.py
(classifier/output/embeddings/monet.npz, same model, local), so no image is re-read here. Artifact concepts (pen ink,
gel, hair) are scored for QA only and must never be tree inputs.

    python concepts/monet_concepts.py   ->  concepts/output/monet_concepts_v1.parquet
    python concepts/monet_concepts.py --emb classifier/output/embeddings/monet__midas.npz --tag midas
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parents[1]
REPO = "chanwkim/monet"
VERSION = "monet_concepts_v1"

TEMPLATES = ["This is skin image of {}", "This is dermatology image of {}", "This is image of {}"]
REFERENCE = ["skin", "normal skin", "skin lesion"]
CONCEPTS = {  # column -> (terms averaged over templates, QA-only?)
    "M_pigmented": (["pigmented lesion", "brown pigmentation"], False),
    "M_dark_pigment": (["black pigmentation", "dark brown lesion"], False),
    "M_blue_gray": (["blue-gray color", "blue-white veil"], False),
    "M_erythema": (["erythema", "redness"], False),
    "M_vessels": (["telangiectasia", "arborizing vessels"], False),
    "M_ulceration_crust": (["ulceration", "crust"], False),
    "M_scale": (["scale", "hyperkeratosis"], False),
    "M_pearly": (["pearly translucent papule", "shiny surface"], False),
    "M_nodule": (["nodule", "raised papule"], False),
    "M_flat": (["flat macule", "patch"], False),
    "M_white_scar": (["white scar-like area", "hypopigmentation"], False),
    "M_waxy_stuck_on": (["waxy stuck-on appearance", "cerebriform surface"], False),
    "M_artifact_ink": (["purple pen marking", "skin marker ink"], True),
    "M_artifact_gel": (["gel", "water drop"], True),
    "M_artifact_hair": (["hair"], True),
}


@torch.no_grad()
def text_embedding(model, processor, terms):
    prompts = [t.format(term) for term in terms for t in TEMPLATES]
    tok = processor(text=prompts, return_tensors="pt", padding=True)
    e = model.get_text_features(**tok)
    e = e if torch.is_tensor(e) else e.pooler_output  # transformers >= 5: projected embedding in pooler_output
    e = e / e.norm(dim=-1, keepdim=True)
    m = e.mean(0)
    return (m / m.norm()).numpy()


def main(emb="classifier/output/embeddings/monet.npz", tag=""):
    z = np.load(ROOT / emb, allow_pickle=True)
    img = z["emb"].astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    model = CLIPModel.from_pretrained(REPO).eval()
    processor = AutoProcessor.from_pretrained(REPO)
    ref = text_embedding(model, processor, REFERENCE)
    out = pd.DataFrame({"isic_id": z["isic_id"]})
    for col, (terms, _) in CONCEPTS.items():
        out[col] = img @ text_embedding(model, processor, terms) - img @ ref
    out_dir = ROOT / "concepts/output"
    name = f"{VERSION}__{tag}" if tag else VERSION
    out.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / f"{name}.json").write_text(json.dumps({
        "version": VERSION, "repo": REPO, "templates": TEMPLATES, "reference": REFERENCE,
        "concepts": {k: {"terms": t, "qa_only": qa} for k, (t, qa) in CONCEPTS.items()},
        "image_embeddings": emb, "n": len(out)}, indent=1))
    print(out.describe().T[["mean", "std"]])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="classifier/output/embeddings/monet.npz")
    ap.add_argument("--tag", default="", help="output monet_concepts_v1__<tag>.parquet (e.g. midas)")
    a = ap.parse_args()
    main(a.emb, a.tag)
