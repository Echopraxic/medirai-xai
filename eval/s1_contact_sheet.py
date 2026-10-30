"""
Contact sheets and a blank review table for S1-9's confident-wrong locked-set images (decision 2026-10-30: Shatha
works under the same AIMI terms, so DDI/DDI-2 thumbnails are included). Reads eval/output/s1_error_lists/all_strata.csv
(eval/s1_error_lists.py) and the image paths from eval/external_test.py. Descriptive only: no model is run.

Outputs go to a gitignored folder (handoff/ by default) and must not be redistributed (DDI/DDI-2: Stanford AIMI terms).

    python eval/s1_contact_sheet.py --out handoff/s1_asks_2026-10-30/confident_wrong
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from external_test import SOURCES  # noqa: E402

REVIEW_COLUMNS = ["image_quality", "framing", "lesion_visible", "suspected_cause", "notes"]
CAUSES = "poor image quality / wide framing / lesion not visible / atypical presentation / likely label issue / unclear"


def main(opts):
    a = pd.read_csv(ROOT / opts.strata)
    cw = a[a.stratum == "confident_wrong"].sort_values(["set", "wrong_confidence"], ascending=[True, False])
    paths = pd.concat([SOURCES[s]()[["image_id", "path"]].assign(set=s) for s in cw.set.unique()])
    paths["image_id"] = paths.image_id.astype(str)
    cw = cw.assign(image_id=cw.image_id.astype(str)).merge(paths, on=["set", "image_id"], how="left")
    assert cw.path.notna().all(), "image path missing for some confident-wrong rows"
    cw.insert(0, "ref", [f"{s}-{k + 1:02d}" for s, g in cw.groupby("set", sort=False) for k in range(len(g))])
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = ["ref", "set", "image_id", "label", "error_type", "diagnosis", "tone", "p_malignant", "entropy_of_expected"]
    cw[keep].assign(**{c: "" for c in REVIEW_COLUMNS}).to_csv(out / "confident_wrong_review.csv", index=False)
    side, cols, per = opts.thumb, 6, 36
    for s, g in cw.groupby("set"):
        for page, start in enumerate(range(0, len(g), per), 1):
            chunk = g.iloc[start:start + per]
            rows = -(-len(chunk) // cols)
            sheet = Image.new("RGB", (cols * side, rows * (side + 30)), "white")
            draw = ImageDraw.Draw(sheet)
            for k, r in enumerate(chunk.itertuples()):
                im = Image.open(r.path).convert("RGB")
                im.thumbnail((side, side))
                x, y = (k % cols) * side, (k // cols) * (side + 30)
                sheet.paste(im, (x + (side - im.width) // 2, y))
                draw.text((x + 4, y + side + 2), f"{r.ref} {r.error_type} p={r.p_malignant:.2f}", fill="black")
                draw.text((x + 4, y + side + 15), str(r.diagnosis)[:30], fill="black")
            sheet.save(out / f"sheet_{s}_{page}.jpg", quality=85)
    (out / "README.txt").write_text(
        f"{len(cw)} confident-wrong images (entropy below the internal-val p75 cut and wrong). Thumbnails are labelled with "
        f"`ref` from confident_wrong_review.csv. Suggested causes: {CAUSES}.\nDDI/DDI-2 images: Stanford AIMI research-use "
        "terms, do not redistribute.\n")
    print(f"{len(cw)} images -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strata", default="eval/output/s1_error_lists/all_strata.csv")
    ap.add_argument("--out", default="handoff/s1_asks_2026-10-30/confident_wrong")
    ap.add_argument("--thumb", type=int, default=220)
    main(ap.parse_args())
