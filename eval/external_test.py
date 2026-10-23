"""
Score uncertaintyNet runs on an external test set (plan T9.6): same preprocessing as training (longest side 768 px,
Lanczos, then the MEDIRV2 test transform), images as framed by the source (no crop: MCR-SL close-ups often show
several moles, so a lesion crop could pick the wrong one).

Sources (builders below; each gives image path, label 1 = malignant, lesion group, diagnosis):
  isic_overview  ISIC "clinical: overview" (splits/sources/isic_clinical_overview.csv, data/isic_clinical_overview/images);
                 Indeterminate dropped; lesions shared with split v2 dropped
  mcr_sl         MCR-SL clinical photos (data/mcr_sl/MCR-SL_1024); label = lesion `malignancy` (4-dermatologist
                 consensus, 29 biopsied); `unknown` and AK dropped
Protocol (classifier sprint, 2026-10-21): ddi, ddi2, mcr_sl, isic_overview are LOCKED: scored only with --final
'<reason>', each look appended to eval/output/external/looks.csv. midas reads splits/midas_v1.csv (default dev_val:
model-selection data).
  ddi, ddi2, midas  Stanford AIMI sets from Redivis (tools/download_redivis.py; 768-px copies in data/<name>/files_768);
                 reported by Fitzpatrick group (and MIDAS by camera distance); see each builder

Near-duplicates of split-v2 images (256-bit dHash, Hamming <= 10, against the 768-px cache) are dropped and counted.

Metrics: image-level accuracy / AUROC / malignant, benign and melanoma recall with 95% CIs from a bootstrap over
lesions; lesion-level AUROC (mean p over a lesion's images); share deferred at the run's internal val p75 entropy, and
entropy AUROC for errors (does uncertainty rise on shifted data?). Internal test numbers of the same run alongside.

    python eval/external_test.py --source mcr_sl --runs rorqual_v2_lr3e-4_20261014/resnet50_det/run \
        narval_v3_medir_v3_lr3e-4_20261015/resnet50_det/run
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
sys.path.insert(0, str(UNC))
sys.path.insert(0, str(ROOT / "eval"))

import dataset  # noqa: E402
import utils  # noqa: E402
from export_predictions import build_network, load_best  # noqa: E402

MAX_SIDE = 768
DUP_HAMMING = 10   # of 256 bits; unrelated photos are >= 21 apart (a 64-bit hash matched plain-skin photos)
V2_CACHE = UNC / "datasets_768" / "ISIC_clinical"


def build_isic_overview():
    meta = pd.read_csv(ROOT / "splits/sources/isic_clinical_overview.csv")
    meta = meta[meta.diagnosis_1.isin(["Benign", "Malignant"])].copy()
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    meta = meta[~meta.lesion_id.isin(set(v2.lesion_id.dropna()))]
    return pd.DataFrame({
        "image_id": meta.isic_id, "path": [str(ROOT / "data/isic_clinical_overview/images" / f"{i}.jpg") for i in meta.isic_id],
        "label": (meta.diagnosis_1 == "Malignant").astype(int), "group": meta.lesion_id.fillna(meta.isic_id),
        "diagnosis": meta.diagnosis_3.fillna(meta.diagnosis_1)})


def build_mcr_sl():
    d = ROOT / "data/mcr_sl/MCR-SL_1024"
    img = pd.read_excel(d / "image.xlsx")
    les = pd.read_excel(d / "lesion.xlsx")
    df = img[img.modality == "clinical"].merge(les[["lesion_id", "malignancy", "lesion_diagnosis"]], on="lesion_id")
    df = df[df.malignancy.isin(["Malignant", "Non-malignant"]) & (df.lesion_diagnosis != "AK")]  # AK: ISIC "Indeterminate"
    return pd.DataFrame({
        "image_id": df.image_id, "path": [str(d / "clinical" / f"{i}.jpg") for i in df.image_id],
        "label": (df.malignancy == "Malignant").astype(int), "group": df.lesion_id,
        "diagnosis": df.lesion_diagnosis.replace({"MEL": "Melanoma", "BCC": "Basal cell carcinoma",
                                                  "SCC": "Squamous cell carcinoma", "NEV": "Nevus",
                                                  "SK": "Seborrheic keratosis"})})


def _fst_group(text):
    """Fitzpatrick type (1-6 as digits or roman numerals at the start, e.g. 'iii darker white skin', '3, 4', 34)."""
    t = str(text).strip().lower()
    roman = {"vi": 6, "v": 5, "iv": 4, "iii": 3, "ii": 2, "i": 1}
    first = next((roman[r] for r in roman if t.split(" ")[0] == r), None)
    if first is None and t[:1].isdigit():
        first = int(t[0])
    return {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}.get(first)


def build_ddi():
    """DDI (Stanford, biopsy-proven, balanced Fitzpatrick I-II / III-IV / V-VI). No patient id in the release, so
    images are the bootstrap unit (656 images of 570 patients)."""
    d = ROOT / "data/ddi"
    m = pd.read_csv(d / "ddi_metadata.csv")
    return pd.DataFrame({
        "image_id": m.DDI_file, "path": [str(d / "files_768" / (Path(f).stem + ".jpg")) for f in m.DDI_file],
        "label": m.malignant.astype(str).str.lower().eq("true").astype(int), "group": m.DDI_file,
        "diagnosis": m.disease.str.replace("-", " ").str.capitalize(),
        "tone": m.skin_tone.map({12: "I-II", 34: "III-IV", 56: "V-VI"})})


def build_ddi2():
    """DDI-2 (Stanford, 550 self-identified Asian patients, biopsy-proven); grouped by patient."""
    d = ROOT / "data/ddi2"
    m = pd.read_csv(d / "final_ddi2_asian_spreadsheet.csv")
    m = m[m.benign_malignant.str.lower().isin(["benign", "malignant"])]
    files = {Path(f).stem: f for f in pd.read_csv(d / "final_ddi2_asian_photos_no_metadata.csv").file_name}
    m = m[m.photo_id.astype(str).isin(files)]
    return pd.DataFrame({
        "image_id": m.photo_id.astype(str), "path": [str(d / "files_768" / f"{i}.jpg") for i in m.photo_id.astype(str)],
        "label": (m.benign_malignant.str.lower() == "malignant").astype(int), "group": m.deidentified_patient_id,
        "diagnosis": m.diagnosis_general.fillna(m.diagnosis_detailed), "tone": m.fitzpatrick_skin_type.map(_fst_group)})


def build_midas(split="dev_val"):
    """MRA-MIDAS phone photos (6 in / 1 ft) from one split of splits/midas_v1.csv (patient-grouped; labels and
    exclusions defined there). Bootstrap unit = patient. dev_val is model-selection data, not a locked test."""
    d = ROOT / "data/midas"
    m = pd.read_csv(ROOT / "splits/midas_v1.csv")
    m = m[(m.split == split) & m.modality.str.startswith("phone")]
    return pd.DataFrame({
        "image_id": m.image_id, "path": [str(d / "files_768" / f"{i}.jpg") for i in m.image_id],
        "label": m.label.astype(int), "group": m.patient_id,
        "diagnosis": m.midas_path.str.split("-", n=1).str[1].str.strip().str.capitalize(),
        "tone": m.midas_fitzpatrick.map(_fst_group), "distance": m.modality.str.replace("phone_", "")})


SOURCES = {"isic_overview": build_isic_overview, "mcr_sl": build_mcr_sl, "ddi": build_ddi, "ddi2": build_ddi2,
           "midas": build_midas}


def load_768(path):
    im = Image.open(path).convert("RGB")
    s = MAX_SIDE / max(im.size)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS) if s < 1 else im


def dhash(im, size=16):
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def v2_hashes(cache_file):
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        return z["ids"], z["h"]
    ids = sorted(p.stem for p in V2_CACHE.glob("*.jpg"))
    h = np.stack([dhash(Image.open(V2_CACHE / f"{i}.jpg")) for i in ids])
    np.savez_compressed(cache_file, ids=np.array(ids), h=h)
    return np.array(ids), h


def near_duplicates(hashes, ref_ids, ref_h, max_dist=DUP_HAMMING):
    """For each external hash, the closest v2 image id if within max_dist bits, else None."""
    bits = np.unpackbits(ref_h, axis=1)
    out = []
    for h in hashes:
        d = (np.unpackbits(h)[None, :] != bits).sum(1)
        out.append(ref_ids[d.argmin()] if d.min() <= max_dist else None)
    return out


def cluster_bootstrap(df, fn, n_boot=2000, seed=0):
    """95% CI of fn(df) resampling lesion groups with replacement."""
    rng = np.random.default_rng(seed)
    groups = df.group.unique()
    idx = {g: np.flatnonzero(df.group.values == g) for g in groups}
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])
        v = fn(df.iloc[take])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def _auroc(d):
    return roc_auc_score(d.label, d.p_malignant) if d.label.nunique() == 2 else np.nan


def _recall(d, mask):
    sub = d[mask(d)]
    return (sub.pred == sub.label).mean() if len(sub) else np.nan


METRICS = {
    "accuracy": lambda d: (d.pred == d.label).mean(),
    "AUROC": _auroc,
    "malignant recall": lambda d: _recall(d, lambda x: x.label == 1),
    "benign recall": lambda d: _recall(d, lambda x: x.label == 0),
    "melanoma recall": lambda d: _recall(d, lambda x: x.diagnosis.str.contains("Melanoma", case=False)),
}


def summarize(df, eoe_threshold, n_boot):
    res = {"n_images": len(df), "n_lesions": int(df.group.nunique()), "n_malignant_images": int(df.label.sum()),
           "n_melanoma_images": int(df.diagnosis.str.contains("Melanoma", case=False).sum())}
    for name, fn in METRICS.items():
        v = fn(df)
        res[name] = {"value": None if np.isnan(v) else float(v),
                     "ci95": cluster_bootstrap(df, fn, n_boot) if not np.isnan(v) else [None, None]}
    les = df.groupby("group").agg(label=("label", "first"), p_malignant=("p_malignant", "mean")).reset_index()
    res["lesion-level AUROC"] = float(_auroc(les)) if les.label.nunique() == 2 else None
    res["deferred share (internal val p75 entropy)"] = float((df.entropy_of_expected > eoe_threshold).mean())
    err = (df.pred != df.label).astype(int)
    res["entropy AUROC for errors"] = float(roc_auc_score(err, df.entropy_of_expected)) if err.nunique() == 2 else None
    for col in ("tone", "distance"):
        if col in df and df[col].notna().any():
            res[f"by {col}"] = {str(k): {"n": int(len(g)), "n_malignant": int(g.label.sum()),
                                         "accuracy": float((g.pred == g.label).mean()),
                                         "AUROC": float(_auroc(g)) if g.label.nunique() == 2 else None,
                                         "AUROC_ci95": cluster_bootstrap(g, _auroc, n_boot) if g.label.nunique() == 2 else None,
                                         "malignant recall": None if not g.label.any() else float(_recall(g, lambda x: x.label == 1)),
                                         "benign recall": None if g.label.all() else float(_recall(g, lambda x: x.label == 0))}
                                for k, g in df.groupby(col)}
    res["accuracy by diagnosis"] = {k: {"n": int(len(g)), "accuracy": float((g.pred == g.label).mean())}
                                    for k, g in df.groupby("diagnosis") if len(g) >= 5}
    return res


def paired_diff(d, tag_a, tag_b, n_boot=2000, seed=0):
    """B - A on the same images, 95% CI from one shared lesion-level bootstrap per metric."""
    rng = np.random.default_rng(seed)
    groups = d.group.unique()
    idx = {g: np.flatnonzero(d.group.values == g) for g in groups}
    view = lambda t, rows: rows.assign(p_malignant=rows[f"{t}__p_malignant"], pred=rows[f"{t}__pred"])
    out = {}
    for name, fn in METRICS.items():
        point = fn(view(tag_b, d)) - fn(view(tag_a, d))
        if np.isnan(point):
            continue
        vals = []
        for _ in range(n_boot):
            r = d.iloc[np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])]
            v = fn(view(tag_b, r)) - fn(view(tag_a, r))
            if not np.isnan(v):
                vals.append(v)
        ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        out[name] = {"diff": float(point), "ci95": ci, "significant": bool(ci[0] > 0 or ci[1] < 0)}
    return out


def internal_reference(run_folder):
    p = pd.read_csv(run_folder / "predictions.csv")
    thr = float(p.loc[p.split == "val", "entropy_of_expected"].quantile(0.75))
    t = p[p.split == "test"]
    return thr, {"accuracy": float((t.pred == t.label).mean()), "AUROC": float(roc_auc_score(t.label, t.p_malignant)),
                 "deferred share": float((t.entropy_of_expected > thr).mean())}


@torch.no_grad()
def score(run, images, batch_size=32):
    run_folder = UNC / "output" / run
    config = utils.read_config(run_folder)
    net = build_network(config, run_folder)
    epoch = load_best(net, run_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()
    _, tf = dataset.medirv2_transforms(config["Network"]["Basic Setup"]["input_size"])
    rows = []
    for i in range(0, len(images), batch_size):
        x = torch.stack([tf(im) for im in images[i:i + batch_size]]).to(device)
        p = F.softmax(net(x).float(), -1).cpu().numpy()
        rows.append(p)
    p = np.concatenate(rows)
    eoe = utils.entropy_of_expected(p[None])
    return pd.DataFrame({"p_malignant": p[:, 1], "pred": p.argmax(1), "entropy_of_expected": eoe}), epoch


def run_tag(run):
    return run.split("/")[0]


def first_run(report):
    return next(iter(report["runs"].values()))["external"]


def write_md(report, path):
    lines = [f"# External test: {report['source']}", "", report["note"], "",
             f"Images {report['n_images_scored']} (dropped as near-duplicates of split v2: {report['near_duplicates_dropped']}).", ""]
    hdr = ["accuracy", "AUROC", "malignant recall", "benign recall", "melanoma recall"]
    lines += ["| run | " + " | ".join(hdr) + " | lesion AUROC | deferred | entropy AUROC errors | internal test acc / AUROC / deferred |",
              "|---" * (len(hdr) + 5) + "|"]
    for tag, r in report["runs"].items():
        e = r["external"]
        cells = []
        for k in hdr:
            v, ci = e[k]["value"], e[k]["ci95"]
            cells.append("n/a" if v is None else f"{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]")
        i = r["internal_test"]
        lines.append(f"| {tag} | " + " | ".join(cells) + f" | {e['lesion-level AUROC'] or float('nan'):.3f} | "
                     f"{e['deferred share (internal val p75 entropy)']:.1%} | {e['entropy AUROC for errors'] or float('nan'):.3f} | "
                     f"{i['accuracy']:.3f} / {i['AUROC']:.3f} / {i['deferred share']:.1%} |")
    lines += ["", "Accuracy by diagnosis (n >= 5 images):", "", "| diagnosis | n | " + " | ".join(report["runs"]) + " |",
              "|---|---|" + "---|" * len(report["runs"])]
    first = next(iter(report["runs"].values()))["external"]["accuracy by diagnosis"]
    for dx, v in sorted(first.items(), key=lambda kv: -kv[1]["n"]):
        accs = [f"{r['external']['accuracy by diagnosis'][dx]['accuracy']:.3f}" for r in report["runs"].values()]
        lines.append(f"| {dx} | {v['n']} | " + " | ".join(accs) + " |")
    for col in ("tone", "distance"):
        key = f"by {col}"
        if key in first_run(report):
            lines += ["", f"By {col} (AUROC [95% CI] / malignant recall / benign recall):", "",
                      f"| {col} | n (malignant) | " + " | ".join(report["runs"]) + " |", "|---|---|" + "---|" * len(report["runs"])]
            for k, v in first_run(report)[key].items():
                cells = []
                for r in report["runs"].values():
                    x = r["external"][key][k]
                    auc = "n/a" if x["AUROC"] is None else f"{x['AUROC']:.3f} [{x['AUROC_ci95'][0]:.2f}, {x['AUROC_ci95'][1]:.2f}]"
                    fmt = lambda z: "n/a" if z is None else f"{z:.2f}"  # noqa: E731
                    cells.append(f"{auc} / {fmt(x['malignant recall'])} / {fmt(x['benign recall'])}")
                lines.append(f"| {k} | {v['n']} ({v['n_malignant']}) | " + " | ".join(cells) + " |")
    for tag, diffs in report.get("paired_vs_first", {}).items():
        lines += ["", f"Paired, {tag} minus {next(iter(report['runs']))} (lesion bootstrap):", "",
                  "| metric | B - A | 95% CI | distinguishable from 0 |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {'yes' if v['significant'] else 'no'} |"
                  for k, v in diffs.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


LOCKED = {"ddi", "ddi2", "mcr_sl", "isic_overview"}  # classifier sprint protocol (2026-10-21): never trained on or selected with
LOOK_LOG = ROOT / "eval/output/external/looks.csv"


def log_look(opts):
    """Every scoring of a locked set is appended to looks.csv (date, source, runs, reason); the count is reported."""
    import datetime
    row = pd.DataFrame([{"time": datetime.datetime.now().isoformat(timespec="seconds"), "source": opts.source,
                         "runs": " ".join(opts.runs), "reason": opts.final}])
    row.to_csv(LOOK_LOG, mode="a", header=not LOOK_LOG.exists(), index=False)
    return int((pd.read_csv(LOOK_LOG).source == opts.source).sum())


def main(opts):
    if opts.source in LOCKED:
        if not opts.final:
            raise SystemExit(f"{opts.source} is a locked external test set: score it only for a final candidate, with "
                             f"--final '<reason>' (logged to {LOOK_LOG.relative_to(ROOT)})")
        print(f"locked set {opts.source}: look #{log_look(opts)}")
    df = (build_midas(opts.midas_split) if opts.source == "midas" else SOURCES[opts.source]()).reset_index(drop=True)
    images = [load_768(p) for p in df.path]
    out = ROOT / "eval/output/external" / (f"midas_{opts.midas_split}" if opts.source == "midas" else opts.source)
    out.mkdir(parents=True, exist_ok=True)
    ref_ids, ref_h = v2_hashes(ROOT / "eval/output/external/_v2_dhash.npz")
    df["v2_near_duplicate"] = near_duplicates([dhash(im) for im in images], ref_ids, ref_h)
    keep = df.v2_near_duplicate.isna().to_numpy()
    df, images = df[keep].reset_index(drop=True), [im for im, k in zip(images, keep) if k]
    report = {"source": opts.source, "near_duplicates_dropped": int((~keep).sum()), "n_images_scored": len(df),
              "runs": {}}
    report["note"] = {"isic_overview": "ISIC clinical overview (HIBA-heavy, biopsied); uncropped.",
                      "mcr_sl": "MCR-SL clinical close-ups (Norway; consensus labels, 29 biopsied; AK excluded); uncropped, may show several lesions.",
                      "ddi": "DDI (Stanford; biopsy-proven; balanced Fitzpatrick groups); image-level bootstrap (no patient id).",
                      "ddi2": "DDI-2 (Stanford; self-identified Asian patients; biopsy-proven); patient-level bootstrap.",
                      "midas": f"MRA-MIDAS {opts.midas_split} (splits/midas_v1.csv; patient-grouped; iPhone 6 in / 1 ft only; AK and re-excision melanocytic lesions excluded). Development data, not a locked test."}[opts.source]
    preds = df.drop(columns="path").copy()
    for run in opts.runs:
        p, epoch = score(run, images)
        thr, internal = internal_reference(UNC / "output" / run)
        d = pd.concat([df, p], axis=1)
        tag = run_tag(run)
        preds[[f"{tag}__{c}" for c in p.columns]] = p.values
        report["runs"][tag] = {"run": run, "best_epoch": epoch, "eoe_threshold": thr, "internal_test": internal,
                               "external": summarize(d, thr, opts.n_boot)}
        print(tag, json.dumps({k: v for k, v in report["runs"][tag]["external"].items() if k != "accuracy by diagnosis"}))
    tags = list(report["runs"])
    if len(tags) > 1:
        report["paired_vs_first"] = {t: paired_diff(preds, tags[0], t, opts.n_boot) for t in tags[1:]}
    preds.to_csv(out / "predictions.csv", index=False)
    (out / "external_report.json").write_text(json.dumps(report, indent=1))
    write_md(report, out / "external_report.md")
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--runs", nargs="+", required=True, help="run folders relative to uncertaintyNet-main/output")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--midas_split", default="dev_val", choices=["train", "dev_val", "dev_test"])
    ap.add_argument("--final", default="", help="required for locked sets: why this look is needed (logged)")
    main(ap.parse_args())
