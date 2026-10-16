"""
Vision-language model as a CONCEPT CHECKER (not an explainer) on Derm7pt (plan T3.3/T3.4).

Each Derm7pt case has a clinical and a dermoscopic photo with dermatologist labels for the 7-point criteria
and elevation. The VLM answers fixed multiple-choice questions about one photo; answers are scored against
the expert labels (concepts/eval_vlm_concepts.py, Cohen's kappa vs chance). It also answers A/B/C questions
(no expert labels in Derm7pt) that are cross-checked against our handcrafted features.

Prior MedirAI prompting lessons applied: deterministic decoding, JSON-schema constrained answers, option
order shuffled per case and per question (models favoured the first or majority option), no "unsure" option
(it was never chosen), no few-shot examples (they were copied).

    python concepts/vlm_concepts.py --model Qwen/Qwen2.5-VL-7B-Instruct --modality clinic \
        --derm7pt data/derm7pt/release_v0 --out concepts/output/vlm
Runs on Rorqual (vLLM, offline). Images are public research data; no patient metadata goes to the model.
"""
import argparse
import base64
import io
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain"))
from llm_generate import vllm_engine, vllm_sampling  # noqa: E402

PROMPT_VERSION = "vlm_concepts_v1"
COLORS = ["light brown", "dark brown", "black", "blue-gray", "red", "white"]
# question id -> (question, options). Option lists double as the JSON-schema enums.
ABC = {
    "asymmetry": ("Is the lesion's shape and color distribution symmetric?",
                  ["symmetric", "asymmetric in one axis", "asymmetric in two axes"]),
    "border": ("Is the lesion border regular or irregular?", ["regular", "irregular"]),
}
QUESTIONS = {
    "derm": {
        **ABC,
        "pigment_network": ("Pigment network:", ["absent", "typical", "atypical"]),
        "streaks": ("Streaks:", ["absent", "regular", "irregular"]),
        "pigmentation": ("Diffuse or localized pigmentation (blotches):", ["absent", "regular", "irregular"]),
        "regression_structures": ("Regression structures (white scar-like or blue-gray peppering areas):",
                                  ["absent", "present"]),
        "dots_and_globules": ("Dots and globules:", ["absent", "regular", "irregular"]),
        "blue_whitish_veil": ("Blue-whitish veil:", ["absent", "present"]),
        "vascular_structures": ("Vascular structures (vessels):", ["absent", "present"]),
    },
    "clinic": {
        **ABC,
        "pigmentation": ("Is the pigmentation regular or irregular?", ["absent", "regular", "irregular"]),
        "blue_whitish_veil": ("Is there a blue-whitish veil (blue-white, structureless area)?", ["absent", "present"]),
        "elevation": ("Is the lesion flat, palpable (slightly raised) or nodular?", ["flat", "palpable", "nodular"]),
    },
}


def schema(modality):
    props = {q: {"type": "string", "enum": opts} for q, (_, opts) in QUESTIONS[modality].items()}
    props["colors"] = {"type": "array", "items": {"type": "string", "enum": COLORS}, "maxItems": len(COLORS)}  # dedup at scoring (xgrammar has no uniqueItems)
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def question_text(modality, case_seed):
    rng = random.Random(case_seed)
    kind = "dermoscopic" if modality == "derm" else "clinical (non-dermoscopic)"
    lines = [f"You are a dermatologist. Look at this {kind} photograph of a single skin lesion and answer each "
             "question with exactly one of the listed options. Judge only what is visible."]
    for q, (text, opts) in QUESTIONS[modality].items():
        shuffled = opts[:]
        rng.shuffle(shuffled)
        lines.append(f'- {q}: {text} Options: {", ".join(shuffled)}')
    colors = COLORS[:]
    rng.shuffle(colors)
    lines.append(f'- colors: Which of these colors are present in the lesion (list all that apply): {", ".join(colors)}')
    lines.append("Answer with one JSON object with exactly these keys: " + ", ".join(list(QUESTIONS[modality]) + ["colors"]))
    return "\n".join(lines)


def find_image(root: Path, rel: str) -> Path:
    """Derm7pt's meta.csv and its image folders disagree on case (meta "FCl/Fcl068.jpg", folder "FCL"). That is
    invisible on Windows/macOS but fails on Linux, so match each path component case-insensitively."""
    path = root / rel
    if path.exists():
        return path
    cur = root
    for part in Path(rel).parts:
        nxt = cur / part
        if not nxt.exists():
            matches = [c for c in cur.iterdir() if c.name.lower() == part.lower()] if cur.is_dir() else []
            if not matches:
                raise FileNotFoundError(path)
            nxt = matches[0]
        cur = nxt
    return cur


def data_uri(path, max_side):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main(opts):
    root = Path(opts.derm7pt)
    meta = pd.read_csv(root / "meta" / "meta.csv")
    if opts.limit:
        meta = meta.head(opts.limit)
    llm = vllm_engine(opts.model, max_model_len=8192, limit_mm_per_prompt={"image": 1},
                      tensor_parallel_size=opts.tp)
    params, guided = vllm_sampling(schema(opts.modality), max_tokens=400)
    out = Path(opts.out) / f"{opts.model.split('/')[-1]}__{opts.modality}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    answers_path = out / "answers.jsonl"
    done = set()
    if answers_path.exists():  # resume: keep the cases already answered (decoding is deterministic)
        done = {json.loads(line)["case_num"] for line in answers_path.read_text(encoding="utf-8").splitlines() if line}
    todo = meta[~meta.case_num.isin(done)]
    print(f"{len(done)} cases already answered, {len(todo)} to go", flush=True)
    with open(answers_path, "a", encoding="utf-8") as fh:
        for k in range(0, len(todo), opts.batch_size):
            batch = todo.iloc[k:k + opts.batch_size]
            convs = [[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri(find_image(root / "images", r[opts.modality]), opts.max_side)}},
                {"type": "text", "text": question_text(opts.modality, int(r.case_num))}]}]
                for _, r in batch.iterrows()]
            outs = llm.chat(convs, params, use_tqdm=False)
            for (_, r), o in zip(batch.iterrows(), outs):
                text = o.outputs[0].text
                try:
                    answers = json.loads(text)
                except json.JSONDecodeError:
                    answers = None
                fh.write(json.dumps({"case_num": int(r.case_num), "answers": answers, "raw": text}) + "\n")
            print(f"{len(done) + min(k + opts.batch_size, len(todo))}/{len(meta)}", flush=True)
    (out / "run_manifest.json").write_text(json.dumps({
        "model": opts.model, "modality": opts.modality, "prompt_version": PROMPT_VERSION, "guided_json": guided,
        "decoding": {"temperature": 0.0, "seed": 0, "max_tokens": 400}, "max_side": opts.max_side,
        "tensor_parallel": opts.tp, "n": len(meta), "resumed_from": len(done), "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modality", choices=["clinic", "derm"], required=True)
    ap.add_argument("--derm7pt", default=str(ROOT / "data/derm7pt/release_v0"))
    ap.add_argument("--out", default=str(ROOT / "concepts/output/vlm"))
    ap.add_argument("--max_side", type=int, default=896)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1, help="GPUs to split the model over (32B on 40 GB A100s needs 2)")
    main(ap.parse_args())
