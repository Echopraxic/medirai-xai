"""
LLM-as-judge for the explanations (plan T6.4, prereg H3), from a model family different from the generators.

The judge sees only the de-identified input JSON (llm_input) and the explanation, and labels each reason
(supported / direction_error / unsupported_concept / magnitude_error / other_error) plus explanation-level items:
uncertainty conveyed, overconfident wording, clarity, clinical tone, overall faithfulness. Greedy decoding with a
JSON schema, as in llm_generate.py.

Planted-error controls give the judge's recall per error type and its false-alarm rate on clean copies before any
human labels exist (same idea as nli_check.calibrate). Calibration against human ratings uses `sheet` (blinded
rating sheet) and, once the sheet is filled in, `agree` (judge vs human, Cohen's kappa per error type).

    python explain/llm_judge.py run --gens explain/output/d5_X/{template__template_v1,Qwen3-32B__hybrid_v1}/generations.jsonl \
        --judge google/gemma-3-27b-it --out explain/output/judge_X          # on a GPU node (slurm/narval_llm_judge.sh)
    python explain/llm_judge.py run ... --backend mock --limit 20            # CPU smoke test
    python explain/llm_judge.py sheet --gens ... --out explain/output/judge_X/human_ratings --n_per_cell 5
    python explain/llm_judge.py agree --sheet .../rating_sheet.csv --key .../rating_key.csv --judged .../judgments.jsonl
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verifier  # noqa: E402

JUDGE_PROMPT_VERSION = "judge_v1"
DECODING = {"temperature": 0.0, "max_new_tokens": 600, "seed": 0}
CLAIM_LABELS = ["supported", "direction_error", "unsupported_concept", "magnitude_error", "other_error"]
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "maxItems": 6, "items": {
            "type": "object",
            "properties": {"index": {"type": "integer"}, "label": {"type": "string", "enum": CLAIM_LABELS},
                           "note": {"type": "string", "maxLength": 160}},
            "required": ["index", "label", "note"]}},
        "uncertainty_conveyed": {"type": "string", "enum": ["yes", "no", "not_applicable"]},
        "overconfident": {"type": "boolean"},
        "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
        "clinical_tone": {"type": "integer", "minimum": 1, "maximum": 5},
        "overall_faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["claims", "uncertainty_conveyed", "overconfident", "clarity", "clinical_tone", "overall_faithfulness"],
}

SYSTEM = """You audit short explanations written for a dermatologist about a skin-lesion classifier's prediction.
Judge the EXPLANATION only against the INPUT it was written from. Do not use outside medical knowledge to decide
whether the prediction is right, and do not reward or punish the prediction itself.

For each numbered reason in the explanation, give one label:
- supported: it states a measure and finding that are in the INPUT evidence, with the same direction
  (supports / argues against the prediction) and a strength or comparison no stronger than the INPUT says.
- direction_error: it says the finding supports the prediction when the INPUT says it argues against it, or the reverse.
- unsupported_concept: it mentions a lesion property, structure or finding that is not in the INPUT evidence.
- magnitude_error: right finding and direction, but clearly overstates or understates the strength or comparison.
- other_error: any other mismatch with the INPUT (wrong measure, contradiction, a diagnosis or advice).

Explanation-level items:
- uncertainty_conveyed: if the INPUT has "uncertain": true, does the explanation clearly say the prediction is
  uncertain or needs review? Use not_applicable when the INPUT has "uncertain": false.
- overconfident: true if the wording sounds more certain than the INPUT confidence and caveats allow.
- clarity (1-5), clinical_tone (1-5, professional and neutral for a dermatologist), overall_faithfulness (1-5).
Answer with JSON only."""

USER = """INPUT:
{input_json}

EXPLANATION (reasons are numbered from 0):
{explanation}"""


def render_explanation(final):
    """Plain text the judge and the human raters read: summary, numbered reasons, notes."""
    if not isinstance(final, dict):
        return str(final)
    lines = [f"Summary: {final.get('summary', '')}"]
    for i, r in enumerate(final.get("reasons") or []):
        lines.append(f"[{i}] {r.get('text', '') if isinstance(r, dict) else r}")
    for key in ("uncertainty_note", "limitations"):
        if final.get(key):
            lines.append(f"{key.replace('_', ' ').capitalize()}: {final[key]}")
    return "\n".join(lines)


def judge_messages(llm_input, final):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(input_json=json.dumps(llm_input, indent=1),
                                                    explanation=render_explanation(final))}]


# ---------------------------------------------------------------- planted-error controls
FLIP = [("argues against", "supports"), ("arguing against", "supporting"), ("supports", "argues against"),
        ("supporting", "arguing against")]
INVENTED = "a blue-white veil and irregular streaks are visible within the lesion"
MAGNITUDE = [("weakly", "decisively"), ("moderately", "overwhelmingly"), ("strongly", "only marginally"),
             ("much more", "slightly more"), ("much less", "slightly less"), ("much lighter", "slightly lighter"),
             ("much darker", "slightly darker"), ("slightly more", "far more"), ("slightly less", "far less"),
             ("slightly", "dramatically")]


def _swap_first(text, pairs):
    for old, new in pairs:
        if old in text:
            return text.replace(old, new, 1)
    return None


def make_controls(rows, seed=0):
    """One clean copy plus, where applicable, one planted error per explanation. Returns (row, truth) pairs where
    truth = {"kind", "reason_index" or None}. Only explanations that pass the rule verifier are used as bases."""
    rng = random.Random(seed)
    out = []
    for row in rows:
        final = row["final_output"]
        if not (isinstance(final, dict) and row["verify"].get("pass_all") and final.get("reasons")):
            continue
        out.append((row, {"kind": "clean", "reason_index": None}))
        k = rng.randrange(len(final["reasons"]))
        kind = rng.choice(["direction_error", "unsupported_concept", "magnitude_error", "uncertainty_dropped"])
        bad = copy.deepcopy(final)
        reason = bad["reasons"][k]
        if kind == "direction_error":
            new = _swap_first(reason.get("text", ""), FLIP)
            if new is None:
                continue
            reason["text"] = new
        elif kind == "unsupported_concept":
            reason["text"] = f"{INVENTED}, which {'supports' if reason.get('direction') == 'supports' else 'argues against'} the {row['llm_input']['prediction']} prediction"
        elif kind == "magnitude_error":
            new = _swap_first(reason.get("text", ""), MAGNITUDE)
            if new is None:
                continue
            reason["text"] = new
        else:
            if not row["llm_input"].get("uncertain"):
                continue
            bad["uncertainty_note"] = ""
            bad["summary"] = f"The model predicts {row['llm_input']['prediction']} with high confidence."
            k = None
        out.append(({**row, "final_output": bad}, {"kind": kind, "reason_index": k}))
    return out


# ---------------------------------------------------------------- backends
class MockJudge:
    """Labels every claim supported; for CPU tests of the plumbing only."""
    name, revision = "mock", "n/a"

    def judge(self, conversations, n_reasons):
        return [json.dumps({"claims": [{"index": i, "label": "supported", "note": ""} for i in range(n)],
                            "uncertainty_conveyed": "not_applicable", "overconfident": False, "clarity": 4,
                            "clinical_tone": 4, "overall_faithfulness": 5}) for n in n_reasons]


class VLLMJudge:
    def __init__(self, model_id, revision=None):
        from llm_generate import vllm_chat, vllm_sampling
        from vllm import LLM
        extra = {"tensor_parallel_size": int(os.environ.get("VLLM_TP", "1"))}
        if "gemma-3" in model_id.lower():
            extra["limit_mm_per_prompt"] = {"image": 0}  # text-only use of a multimodal checkpoint
        dtype = "auto" if "awq" in model_id.lower() else "bfloat16"
        self.llm = LLM(model=model_id, revision=revision, seed=DECODING["seed"], dtype=dtype, max_model_len=4096,
                       **extra)
        self.params, self.guided = vllm_sampling(JUDGE_SCHEMA, DECODING["max_new_tokens"])
        self._chat = vllm_chat
        self.name, self.revision = model_id, revision or "see HF cache snapshot"

    def judge(self, conversations, n_reasons):
        return self._chat(self.llm, self.name, conversations, self.params)


# ---------------------------------------------------------------- scoring
def parse_judgment(raw, n_reasons):
    parsed, ok = verifier.parse(raw)
    if not ok or not isinstance(parsed, dict):
        return None
    labels = {c.get("index"): c.get("label") for c in parsed.get("claims", []) if isinstance(c, dict)}
    parsed["labels"] = [labels.get(i, "missing") for i in range(n_reasons)]
    return parsed


def control_hit(j, truth):
    """Did the judge find the planted error (or, for clean copies, raise no alarm)?"""
    if j is None:
        return False
    if truth["kind"] == "clean":
        return all(lab == "supported" for lab in j["labels"]) and not j.get("overconfident")
    if truth["kind"] == "uncertainty_dropped":
        return j.get("uncertainty_conveyed") == "no" or bool(j.get("overconfident"))
    lab = j["labels"][truth["reason_index"]]
    # an invented concept labelled other_error still counts as caught; the exact label is reported separately
    return lab != "supported" and lab != "missing"


def summarize_arm(items):
    labs = Counter(lab for it in items if it["judgment"] for lab in it["judgment"]["labels"])
    n_claims = sum(labs.values())
    ok = [it for it in items if it["judgment"]]
    by_stratum = defaultdict(list)
    for it in ok:
        by_stratum[it["stratum"]].append(bool(it["judgment"].get("overconfident")))
    unc = [it["judgment"].get("uncertainty_conveyed") == "yes" for it in ok if it["llm_input"].get("uncertain")]
    verif_dir_err = [it for it in ok if it["verify"].get("direction", 1.0) < 1.0]
    return {
        "n": len(items), "parse_ok": len(ok) / max(len(items), 1),
        "claim_labels": {k: v / max(n_claims, 1) for k, v in sorted(labs.items())},
        "uncertainty_conveyed_on_uncertain": sum(unc) / max(len(unc), 1),
        "overconfident_by_stratum": {s: sum(v) / len(v) for s, v in sorted(by_stratum.items())},
        "means": {k: sum(it["judgment"][k] for it in ok) / max(len(ok), 1)
                  for k in ("clarity", "clinical_tone", "overall_faithfulness")},
        "verifier_direction_errors_caught": (sum("direction_error" in it["judgment"]["labels"] for it in verif_dir_err)
                                             / len(verif_dir_err)) if verif_dir_err else None,
    }


def load_base(paths, limit=0):
    rows = []
    for p in paths:
        arm = Path(p).parent.name
        got = [dict(json.loads(line), arm=arm) for line in open(p, encoding="utf-8")]
        got = [r for r in got if r.get("variant", "base") == "base"]
        for r in got:  # the template arm stores its output as a JSON string
            if isinstance(r["final_output"], str):
                parsed, ok = verifier.parse(r["final_output"])
                r["final_output"] = parsed if ok else r["final_output"]
        rows += got[:limit] if limit else got
    return rows


def run(opts):
    rows = load_base(opts.gens, opts.limit)
    controls = [c for arm in opts.control_arms for c in make_controls([r for r in rows if r["arm"] == arm], seed=0)]
    jobs = [(r, None) for r in rows] + controls
    backend = MockJudge() if opts.backend == "mock" else VLLMJudge(opts.judge, opts.revision)
    out = Path(opts.out) / backend.name.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    t0, items = time.time(), []
    with open(out / "judgments.jsonl", "w", encoding="utf-8") as fh:
        for k in range(0, len(jobs), opts.batch_size):
            batch = jobs[k:k + opts.batch_size]
            n_reasons = [len((r["final_output"] or {}).get("reasons") or []) if isinstance(r["final_output"], dict) else 0
                         for r, _ in batch]
            raws = backend.judge([judge_messages(r["llm_input"], r["final_output"]) for r, _ in batch], n_reasons)
            for (r, truth), n, raw in zip(batch, n_reasons, raws):
                j = parse_judgment(raw, n)
                it = {"arm": r["arm"], "row_id": r["row_id"], "stratum": r["stratum"], "control": truth,
                      "llm_input": r["llm_input"], "verify": r["verify"], "judgment": j, "raw": raw}
                items.append(it)
                fh.write(json.dumps({k2: v for k2, v in it.items() if k2 != "llm_input"}) + "\n")
            print(f"{min(k + opts.batch_size, len(jobs))}/{len(jobs)}", flush=True)

    arms = sorted({it["arm"] for it in items if it["control"] is None})
    summary = {"arms": {a: summarize_arm([it for it in items if it["arm"] == a and it["control"] is None]) for a in arms}}
    ctrl = [it for it in items if it["control"] is not None]
    if ctrl:
        kinds = sorted({it["control"]["kind"] for it in ctrl})
        summary["controls"] = {
            "base_arms": opts.control_arms,
            **{kind: {"n": len(g), ("specificity" if kind == "clean" else "recall"):
                      sum(control_hit(it["judgment"], it["control"]) for it in g) / len(g),
                      "labels_on_planted": dict(Counter(it["judgment"]["labels"][it["control"]["reason_index"]]
                                                        for it in g if it["judgment"] and it["control"]["reason_index"] is not None))}
               for kind in kinds for g in [[it for it in ctrl if it["control"]["kind"] == kind]]}}
    manifest = {"judge": backend.name, "judge_revision": backend.revision, "backend": opts.backend,
                "guided_json": getattr(backend, "guided", False), "prompt_version": JUDGE_PROMPT_VERSION,
                "prompt_sha256_16": hashlib.sha256((SYSTEM + USER).encode()).hexdigest()[:16], "decoding": DECODING,
                "generations": [str(p) for p in opts.gens],
                "generations_sha256_16": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] for p in opts.gens},
                "n_items": len(items), "seconds": round(time.time() - t0, 1), "python": platform.python_version(),
                "vllm_tp": os.environ.get("VLLM_TP", "1")}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "judge_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- human ratings
SHEET_COLS = ["rating_id", "input", "explanation", "claim_labels (one per reason: S/D/U/M/O)", "uncertainty_conveyed (y/n/na)",
              "overconfident (y/n)", "clarity 1-5", "clinical_tone 1-5", "overall_faithfulness 1-5", "comment"]
LETTER = {"S": "supported", "D": "direction_error", "U": "unsupported_concept", "M": "magnitude_error", "O": "other_error"}


def render_input(inp):
    lines = [f"Prediction: {inp['prediction']} (confidence {inp['confidence']}); uncertain: {inp['uncertain']}"
             + (f" ({inp['uncertainty_reason']})" if inp.get("uncertainty_reason") else "")]
    for e in inp["evidence"]:
        lines.append(f"- {e['measure']}: {e['finding']}; {e.get('relative_to_benign', '')}; {e['direction']} ({e['strength']})")
    return "\n".join(lines)


def sheet(opts):
    """Blinded sheet: arm hidden, order shuffled (seed 0), n_per_cell cases per arm x stratum. First 30 rows = pilot."""
    rng = random.Random(0)
    rows = load_base(opts.gens)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["stratum"])].append(r)
    picked = [r for key in sorted(cells) for r in rng.sample(cells[key], min(opts.n_per_cell, len(cells[key])))]
    rng.shuffle(picked)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rating_sheet.csv", "w", newline="", encoding="utf-8") as fs, \
         open(out / "rating_key.csv", "w", newline="", encoding="utf-8") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(SHEET_COLS)
        wk.writerow(["rating_id", "arm", "row_id", "stratum", "n_reasons", "pilot"])
        for i, r in enumerate(picked):
            rid = f"R{i:03d}"
            ws.writerow([rid, render_input(r["llm_input"]), render_explanation(r["final_output"])] + [""] * 7)
            n = len(r["final_output"].get("reasons") or []) if isinstance(r["final_output"], dict) else 0
            wk.writerow([rid, r["arm"], r["row_id"], r["stratum"], n, int(i < 30)])
    print(f"{len(picked)} rows -> {out / 'rating_sheet.csv'} (key kept separate: rating_key.csv)")


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agree(opts):
    """Judge vs human, per error type: binary kappa and judge recall on human-labelled errors (prereg H3)."""
    key = {r["rating_id"]: r for r in csv.DictReader(open(opts.key, encoding="utf-8"))}
    judged = {(j["arm"], j["row_id"]): j for j in map(json.loads, open(opts.judged, encoding="utf-8"))
              if j["control"] is None and j["judgment"]}
    human, judge = [], []
    for r in csv.DictReader(open(opts.sheet, encoding="utf-8")):
        codes = r[SHEET_COLS[3]].replace(",", " ").split()
        k = key[r["rating_id"]]
        j = judged.get((k["arm"], k["row_id"]))
        if not codes or j is None:
            continue
        for i, c in enumerate(codes[:len(j["judgment"]["labels"])]):
            human.append(LETTER.get(c.upper(), "other_error"))
            judge.append(j["judgment"]["labels"][i])
    res = {"n_claims": len(human)}
    for lab in CLAIM_LABELS[1:]:
        h = [x == lab for x in human]
        g = [x == lab for x in judge]
        res[lab] = {"human_n": sum(h), "kappa": cohen_kappa(h, g),
                    "judge_recall": (sum(x and y for x, y in zip(h, g)) / sum(h)) if sum(h) else None}
    print(json.dumps(res, indent=1))
    if opts.out:
        Path(opts.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gens", nargs="+", required=True, help="generations.jsonl files (base variants are judged)")
    r.add_argument("--judge", default="google/gemma-3-27b-it")
    r.add_argument("--revision", default=None)
    r.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    r.add_argument("--control_arms", nargs="*", default=["template__template_v1", "Qwen3-32B__hybrid_v1"],
                   help="arms whose passing explanations get planted errors (none = no controls)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--batch_size", type=int, default=128)
    r.add_argument("--out", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--gens", nargs="+", required=True)
    s.add_argument("--n_per_cell", type=int, default=5)
    s.add_argument("--out", required=True)
    g = sub.add_parser("agree")
    g.add_argument("--sheet", required=True)
    g.add_argument("--key", required=True)
    g.add_argument("--judged", required=True)
    g.add_argument("--out", default="")
    opts = ap.parse_args()
    {"run": run, "sheet": sheet, "agree": agree}[opts.cmd](opts)
