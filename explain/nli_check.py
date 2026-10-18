"""
Semantic claim check with natural-language inference (plan T6.1; makes criterion C3 semantic, not only structural).

The rule-based verifier checks that each reason cites a supplied feature_id and states the right direction. It
cannot tell whether the sentence actually SAYS what the evidence says ("border outline is irregular" vs a sentence
that cites the border item but calls it regular). Here each claim is the NLI hypothesis and the evidence item it
cites, written as a canonical sentence, is the premise:

  premise     "<Measure> is <finding> (<relative_to_benign>); this <strength>ly <supports|argues against> the
               <prediction> prediction."          (the template wording: faithful by construction)
  hypothesis  the generated reason text
  entailed    P(entailment) >= ENTAIL_MIN;  contradicted: P(contradiction) >= CONTRA_MIN

The check is itself validated before use (RQ1): `calibrate` scores the faithful template claims (should be
entailed) and three corruptions of each (should not be): the finding word replaced by its opposite, the direction
reversed, and the measure replaced by another item's measure. Report its detection rate and false-alarm rate.

    python explain/nli_check.py calibrate --inputs trees/output_v1.0/explanation_inputs.jsonl --limit 300
    python explain/nli_check.py score explain/output/<run>/<model>__<prompt>/generations.jsonl

Default model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (MIT). Runs locally; only the generated
text and the structured evidence (no images, metadata or identifiers) are scored.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

import llm_input as li
from template import PHRASES

MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ENTAIL_MIN = 0.5
CONTRA_MIN = 0.5


def premise(item, prediction):
    verb = "supports" if item["direction"] == "supports" else "argues against"
    return (f"{item['measure'].capitalize()} is {item['finding']} ({item['relative_to_benign']}); this "
            f"{item['strength']}ly {verb} the {prediction} prediction.")


class NLIScorer:
    """Callable: list of (premise, hypothesis) -> array [n, 3] of (entailment, neutral, contradiction) probs."""

    def __init__(self, model=MODEL, device=None, batch_size=32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.order = [labels["entailment"], labels["neutral"], labels["contradiction"]]
        self.batch_size = batch_size
        self.name = model

    def __call__(self, pairs):
        out = []
        for k in range(0, len(pairs), self.batch_size):
            p, h = zip(*pairs[k:k + self.batch_size])
            enc = self.tok(list(p), list(h), return_tensors="pt", padding=True, truncation=True).to(self.device)
            with self.torch.no_grad():
                probs = self.model(**enc).logits.float().softmax(-1).cpu().numpy()
            out.append(probs[:, self.order])
        return np.concatenate(out) if out else np.zeros((0, 3))


def judge(probs):
    return {"entailed": bool(probs[0] >= ENTAIL_MIN), "contradicted": bool(probs[2] >= CONTRA_MIN),
            "p_entail": round(float(probs[0]), 4), "p_contra": round(float(probs[2]), 4)}


def claim_pairs(gen):
    """(premise, hypothesis, feature_id) for every reason that cites a supplied item."""
    inp = gen["llm_input"]
    items = {e["feature_id"]: e for e in inp["evidence"]}
    out = gen["final_output"]
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except json.JSONDecodeError:
            out = {}
    pairs = []
    for r in (out.get("reasons") or []) if isinstance(out, dict) else []:
        if isinstance(r, dict) and r.get("feature_id") in items:
            pairs.append((premise(items[r["feature_id"]], inp["prediction"]), r.get("text", ""), r["feature_id"]))
    return pairs


def score(path, scorer):
    gens = [json.loads(line) for line in open(path, encoding="utf-8")]
    meta = [(g, p) for g in gens for p in claim_pairs(g)]
    probs = scorer([(p, h) for _, (p, h, _) in meta])
    per_claim = [{"row_id": g.get("row_id", g["isic_id"]), "stratum": g.get("stratum"), "feature_id": f, "text": h,
                  **judge(pr)} for (g, (p, h, f)), pr in zip(meta, probs)]
    n_cited = len(per_claim)
    report = {"generations": str(path), "n_claims": n_cited,
              "entailed_rate": float(np.mean([c["entailed"] for c in per_claim])) if per_claim else None,
              "contradicted_rate": float(np.mean([c["contradicted"] for c in per_claim])) if per_claim else None,
              "by_stratum": {}}
    for s in sorted({c["stratum"] for c in per_claim if c["stratum"]}):
        cs = [c for c in per_claim if c["stratum"] == s]
        report["by_stratum"][s] = {"n": len(cs), "entailed_rate": float(np.mean([c["entailed"] for c in cs])),
                                   "contradicted_rate": float(np.mean([c["contradicted"] for c in cs]))}
    out = Path(path).with_name("nli_scores.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with open(Path(path).with_name("nli_claims.jsonl"), "w", encoding="utf-8") as fh:
        for c in per_claim:
            fh.write(json.dumps(c) + "\n")
    return report


# ---------------------------------------------------------------------------------------- calibration
def corruptions(item, prediction, other_measure):
    """Unfaithful versions of the faithful template claim for `item`."""
    _, high, low = PHRASES.get(item["feature_id"], (None, None, None))
    out = {}
    if high and low and item["finding"] in (high, low):
        opposite = low if item["finding"] == high else high
        out["finding_flipped"] = premise(dict(item, finding=opposite), prediction)
    flipped = "argues_against" if item["direction"] == "supports" else "supports"
    out["direction_flipped"] = premise(dict(item, direction=flipped), prediction)
    if other_measure and other_measure != item["measure"]:
        out["measure_swapped"] = premise(dict(item, measure=other_measure), prediction)
    return out


def calibrate(inputs, scorer, limit=300, seed=0):
    rows = [json.loads(line) for line in open(inputs, encoding="utf-8")]
    threshold = li.entropy_threshold_from_val(rows)
    test = [r for r in rows if r["split"] == "test"]
    rng = random.Random(seed)
    rng.shuffle(test)
    pairs, kinds = [], []
    for r in test[:limit]:
        inp = li.build(r["llm_input"], r["mask_quality"], threshold)
        mapped = [e for e in inp["evidence"] if e["clinically_mapped"]]
        for e in mapped:
            prem = premise(e, inp["prediction"])
            others = [o["measure"] for o in mapped if o["measure"] != e["measure"]]
            pairs.append((prem, prem.replace(" (", ", ", 1).replace(");", ",", 1)))  # faithful, lightly reworded
            kinds.append("faithful")
            for kind, hyp in corruptions(e, inp["prediction"], others[0] if others else None).items():
                pairs.append((prem, hyp))
                kinds.append(kind)
    probs = scorer(pairs)
    report = {"model": getattr(scorer, "name", "custom"), "entail_min": ENTAIL_MIN, "n_pairs": len(pairs)}
    for kind in sorted(set(kinds)):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        ent = probs[idx, 0] >= ENTAIL_MIN
        report[kind] = {"n": len(idx), "entailed_rate": float(ent.mean()),
                        "contradicted_rate": float((probs[idx, 2] >= CONTRA_MIN).mean())}
    report["false_alarm_rate"] = 1 - report["faithful"]["entailed_rate"]
    report["detection_rate"] = {k: 1 - v["entailed_rate"] for k, v in report.items()
                                if isinstance(v, dict) and k != "faithful"}
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--inputs", required=True)
    c.add_argument("--limit", type=int, default=300)
    c.add_argument("--out", default="explain/output/nli_calibration.json")
    c.add_argument("--model", default=MODEL)
    s = sub.add_parser("score")
    s.add_argument("generations", nargs="+")
    s.add_argument("--model", default=MODEL)
    opts = ap.parse_args()
    nli = NLIScorer(opts.model)
    if opts.cmd == "calibrate":
        rep = calibrate(opts.inputs, nli, opts.limit)
        Path(opts.out).parent.mkdir(parents=True, exist_ok=True)
        Path(opts.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    else:
        rep = [score(g, nli) for g in opts.generations]
    print(json.dumps(rep, indent=1))
