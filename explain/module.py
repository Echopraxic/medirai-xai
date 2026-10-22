"""
Explanation module (plan T8.1; D4 prototype, D7 integration target): one call per prediction.

    ex = Explainer.from_dirs("trees/output_v1.0_detv2", "features/output/feature_table_v1.0.parquet")
    out = ex.explain(features={...}, prediction={"p_malignant": 0.91, "entropy_of_expected": 0.2},
                     mask_quality="ok", audience="dermatologist", language="en")
    out["text"]          display text
    out["structured"]    {summary, reasons[{feature_id, direction, text}], uncertainty_note, limitations}
    out["top_concepts"]  TreeSHAP top-3 concept groups (feature_id, direction, strength, benign percentile)
    out["verification"]  verifier result of what is returned; out["fallback_used"] says whether the LLM text failed
    out["provenance"]    backend/model, prompt version + hash, input/phrase-bank versions, input hash, trees folder

Pipeline: feature values -> binary surrogate TreeSHAP -> concept-grouped top-3 (trees/treeshap_top3.case_reasons) ->
number-free LLM input (llm_input.build) -> reasons written by the LLM (hybrid prompt; summary, caveats and
limitations built in code) -> rule verifier. Only the number-free input reaches the LLM: no image, metadata or id.

Safety: an LLM explanation that fails any verifier rule is replaced by the template explanation of the same input
(fallback_used = the failed rules), so the reader never sees an ungrounded claim.
Audience (MedirAI, M-5: dermatologists first, GPs second): the reasons are the same for both; for
"general_practitioner" a code-built plain-language glossary of the cited concepts is appended (GLOSSARY, DRAFT
until the clinical reviewer approves it). Language: "en", or "fr" (template layer explain/lang_fr.py, DRAFT; the LLM
is not used for French until the reviewer approves the French phrase bank).
Explanation features (Explainer.from_dirs(..., explanation_run=<run>); off by default; review 2026-10-21): reasons
citing a feature the ResNet does not encode are labelled; out-of-scope images (DINOv2 space) get a warning; the
named-concept share sentence appears only when named features explain little; the similar-case panel (DINOv2 space,
in-scope only, outside the verified text) is for the clinician study arm, never the product.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "trees"))

import llm_input as li  # noqa: E402
import verifier  # noqa: E402
from lang_fr import NBSP, render_llm_input_fr  # noqa: E402
from llm_generate import TemplateBackend, assemble, load_prompt, messages  # noqa: E402
from template import render_llm_input  # noqa: E402

MODULE_VERSION = "explain_module_v1"
AUDIENCES = ("dermatologist", "general_practitioner")
GLOSSARY_VERSION = "gp_glossary_v1_draft"
GLOSSARY = {  # concept group -> (en, fr); DRAFT, for the clinical reviewer (M-6). No diagnosis names, no advice.
    "asymmetry": ("Asymmetry: how much the two halves of the lesion differ in outline or colour; it is one of the "
                  "ABCD criteria of skin-lesion assessment.",
                  "Asymétrie : degré de différence entre les deux moitiés de la lésion, par le contour ou la couleur; "
                  "c'est l'un des critères ABCD de l'évaluation des lésions cutanées."),
    "border": ("Border: how regular and well defined the edge of the lesion is; it is one of the ABCD criteria.",
               "Bordure : régularité et netteté du bord de la lésion; c'est l'un des critères ABCD."),
    "color": ("Colour: the shades inside the lesion compared with the surrounding skin; it is one of the ABCD "
              "criteria.",
              "Couleur : les teintes à l'intérieur de la lésion comparées à la peau environnante; c'est l'un des "
              "critères ABCD."),
    "texture/surface": ("Texture: how uniform the surface of the lesion looks in the photograph; it describes the "
                        "image, not a clinical sign.",
                        "Texture : uniformité de la surface de la lésion sur la photo; elle décrit l'image, pas un "
                        "signe clinique."),
    "shape": ("Shape pattern: a mathematical description of the outline with no direct clinical equivalent.",
              "Motif de forme : une description mathématique du contour, sans équivalent clinique direct."),
}
HEADINGS = {"en": ("Reasons", "Limitations", "What these terms mean", "Similar biopsied cases from the training data"),
            "fr": ("Raisons", "Limites", "Définitions", "Cas biopsiés semblables dans les données d'entraînement")}
COUNT_WORDS = {"en": ["none", "one", "two", "three", "four", "five"], "fr": ["aucun", "un", "deux", "trois", "quatre", "cinq"]}


class Explainer:
    def __init__(self, surrogate, gt_model, feats, benign_ref, strength_thresholds, entropy_threshold,
                 backend=None, prompt="hybrid_v1", trees_dir=None, extras=None):
        import shap
        self.surrogate, self.gt_model, self.feats = surrogate, gt_model, feats
        self.shap = shap.TreeExplainer(surrogate, feature_perturbation="tree_path_dependent")
        self.ref, self.thresholds, self.entropy_threshold = benign_ref, tuple(strength_thresholds), entropy_threshold
        self.backend = backend or TemplateBackend()
        self.spec = ({"version": "template_v1", "sha256_16": "n/a"} if isinstance(self.backend, TemplateBackend)
                     else load_prompt(prompt))
        self.trees_dir = str(trees_dir) if trees_dir else None
        self.extras = extras  # interpret.explanation_features.ExplanationFeatures, or None (features off)

    @classmethod
    def from_dirs(cls, trees_dir, features_table, backend=None, prompt="hybrid_v1", entropy_threshold=None,
                  explanation_run=None):
        """Load the binary surrogate + ground-truth tree, validation strength tertiles and the benign reference.
        entropy_threshold defaults to the validation 75th percentile stored with the trees' explanation inputs."""
        import xgboost as xgb
        trees = ROOT / trees_dir
        feats = json.loads((trees / "features_used.json").read_text())
        models = []
        for name in ("binary_surrogate", "binary_gt"):
            m = xgb.XGBClassifier()
            m.load_model(trees / f"{name}.json")
            models.append(m)
        table = pd.read_parquet(ROOT / features_table)
        benign = table[(table.split == "train") & (table.label == 0)]
        ref = {f: np.sort(benign[f].dropna().to_numpy()) for f in feats}
        thresholds = json.loads((trees / "treeshap_summary.json").read_text())["strength_thresholds_val"]
        if entropy_threshold is None:
            rows = [json.loads(line) for line in open(trees / "explanation_inputs.jsonl", encoding="utf-8")]
            entropy_threshold = li.entropy_threshold_from_val(rows)
        extras = None
        if explanation_run:  # artifacts from interpret/explanation_features.py fit for the explained model
            sys.path.insert(0, str(ROOT / "interpret"))
            from explanation_features import ExplanationFeatures
            extras = ExplanationFeatures(explanation_run)
        return cls(models[0], models[1], feats, ref, thresholds, entropy_threshold, backend, prompt, trees_dir, extras)

    def payload(self, features, prediction):
        """One case -> the structured payload treeshap_top3 writes (numbers included; never shown to the LLM)."""
        from treeshap_top3 import case_reasons, grouped
        x = pd.DataFrame([{f: features[f] for f in self.feats}])
        pred_malignant = prediction["p_malignant"] >= 0.5
        sign = 1.0 if pred_malignant else -1.0
        signed = pd.DataFrame(self.shap.shap_values(x), columns=self.feats) * sign
        g = grouped(signed, self.feats)
        not_encoded = self.extras.not_cited if self.extras else None   # labelled, not dropped (decision 2026-10-21)
        reasons = case_reasons(g.iloc[0], signed.iloc[0], x.iloc[0], self.feats, self.ref, self.thresholds,
                               not_encoded=not_encoded)
        p_sur = float(self.surrogate.predict_proba(x)[0, 1])
        p_gt = float(self.gt_model.predict_proba(x)[0, 1])
        uncertainty = {"entropy_of_expected": float(prediction["entropy_of_expected"])}
        if prediction.get("deferred") is not None:
            uncertainty["deferred"] = bool(prediction["deferred"])
        return {"prediction": "malignant" if pred_malignant else "benign",
                "probability_malignant": round(float(prediction["p_malignant"]), 3), "uncertainty": uncertainty,
                "surrogate_agrees": (p_sur >= 0.5) == pred_malignant,
                "feature_model_p_predicted": round(p_gt if pred_malignant else 1 - p_gt, 3),
                "top_concepts": reasons, "audience": "dermatologist"}

    def explain(self, features, prediction, mask_quality, audience="dermatologist", language="en", case=None,
                case_panel=False):
        """case (optional, used only when explanation features are on): {"concepts": handcrafted + MONET values,
        "dinov2_emb": DINOv2-L CLS embedding}. case_panel: similar training cases in DINOv2 space, for the clinician
        study arm only (never in the product: the review found panels reinforce most wrong predictions)."""
        if audience not in AUDIENCES:
            raise ValueError(f"audience must be one of {AUDIENCES}")
        if language not in HEADINGS:
            raise ValueError("language must be 'en' or 'fr'")
        payload = self.payload(features, prediction)
        payload["audience"] = audience.replace("_", " ")
        similar, extra_prov = [], {}
        if self.extras and case:
            if case.get("dinov2_emb") is not None:
                payload["in_scope"], extra_prov["scope_distance"] = self.extras.in_scope(case["dinov2_emb"])
                payload["tone_unvalidated"], extra_prov["darker_skin_probability"] = self.extras.tone_scope(case["dinov2_emb"])
            if case.get("concepts") is not None:
                extra_prov["concept_share"], band = self.extras.concept_share(case["concepts"], prediction["p_malignant"])
                extra_prov["concept_share_band"] = band
                if band == "little":   # shown only when named features explain little (decision 2026-10-21)
                    payload["named_concept_share"] = band
            if case_panel and case.get("dinov2_emb") is not None and payload.get("in_scope", True):
                similar = self.extras.similar_cases(case["dinov2_emb"], space="dinov2")
        inp = li.build(payload, mask_quality, self.entropy_threshold)
        fallback = None
        if language == "fr":
            structured, backend_name = render_llm_input_fr(inp), "template_fr"
        else:
            msgs = [None] if isinstance(self.backend, TemplateBackend) else [messages(self.spec, inp)]
            raw = self.backend.generate(msgs, [inp])[0]
            structured, backend_name = verifier.parse(assemble(self.spec, inp, raw))[0], self.backend.name
            check = verifier.verify(inp, structured or "")
            if not check["pass_all"]:
                fallback = verifier.failed_rules(check)
                structured, backend_name = render_llm_input(inp), "template (fallback)"
        check = verifier.verify(inp, structured) if language == "en" else None
        concepts = []
        for r in payload["top_concepts"]:
            if r["concept"] not in concepts:
                concepts.append(r["concept"])
        glossary = ([GLOSSARY[c][0 if language == "en" else 1] for c in concepts if c in GLOSSARY]
                    if audience == "general_practitioner" else [])
        return {"text": to_text(structured, glossary, language, similar), "structured": structured,
                "glossary": glossary, "similar_cases": similar,
                "top_concepts": payload["top_concepts"], "llm_input": inp, "verification": check,
                "fallback_used": fallback,
                "provenance": {"module": MODULE_VERSION, "backend": backend_name,
                               "model_revision": getattr(self.backend, "revision", "n/a"),
                               "prompt_version": self.spec["version"], "prompt_sha256_16": self.spec["sha256_16"],
                               "llm_input_version": li.LLM_INPUT_VERSION, "phrase_bank": li.PHRASE_BANK_VERSION,
                               "glossary": GLOSSARY_VERSION if glossary else None, "audience": audience,
                               "language": language, "input_hash": li.input_hash(inp), "trees": self.trees_dir,
                               "entropy_threshold": self.entropy_threshold,
                               "explanation_features": bool(self.extras), **extra_prov}}


def similar_cases_text(similar, language="en"):
    """Code-built panel (outside the verified explanation: it names diagnoses). Counts in words, no digits."""
    words = COUNT_WORDS[language]
    n_mal = sum(c["label"] for c in similar)
    if language == "fr":
        head = f"{words[n_mal].capitalize()} sur {words[len(similar)]} étaient malins."
    else:
        head = f"{words[n_mal].capitalize()} of {words[len(similar)]} were malignant."
    return head + " " + "; ".join(c["diagnosis"] or "diagnosis not recorded" for c in similar) + "."


def to_text(structured, glossary=(), language="en", similar=()):
    reasons_h, limits_h, gloss_h, similar_h = HEADINGS[language]
    colon = NBSP + ":" if language == "fr" else ":"  # OQLF: non-breaking space before the colon
    parts = [structured["summary"]]
    if structured.get("uncertainty_note"):
        parts.append(structured["uncertainty_note"])
    parts.append(f"{reasons_h}{colon}\n" + "\n".join(f"- {r['text']}" for r in structured["reasons"]))
    parts.append(f"{limits_h}{colon} {structured['limitations']}")
    if similar:
        parts.append(f"{similar_h}{colon} {similar_cases_text(similar, language)}")
    if glossary:
        parts.append(f"{gloss_h}{colon}\n" + "\n".join(f"- {g}" for g in glossary))
    return "\n\n".join(parts)
