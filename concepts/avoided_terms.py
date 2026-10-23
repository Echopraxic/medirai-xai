"""
Avoided terms v1 (English + Quebec French) and the bilingual clinical-review package (plan T3.5, M-6).

Explanations must not use these words: our features failed to track them (MONET / Derm7pt checks), they cannot be
assessed from one clinical photograph, or they would overstate what the model does (diagnosis names, advice,
certainty). Each entry gives the reason, the evidence, what to say instead, and regex stems for both languages.
The English stems for diagnoses, advice and certainty come from explain/verifier.py, so the list and the checker
agree; the French stems are for a French verifier (not built yet). French wording is a DRAFT for the reviewer.

    python concepts/avoided_terms.py   # -> concepts/output/{avoided_terms_v1_en_fr.csv, phrase_bank_v1_en_fr.csv,
                                       #    clinical_review_v1_en_fr.md}
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain"))
import lang_fr  # noqa: E402
import verifier  # noqa: E402

VERSION = "avoided_terms_v1_en_fr_draft"
MONET = "MILK10k MONET scores vs our features (clinical photos)"

# (category, term EN, term FR, reason, evidence, say instead EN, say instead FR, stems EN, stems FR)
AVOIDED = [
    ("failed validation", "erythema", "érythème",
     "Our redness features do not track erythema on clinical photos.",
     f"{MONET}: erythema vs red tone rho 0.07, vs red-area share 0.12",
     "red tone relative to surrounding skin", "teinte rouge par rapport à la peau environnante",
     ["erythem"], ["érythèm", "erythem"]),
    ("failed validation", "vessels / vascular structures / telangiectasia",
     "vaisseaux / structures vasculaires / télangiectasies",
     "Red-area features do not track vessels; vascular patterns need dermoscopy.",
     f"{MONET}: vessels vs red-area share 0.16, vs red tone -0.02; Derm7pt vascular structures (VLM) kappa <= 0.40",
     "share of red-toned areas", "proportion de zones de teinte rouge",
     ["vessel", "vascular", "telangiect", "arboriz"], ["vaisseau", "vasculaire", "télangiect", "arboris"]),
    ("failed validation", "ulceration / crust", "ulcération / croûte",
     "Texture and yellow-tone features do not track ulceration or crust.",
     f"{MONET}: ulceration/crust vs roughness 0.14, vs texture contrast 0.14, vs yellow tone -0.07",
     "fine-scale image texture; yellow tone relative to surrounding skin",
     "texture de l'image à petite échelle; teinte jaune par rapport à la peau environnante",
     ["ulcer", "crust"], ["ulcér", "croût"]),
    ("failed validation", "scale / scaling", "squames / desquamation",
     "No feature was shown to measure scale; the texture proxies failed the crust check.",
     "Untested; texture proxies fail the MONET crust check",
     "image texture", "texture de l'image", [r"(?<!fine-)\bscal(e|es|ing|y)\b", "keratin"], ["squam", "kérat"]),
    ("dermoscopic structure", "pigment network, dots/globules, streaks/pseudopods, regression structures",
     "réseau pigmentaire, points/globules, stries/pseudopodes, structures de régression",
     "Dermoscopic structures are not measured from clinical photos.",
     "Derm7pt dermoscopic criteria, Qwen2.5-VL checker: kappa <= 0.16; no feature targets them",
     "(do not mention)", "(ne pas mentionner)",
     ["pigment network", "network", "globul", r"\bdots?\b", "streak", "pseudopod", "regression"],
     ["réseau", "globul", r"\bpoints?\b", "strie", "pseudopode", "régression"]),
    ("partial validation", "blue-white veil", "voile bleu-blanc",
     "Our blue-gray share separates the veil only partly (AUROC 0.70), and the veil is a dermoscopic sign.",
     "Derm7pt expert labels (clinical photos): blue-whitish veil vs blue-gray share AUROC 0.70",
     "share of blue-gray areas", "proportion de zones gris-bleu", ["veil", "blue-white"], ["voile", "bleu-blanc"]),
    ("not validated", "number of colours", "nombre de couleurs",
     "The colour count could not be validated (ENHANCE colour raters disagree); PH2/Derm7pt pending.",
     "ENHANCE colour labels unusable (raters agree below the usability threshold)",
     "color variety", "variété des couleurs",
     [r"\b(two|three|four|five|six|\d+) colou?rs?\b", "number of colou?rs"],
     [r"\b(deux|trois|quatre|cinq|six|\d+) couleurs\b", "nombre de couleurs"]),
    ("not measurable from one image", "diameter / size in mm / large / small", "diamètre / taille en mm / grande / petite",
     "No millimetre calibration on these photos; pixel size depends on framing.",
     "No scale reference in ISIC clinical close-ups; pixel diameter is a framing shortcut",
     "state the limitation: size cannot be given in millimetres",
     "indiquer la limite : la taille ne peut pas être donnée en millimètres",
     ["diameter", "millimet", r"\bmm\b", r"\blarge\b", r"\bsmall\b"],
     ["diamètre", "millimètre", r"\bmm\b", r"\bgrande?\b(?! (axe|échelle))", r"\bpetite?\b(?! (axe|échelle))"]),
    ("not measurable from one image", "evolution / growth / change over time", "évolution / croissance / changement",
     "A single image cannot show change over time (the E of ABCDE).", "Single time point",
     "state the limitation: evolution cannot be assessed", "indiquer la limite : l'évolution ne peut pas être évaluée",
     ["evolv", "evolution", "growth", "growing", "changing"], ["évolu", "croissance", r"grossi(t|ssement)", "changement"]),
    ("not measurable from one image", "raised / elevated / nodule / papule / plaque",
     "surélevée / en relief / nodule / papule / plaque",
     "Elevation is not measured; the texture proxy only partly tracks it (AUROC 0.64).",
     "Derm7pt elevation vs roughness AUROC 0.64 (partial); VLM elevation kappa 0.58 is not one of our features",
     "fine-scale image texture", "texture de l'image à petite échelle",
     ["elevat", "raised", "nodul", "papul", "plaque"], ["surélev", "en relief", "nodul", "papul", "plaque"]),
    ("not in the data", "symptoms (itching, bleeding)", "symptômes (démangeaisons, saignement)",
     "The model sees only the image.", "No symptom data", "(do not mention)", "(ne pas mentionner)",
     ["itch", "bleed"], ["démange", "prurit", "saign"]),
    ("overstates the model", "named diagnoses", "diagnostics nommés",
     "The model predicts benign vs malignant only; naming a diagnosis invents a claim.",
     "Binary classifier; verifier rule `named_diagnosis`",
     "benign / malignant prediction", "prédiction bénigne / maligne",
     verifier.DIAGNOSES,
     ["mélanome", "carcinome", "cbc", "csc", "naevus", "nævus", "kératose", "lentigo", "dermatofibrome", "angiome",
      "actinique", "bowen", "kératoacanthome", "grain de beauté"]),
    ("overstates the model", "management advice", "conseils de prise en charge",
     "Explanations describe the model, not what to do; decisions stay with the clinician.",
     "Verifier rule `advice`", "(do not mention)", "(ne pas mentionner)",
     verifier.ADVICE, ["excis", "biops", "retir", "trait", "crème", "chirurg", "suivi dans", "diriger vers", "orienter vers",
                       "prescri", "surveill"]),
    ("overstates the model", "certainty wording", "formulations de certitude",
     "Confident wording on uncertain or wrong predictions is the false-confidence risk (C6).",
     "Verifier rule `confident_terms`; C6", "the model's confidence category (high / moderate / low)",
     "la catégorie de confiance du modèle (élevée / modérée / faible)",
     verifier.CONFIDENT, ["clairement", "certainement", "assurément", "sans aucun doute", "confirm", "concluant",
                          "évident", "fortement évocat", "typique d'une lésion maligne", "caractéristique de"]),
    ("language (French only)", "—", "« supporte » (anglicisme)",
     "OQLF: 'supporter' in the sense of 'support' is an anglicism.", "OQLF Banque de dépannage linguistique",
     "—", "appuie / va à l'encontre de", [], [r"\bsupport"]),
]
COLUMNS = ["category", "term_en", "term_fr", "reason", "evidence", "say_instead_en", "say_instead_fr",
           "stems_en", "stems_fr"]


def strip_limitations(text):
    """The fixed limitation sentences may name size/millimetres/evolution to say they cannot be assessed."""
    from llm_input import LIMITATIONS
    for sentence in LIMITATIONS + lang_fr.LIMITATIONS_FR:
        text = text.replace(sentence, " ")
    return text


def hits(text, lang="en"):
    """Avoided terms found in an explanation (fixed limitation sentences excluded). lang: 'en' or 'fr'."""
    text, col, name = strip_limitations(text), (7 if lang == "en" else 8), (1 if lang == "en" else 2)
    return sorted({a[name] for a in AVOIDED for stem in a[col] if re.search(stem, text, re.I)})


def avoided_table():
    rows = [dict(zip(COLUMNS, a)) for a in AVOIDED]
    for r in rows:
        r["stems_en"], r["stems_fr"] = " | ".join(r["stems_en"]), " | ".join(r["stems_fr"])
        r["fr_status"] = "draft: to approve"
        r["version"] = VERSION
    return pd.DataFrame(rows)


def bilingual_phrase_bank():
    pb = pd.read_csv(ROOT / "concepts/output/phrase_bank_v1.csv")

    def fr(fid):
        if fid.startswith("S_hu"):
            return f"{lang_fr.UNMAPPED_SHAPE} : '{lang_fr.TYPICAL}' / '{lang_fr.UNUSUAL}'"
        m, hi, lo = lang_fr.PHRASES_FR[fid]
        return f"{m} : '{hi}' (élevé) / '{lo}' (faible)"
    pb["explanation_wording_fr"] = pb.feature_id.map(fr)
    pb.loc[pb.shown_in_explanations.str.startswith("no"), "explanation_wording_fr"] = "—"
    pb["fr_status"] = "draft: to approve"
    pb["fr_version"] = lang_fr.FR_VERSION
    return pb


def review_doc(pb, av):
    shown = pb[pb.explanation_wording != "—"]
    lines = [
        "# Clinical vocabulary review / Révision du vocabulaire clinique (v1, EN + FR-QC)",
        "",
        "For the clinical reviewer (role 1 in the dermatologist request). Please mark each row **OK**, **change "
        "(suggestion)** or **remove**. French wording is a draft in Quebec French and needs your approval before any "
        "rater sees it. *Pour la personne réviseure clinique : indiquez pour chaque ligne* **OK**, **modifier "
        "(suggestion)** *ou* **retirer**. *Le libellé français est une ébauche en français québécois.*",
        "",
        "Status: *validated* = agrees with expert labels at least as well as a single human rater; *partial* = some "
        "agreement; *failed* = tested and does not track the concept; *untestable* = available labels unreliable; "
        "*untested* = no labelled data yet. For failed or untested features the wording describes the image property, "
        "not a clinical sign.",
        "",
        "## 1. Phrases used in explanations / Termes utilisés dans les explications",
        "",
        "| feature | status | English | Français (ébauche) | your decision |",
        "|---|---|---|---|---|",
    ]
    for r in shown.itertuples():
        lines.append(f"| `{r.feature_id}` | {r.status} | {r.explanation_wording} | {r.explanation_wording_fr} | |")
    lines += ["", "Sentence pattern / Modèle de phrase:", "",
              "- EN: *Shape symmetry is asymmetric (much more asymmetric than most benign lesions); this weakly supports "
              "the malignant prediction.*",
              "- FR: *Symétrie de la forme : asymétrique (nettement plus asymétrique que la plupart des lésions "
              "bénignes); ce constat appuie faiblement la prédiction « maligne ».*", "",
              "## 2. Avoided terms / Termes à éviter", "",
              "| category | English | Français | why | say instead / formulation à privilégier | your decision |",
              "|---|---|---|---|---|---|"]
    for r in av.itertuples():
        inst = r.say_instead_en if r.say_instead_en == r.say_instead_fr else f"{r.say_instead_en} / {r.say_instead_fr}"
        lines.append(f"| {r.category} | {r.term_en} | {r.term_fr} | {r.reason} | {inst} | |")
    lines += ["", "## 3. Caveats and limitations / Mises en garde et limites", "",
              "| English | Français (ébauche) | your decision |", "|---|---|---|"]
    from llm_input import LIMITATIONS
    en_notes = ["The evidence is mixed.", "The model is uncertain about this case and it should be reviewed by a clinician.",
                "The interpretable model does not reproduce this prediction, so the reasons may not reflect what drove it.",
                "An independent model based on the measured lesion features gives this prediction less support, so it "
                "should be reviewed with caution.", "The automatic lesion outline may be inaccurate."]
    for en, fr in zip(en_notes + LIMITATIONS, list(lang_fr.NOTES.values()) + lang_fr.LIMITATIONS_FR):
        lines.append(f"| {en} | {fr} | |")
    lines += ["", "## 4. Questions / Questions", "",
              "1. Counter-intuitive reasons: a value typical of benign lesions sometimes counts toward malignant in the "
              "model (e.g. 'shape symmetry is within the typical range of benign lesions; this weakly supports the "
              "malignant prediction'). Show it, explain it, or leave it out? / *Raisons contre-intuitives : les "
              "afficher, les expliquer ou les omettre?*",
              "2. 'Shape symmetry: asymmetric' repeats the measure's own word. Better: 'shape: asymmetric'? / "
              "*« Symétrie de la forme : asymétrique » : préférez-vous « forme : asymétrique »?*",
              "3. Reader: dermatologist or general practitioner? / *Lectorat : dermatologue ou médecin de famille?*",
              "4. Anything a dermatologist would expect to see that is missing? / *Manque-t-il un élément attendu?*", ""]
    return "\n".join(lines)


def main():
    out = ROOT / "concepts/output"
    av, pb = avoided_table(), bilingual_phrase_bank()
    av.to_csv(out / "avoided_terms_v1_en_fr.csv", index=False, encoding="utf-8-sig")
    pb.to_csv(out / "phrase_bank_v1_en_fr.csv", index=False, encoding="utf-8-sig")
    (out / "clinical_review_v1_en_fr.md").write_text(review_doc(pb, av), encoding="utf-8")
    print(f"{len(av)} avoided-term entries, {len(pb)} phrase-bank rows -> {out}")


if __name__ == "__main__":
    main()
