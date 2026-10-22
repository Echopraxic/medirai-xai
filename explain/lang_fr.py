"""
Quebec French for the explanation layer the reader sees (clinician study, MedirAI product). DRAFT: every term needs
approval by a French-speaking clinical reviewer (DERMATOLOGIST_REQUEST.md, role 1) before any rater sees it.

Mirrors explain/template.py PHRASES and template.render_llm_input, working from the number-free llm_input
(llm_input.build), so French and English explanations state exactly the same facts. Conventions:
  - finding words agree in gender/number with their measure (each entry is written as a pair);
  - comparisons are built so they never need further agreement ("plus <finding> que la plupart des lésions bénignes");
  - Quebec typography (OQLF): non-breaking space before ":", none before ";", « » with non-breaking spaces;
  - "appuie" / "va à l'encontre de" (not the anglicism "supporte"); inclusive doublets for clinicians.
"""
NBSP = " "
FR_VERSION = "phrase_bank_v1_fr_draft"

# feature -> (measure, finding when high, finding when low); same keys and meaning as template.PHRASES
PHRASES_FR = {
    "A_shape_asym_major": ("symétrie de la forme selon le grand axe", "asymétrique", "symétrique"),
    "A_shape_asym_minor": ("symétrie de la forme selon le petit axe", "asymétrique", "symétrique"),
    "A_shape_asym_max": ("symétrie de la forme", "asymétrique", "symétrique"),
    "A_color_asym_deltaE": ("répartition des couleurs entre les deux moitiés", "inégale", "égale"),
    "B_circularity": ("contour de la lésion", "rond et régulier", "irrégulier"),
    "B_convexity": ("tracé de la bordure", "lisse", "échancré ou dentelé"),
    "B_fractal_dim": ("bordure (complexité du tracé)", "déchiquetée", "lisse"),
    "B_radial_cv": ("contour de la lésion", "irrégulier", "régulier"),
    "B_edge_sharpness": ("transition entre la lésion et la peau environnante", "abrupte", "graduelle"),
    "B_edge_sharpness_cv": ("netteté du bord le long du contour", "variable", "uniforme"),
    "C_frac_white": ("proportion de zones blanches ou pâles", "élevée", "faible"),
    "C_frac_red": ("proportion de zones de teinte rouge", "élevée", "faible"),
    "C_frac_light_brown": ("proportion de zones brun clair", "élevée", "faible"),
    "C_frac_dark_brown": ("proportion de zones brun foncé", "élevée", "faible"),
    "C_frac_blue_gray": ("proportion de zones gris-bleu", "élevée", "faible"),
    "C_frac_black": ("proportion de zones noires", "élevée", "faible"),
    "C_n_colors": ("variété des couleurs", "élevée", "faible"),  # never a colour count (not validated)
    "C_color_entropy": ("variété des couleurs", "élevée", "faible"),
    "C_color_std": ("variation des couleurs", "élevée", "faible"),
    "C_deltaE_vs_skin": ("contraste de couleur avec la peau environnante", "élevé", "faible"),
    "C_darkness_vs_skin": ("pigmentation par rapport à la peau environnante", "foncée", "claire"),
    "C_redness_vs_skin": ("teinte rouge par rapport à la peau environnante", "plus rouge", "moins rouge"),
    "C_yellowness_vs_skin": ("teinte jaune par rapport à la peau environnante", "plus jaune", "moins jaune"),
    "T_glcm_contrast": ("contraste de texture de l'image", "élevé", "faible"),
    "T_glcm_homogeneity": ("texture de l'image", "homogène", "hétérogène"),
    "T_glcm_energy": ("uniformité de la texture", "élevée", "faible"),
    "T_glcm_correlation": ("régularité de la texture", "élevée", "faible"),
    "T_glcm_entropy": ("complexité de la texture", "élevée", "faible"),
    "T_lbp_entropy": ("micro-texture", "variée", "uniforme"),
    "T_lbp_flat_fraction": ("micro-texture plane", "fréquente", "rare"),
    "T_roughness": ("texture de l'image à petite échelle", "grossière", "fine"),
    "S_eccentricity": ("forme (allongement)", "allongée", "ronde"),
    "S_solidity": ("forme (compacité)", "compacte", "lobulée"),
    "S_extent": ("forme (remplissage du cadre englobant)", "compacte", "étalée"),
    "S_axis_ratio": ("forme (rapport des axes)", "ronde", "allongée"),
}
UNMAPPED_SHAPE = "motif de forme sans nom clinique"
TYPICAL, UNUSUAL = "typique", "inhabituel"
QUANTITY_WORDS = {"élevée", "élevé", "faible"}  # the measure is the quantity itself -> "au-dessus / en dessous"

PREDICTION = {"malignant": "maligne", "benign": "bénigne"}
CONFIDENCE = {"high": "élevée", "moderate": "modérée", "low": "faible"}
STRENGTH = {"weak": "faiblement", "moderate": "modérément", "strong": "fortement"}
CONCEPT = {"asymmetry": "asymétrie", "border": "bordure", "color": "couleur", "texture/surface": "texture/surface",
           "shape": "forme"}

NOTES = {
    "mixed": "Les éléments sont partagés.",
    "uncertain": "Le modèle est incertain pour ce cas, qui devrait être examiné par un clinicien ou une clinicienne.",
    "surrogate": ("Le modèle interprétable ne reproduit pas cette prédiction; les raisons ci-dessous pourraient donc ne "
                  "pas refléter ce qui l'a motivée."),
    "independent": ("Un modèle indépendant, fondé sur les caractéristiques mesurées de la lésion, appuie moins cette "
                    "prédiction; elle devrait être examinée avec prudence."),
    "outline": "Le contour automatique de la lésion pourrait être inexact.",
    "scope": ("Cette image diffère des images sur lesquelles le modèle a été entraîné; la prédiction et les raisons "
              "ci-dessous sont donc moins fiables, et le cas devrait être examiné par un clinicien ou une clinicienne."),
}
SHARE_SENTENCE_FR = {  # same keys as template.SHARE_SENTENCE
    "most": "Les caractéristiques nommées ci-dessous expliquent l'essentiel de cette décision.",
    "about half": ("Les caractéristiques nommées ci-dessous expliquent environ la moitié de cette décision; le reste "
                   "provient de motifs de l'image sans nom clinique."),
    "little": ("Les caractéristiques nommées ci-dessous expliquent peu de cette décision; l'essentiel provient de motifs "
               "de l'image sans nom clinique."),
}
LIMITATIONS_FR = [
    "Les caractéristiques sont mesurées à partir d'un contour automatique de la lésion, sur une seule photographie clinique.",
    ("La taille de la lésion ne peut pas être donnée en millimètres, et son évolution dans le temps ne peut pas être "
     "évaluée à partir d'une seule image."),
    "Les raisons proviennent d'un modèle interprétable qui imite approximativement le classificateur d'images.",
]


def q(text):
    """French quotation marks with non-breaking spaces."""
    return f"«{NBSP}{text}{NBSP}»"


def finding_fr(evidence):
    """English finding word (llm_input) -> agreed French word, via the feature's EN/FR phrase pair."""
    from template import PHRASES
    fid, word = evidence["feature_id"], evidence["finding"]
    if word == "typical":
        return TYPICAL
    if not evidence.get("clinically_mapped", True):
        return UNUSUAL
    _, high_en, low_en = PHRASES[fid]
    _, high_fr, low_fr = PHRASES_FR[fid]
    return high_fr if word == high_en else low_fr if word == low_en else word


def relative_fr(evidence, word_fr):
    """Same band as llm_input.relative, read back from its English phrase (no numbers in llm_input)."""
    rel = evidence["relative_to_benign"]
    if word_fr in (TYPICAL, UNUSUAL) and "typical range" in rel:
        return "dans l'intervalle habituel des lésions bénignes"
    much = "nettement " if rel.startswith("much ") else ""
    if word_fr in QUANTITY_WORDS:
        side = "au-dessus" if "higher" in rel else "en dessous"
        return f"{much}{side} de la plupart des lésions bénignes"
    if word_fr.startswith(("plus ", "moins ")):
        return f"{much}{word_fr} que la plupart des lésions bénignes"
    return f"{much}plus {word_fr} que la plupart des lésions bénignes"


def encoding_label_fr(evidence):
    """French twin of template.encoding_label."""
    if evidence.get("model_encoded", True):
        return ""
    if evidence["feature_id"].startswith(("A_shape", "B_", "S_")):
        return " (mesure du contour que le modèle d'image ne semble pas utiliser)"
    return " (mesure que le modèle d'image ne semble pas utiliser)"


def reason_fr(evidence, prediction):
    mapped = evidence.get("clinically_mapped", True)
    measure = PHRASES_FR[evidence["feature_id"]][0] if mapped else UNMAPPED_SHAPE
    word = finding_fr(evidence)
    adv, pred = STRENGTH[evidence["strength"]], q(PREDICTION[prediction])
    verb = (f"appuie {adv} la prédiction {pred}" if evidence["direction"] == "supports"
            else f"va {adv} à l'encontre de la prédiction {pred}")
    return (f"{measure[0].upper()}{measure[1:]}{NBSP}: {word} ({relative_fr(evidence, word)}); ce constat {verb}"
            f"{encoding_label_fr(evidence)}.")


def render_llm_input_fr(inp):
    """French twin of template.render_llm_input: same facts, same order, same caveats."""
    pred = inp["prediction"]
    summary = f"Le modèle prédit une lésion {PREDICTION[pred]}, avec une confiance {CONFIDENCE[inp['confidence']]}."
    if any(e["direction"] == "argues_against" for e in inp["evidence"]):
        summary += " " + NOTES["mixed"]
    if inp.get("named_concept_share"):
        summary += " " + SHARE_SENTENCE_FR[inp["named_concept_share"]]
    notes = []
    if inp.get("in_scope") is False:
        notes.append(NOTES["scope"])
    if inp["uncertain"]:
        notes.append(NOTES["uncertain"])
    if not inp["surrogate_agrees"]:
        notes.append(NOTES["surrogate"])
    if inp.get("independent_check_weak"):
        notes.append(NOTES["independent"])
    if not inp["segmentation_reliable"]:
        notes.append(NOTES["outline"])
    reasons = [{"feature_id": e["feature_id"], "direction": e["direction"], "text": reason_fr(e, pred)}
               for e in inp["evidence"]]
    return {"summary": summary, "reasons": reasons, "uncertainty_note": " ".join(notes),
            "limitations": " ".join(LIMITATIONS_FR)}
