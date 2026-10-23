"""
Clinician rater study materials (plan T7.2-T7.3): case set, counterbalancing, answer key, bilingual (EN / FR-QC)
LimeSurvey import files for Poly-Opinion, and an offline HTML preview for dry runs.

Design (T7.2): per case, sequential reveal with no going back:
  1. the rater's own impression (benign/malignant, most likely diagnosis, confidence 1-5);
  2. the model's prediction and probability (label-only view), impression again;
  3. ONE explanation, Grad-CAM or text (assigned per rater x case, balanced), rated 1-5 on clinical relevance,
     trust and clarity; supports / changes / no effect; final impression; optional comment.
Cases are stratified by true label x model correct/wrong x confident/uncertain (n_per_cell each; 5 -> 40 cases).
Raters see case codes only; isic_id, label and stratum are in case_key.csv, which is never uploaded.
Text explanations are the template arm on the number-free LLM input (explain/template.render_llm_input and its
French twin explain/lang_fr.render_llm_input_fr). French is a DRAFT until the clinical reviewer approves it.

    python eval/clinician_study/build_survey.py --pilot 5 --raters P --out eval/output/clinician_study_pilot
    python eval/clinician_study/build_survey.py --raters A B C --out eval/output/clinician_study_v1 \
        --img-base https://<poly-opinion-host>/upload/surveys/<sid>/images      # after creating the survey
"""
import argparse
import csv
import hashlib
import html
import json
import random
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "explain"))
import lang_fr  # noqa: E402
import llm_input as li  # noqa: E402
from template import render_llm_input  # noqa: E402

STUDY_VERSION = "clinician_study_v1_draft"
LANGS = ("en", "fr")
UI = {
    "en": {
        "title": "Clinician review of AI explanations (rater {r})",
        "welcome": ("Thank you for taking part. You will see {n} skin-lesion photographs from public research "
                    "archives. For each case you first give your own impression, then see the AI model's prediction, "
                    "then one explanation of that prediction. You cannot go back to change an earlier answer. "
                    "Some predictions are wrong; you will not be told which. About {minutes} minutes."),
        "end": "Thank you. Your coded answers have been recorded.",
        "bg": "About you", "bg_years": "Years in dermatology practice", "bg_derm": "How often do you use dermoscopy?",
        "bg_ai": "Experience with AI decision-support tools",
        "years": [("Y1", "Fewer than 5"), ("Y2", "5 to 10"), ("Y3", "11 to 20"), ("Y4", "More than 20")],
        "derm": [("D1", "Daily"), ("D2", "Weekly"), ("D3", "Rarely"), ("D4", "Never")],
        "ai": [("A1", "None"), ("A2", "Tried a few times"), ("A3", "Use regularly")],
        "case": "Case {i} of {n}", "step1": "Your own impression", "step2": "The AI model's prediction",
        "step3": "Explanation",
        "imp": "Your impression", "imp2": "Your impression after seeing the model's prediction",
        "imp3": "Your final impression", "bm": [("B", "Benign"), ("M", "Malignant")],
        "dx": "Most likely diagnosis (optional)", "conf": "Confidence in your impression (1 = very low, 5 = very high)",
        "model": "The model predicts: <b>{label}</b> (probability of malignancy {p}).",
        "labels": {"malignant": "malignant", "benign": "benign"},
        "cam_caption": ("Heat map (Grad-CAM): warmer colours mark the image regions that most influenced the model's "
                        "prediction."),
        "rel": "Clinical relevance of this explanation (1 = none, 5 = high)",
        "trust": "How much does this explanation make you trust the prediction? (1 = not at all, 5 = fully)",
        "clar": "Clarity (1 = very unclear, 5 = very clear)",
        "eff": "This explanation…", "effs": [("S", "supports my judgment"), ("C", "changes my judgment"),
                                             ("N", "has no effect on my judgment")],
        "comment": "Comment (optional)", "next": "Next", "download": "Download answers (CSV)",
        "required": "Please answer every required question.",
    },
    "fr": {
        "title": "Évaluation clinique d'explications produites par l'IA (personne évaluatrice {r})",
        "welcome": ("Merci de votre participation. Vous verrez {n} photographies de lésions cutanées provenant "
                    "d'archives de recherche publiques. Pour chaque cas, vous donnez d'abord votre propre impression, "
                    "puis vous voyez la prédiction du modèle d'IA, puis une explication de cette prédiction. Il n'est "
                    "pas possible de revenir en arrière pour modifier une réponse. Certaines prédictions sont "
                    "erronées; on ne vous dira pas lesquelles. Environ {minutes} minutes."),
        "end": "Merci. Vos réponses codées ont été enregistrées.",
        "bg": "À votre sujet", "bg_years": "Années de pratique en dermatologie",
        "bg_derm": "À quelle fréquence utilisez-vous la dermoscopie?",
        "bg_ai": "Expérience avec les outils d'aide à la décision fondés sur l'IA",
        "years": [("Y1", "Moins de 5"), ("Y2", "De 5 à 10"), ("Y3", "De 11 à 20"), ("Y4", "Plus de 20")],
        "derm": [("D1", "Tous les jours"), ("D2", "Chaque semaine"), ("D3", "Rarement"), ("D4", "Jamais")],
        "ai": [("A1", "Aucune"), ("A2", "Essayés quelques fois"), ("A3", "Utilisation régulière")],
        "case": "Cas {i} sur {n}", "step1": "Votre impression", "step2": "La prédiction du modèle d'IA",
        "step3": "Explication",
        "imp": "Votre impression", "imp2": "Votre impression après avoir vu la prédiction du modèle",
        "imp3": "Votre impression finale", "bm": [("B", "Bénigne"), ("M", "Maligne")],
        "dx": "Diagnostic le plus probable (facultatif)",
        "conf": "Confiance dans votre impression (1 = très faible, 5 = très élevée)",
        "model": "Le modèle prédit : <b>{label}</b> (probabilité de malignité {p}).",
        "labels": {"malignant": "lésion maligne", "benign": "lésion bénigne"},
        "cam_caption": ("Carte de chaleur (Grad-CAM) : les couleurs chaudes indiquent les régions de l'image qui ont "
                        "le plus influencé la prédiction du modèle."),
        "rel": "Pertinence clinique de cette explication (1 = nulle, 5 = élevée)",
        "trust": "Dans quelle mesure cette explication vous fait-elle confiance à la prédiction? (1 = pas du tout, 5 = entièrement)",
        "clar": "Clarté (1 = très peu claire, 5 = très claire)",
        "eff": "Cette explication…", "effs": [("S", "appuie mon jugement"), ("C", "modifie mon jugement"),
                                              ("N", "n'a pas d'effet sur mon jugement")],
        "comment": "Commentaire (facultatif)", "next": "Suivant", "download": "Télécharger les réponses (CSV)",
        "required": "Veuillez répondre à toutes les questions obligatoires.",
    },
}
UI["fr"]["trust"] = ("Dans quelle mesure cette explication augmente-t-elle votre confiance envers la prédiction? "
                     "(1 = pas du tout, 5 = entièrement)")


def prob(p, lang):
    return f"{p:.2f}".replace(".", ",") if lang == "fr" else f"{p:.2f}"


# ---------------------------------------------------------------- case set
def load_pool(inputs, cams_path, images):
    rows = [json.loads(line) for line in open(inputs, encoding="utf-8")]
    threshold = li.entropy_threshold_from_val(rows)
    cams = np.load(cams_path)
    with_cam = set(cams["isic_id"].tolist())
    pool = []
    for r in rows:
        if r["split"] != "test" or r["isic_id"] not in with_cam or not (Path(images) / f"{r['isic_id']}.jpg").exists():
            continue
        inp = li.build(r["llm_input"], r["mask_quality"], threshold)
        correct = (inp["prediction"] == "malignant") == (r["label"] == 1)
        r.update(inp=inp, correct=correct, stratum=f"{'malignant' if r['label'] else 'benign'}|"
                 f"{'correct' if correct else 'wrong'}|{'uncertain' if inp['uncertain'] else 'confident'}")
        pool.append(r)
    return pool, threshold


def select_cases(pool, n_per_cell, pilot, seed):
    rng = random.Random(seed)
    cells = {}
    for r in pool:
        cells.setdefault(r["stratum"], []).append(r)
    keys = sorted(cells)
    if pilot:  # round-robin over cells so a small pilot still covers wrong and uncertain cases
        order = [rng.sample(cells[k], len(cells[k])) for k in keys]
        picked = [order[i % len(keys)][i // len(keys)] for i in range(pilot)]
    else:
        picked = [r for k in keys for r in rng.sample(cells[k], min(n_per_cell, len(cells[k])))]
    rng.shuffle(picked)  # case codes must not reveal the stratum
    return [dict(r, case=f"C{i + 1:02d}") for i, r in enumerate(picked)]


def arm_for(case_index, rater_index):
    return "gradcam" if (case_index + rater_index) % 2 == 0 else "text"


def write_images(cases, cams_path, images, out):
    cams = np.load(cams_path)
    index = {k: i for i, k in enumerate(cams["isic_id"].tolist())}
    jet = colormaps["jet"]
    (out / "images").mkdir(parents=True, exist_ok=True)
    for c in cases:
        img = Image.open(Path(images) / f"{c['isic_id']}.jpg").convert("RGB")
        img.save(out / "images" / f"{c['case']}.jpg", quality=92)
        cam = cams["cam"][index[c["isic_id"]]].astype(np.float32)  # 224 x 224 on the squashed image (test transform)
        cam = np.asarray(Image.fromarray(cam).resize(img.size, Image.BILINEAR))
        heat = jet(np.clip(cam, 0, 1))[..., :3]
        blend = (0.55 * np.asarray(img, dtype=np.float32) / 255 + 0.45 * heat) * 255
        Image.fromarray(blend.clip(0, 255).astype(np.uint8)).save(out / "images" / f"{c['case']}_gradcam.jpg", quality=92)


def explanation_html(inp, lang):
    e = render_llm_input(inp) if lang == "en" else lang_fr.render_llm_input_fr(inp)
    parts = [f"<p>{html.escape(e['summary'])}</p>"]
    if e["uncertainty_note"]:
        parts.append(f"<p><i>{html.escape(e['uncertainty_note'])}</i></p>")
    parts.append("<ol>" + "".join(f"<li>{html.escape(r['text'])}</li>" for r in e["reasons"]) + "</ol>")
    parts.append(f"<p><small>{html.escape(e['limitations'])}</small></p>")
    return "".join(parts)


# ---------------------------------------------------------------- survey structure (shared by LimeSurvey + preview)
def steps_for(cases, rater, rater_index, lang, img_base):
    t, n = UI[lang], len(cases)
    out = [{"group": "BG", "title": t["bg"], "html": "", "questions": [
        {"code": "BGyears", "type": "L", "text": t["bg_years"], "options": t["years"], "mandatory": True},
        {"code": "BGderm", "type": "L", "text": t["bg_derm"], "options": t["derm"], "mandatory": True},
        {"code": "BGai", "type": "L", "text": t["bg_ai"], "options": t["ai"], "mandatory": True}]}]
    order = list(range(n))
    random.Random(1000 + rater_index).shuffle(order)  # case order differs per rater
    for pos, ci in enumerate(order, 1):
        c = cases[ci]
        code, arm = c["case"], arm_for(ci, rater_index)
        img = f'<p><img src="{img_base}/{code}.jpg" alt="{code}" style="max-width:100%;max-height:60vh"></p>'
        head = t["case"].format(i=pos, n=n)
        p = c["llm_input"]["probability_malignant"]
        label = t["labels"][c["inp"]["prediction"]]
        if arm == "gradcam":
            expl = (f'<p><img src="{img_base}/{code}_gradcam.jpg" alt="{code} Grad-CAM" style="max-width:100%;'
                    f'max-height:60vh"></p><p>{html.escape(t["cam_caption"])}</p>')
        else:
            expl = img + explanation_html(c["inp"], lang)
        model = f"<p>{t['model'].format(label=html.escape(label), p=prob(p, lang))}</p>"
        out += [
            {"group": f"{code}s1", "title": f"{head}: {t['step1']}", "html": img, "case": code, "arm": arm,
             "questions": [{"code": f"{code}imp", "type": "L", "text": t["imp"], "options": t["bm"], "mandatory": True},
                           {"code": f"{code}dx", "type": "S", "text": t["dx"], "mandatory": False},
                           {"code": f"{code}conf", "type": "5", "text": t["conf"], "mandatory": True}]},
            {"group": f"{code}s2", "title": f"{head}: {t['step2']}", "html": img + model, "case": code, "arm": arm,
             "questions": [{"code": f"{code}imp2", "type": "L", "text": t["imp2"], "options": t["bm"], "mandatory": True}]},
            {"group": f"{code}s3", "title": f"{head}: {t['step3']}", "html": model + expl, "case": code, "arm": arm,
             "questions": [{"code": f"{code}rel", "type": "5", "text": t["rel"], "mandatory": True},
                           {"code": f"{code}trust", "type": "5", "text": t["trust"], "mandatory": True},
                           {"code": f"{code}clar", "type": "5", "text": t["clar"], "mandatory": True},
                           {"code": f"{code}eff", "type": "L", "text": t["eff"], "options": t["effs"], "mandatory": True},
                           {"code": f"{code}imp3", "type": "L", "text": t["imp3"], "options": t["bm"], "mandatory": True},
                           {"code": f"{code}com", "type": "T", "text": t["comment"], "mandatory": False}]},
        ]
    return out


# ---------------------------------------------------------------- LimeSurvey TSV (Poly-Opinion import)
TSV_COLS = ["id", "related_id", "class", "type/scale", "name", "relevance", "text", "help", "language", "validation",
            "mandatory", "other", "default", "same_default"]


def limesurvey_tsv(steps_by_lang, rater, n_cases):
    """Tab-separated survey structure: S rows, then the full primary-language block (SL, G, Q, A), then the French
    copy with the same rows in the same order and the same ids (LimeSurvey manual, TSV survey structure)."""
    minutes = round(10 + 1.6 * n_cases)
    rows = [dict(zip(TSV_COLS, ["", "", "S", "", k, "", v, "", "", "", "", "", "", ""])) for k, v in [
        ("language", "en"), ("additional_languages", "fr"), ("format", "G"), ("allowprev", "N"),
        ("savetimings", "Y"), ("datestamp", "Y"), ("anonymized", "Y")]]
    for lang in LANGS:
        t = UI[lang]
        for k, v in [("surveyls_title", t["title"].format(r=rater)),
                     ("surveyls_welcometext", t["welcome"].format(n=n_cases, minutes=minutes)),
                     ("surveyls_endtext", t["end"])]:
            rows.append(dict(zip(TSV_COLS, ["", "", "SL", "", k, "", v, "", lang] + [""] * 5)))
        qid = 1000
        for gid, step in enumerate(steps_by_lang[lang], 1):
            rows.append(dict(zip(TSV_COLS, [gid, "", "G", "", step["title"], "1", step["html"], "", lang] + [""] * 5)))
            for q in step["questions"]:
                qid += 1
                rows.append(dict(zip(TSV_COLS, [qid, "", "Q", q["type"], q["code"], "1", q["text"], "", lang, "",
                                                "Y" if q["mandatory"] else "N", "N", "", ""])))
                for code, label in q.get("options", []):
                    rows.append(dict(zip(TSV_COLS, [qid, "", "A", "0", code, "", label, "", lang] + [""] * 5)))
    return rows


def write_tsv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:  # UTF-8 with BOM, as LimeSurvey exports
        w = csv.DictWriter(fh, fieldnames=TSV_COLS, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- offline HTML preview (dry run without LimeSurvey)
PREVIEW = """<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font-family:system-ui,sans-serif;max-width:820px;margin:0 auto;padding:16px;background:#fff;color:#1d1d1f;line-height:1.45}}
fieldset{{border:1px solid #ccc;border-radius:6px;margin:12px 0;padding:8px 12px}} legend{{font-weight:600}}
label{{display:inline-block;margin:4px 14px 4px 0}} button{{font-size:1rem;padding:8px 18px;margin-top:10px}}
.err{{color:#b00020}} textarea,input[type=text]{{width:100%;font-size:1rem}} small{{color:#555}}
</style></head><body><h1>{title}</h1><div id="app"></div><script>
const STEPS={steps};const UI={ui};const answers=[];let k=-1,t0=Date.now();
function scale(q){{return [1,2,3,4,5].map(v=>[String(v),String(v)])}}
function render(){{const app=document.getElementById('app');
 if(k<0){{app.innerHTML='<p>'+UI.welcome+'</p><button onclick="next()">'+UI.next+'</button>';return}}
 if(k>=STEPS.length){{const csv=['rater,step,case,arm,question,answer,seconds'].concat(answers.map(a=>a.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(','))).join('\\n');
  const url=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));app.innerHTML='<p>'+UI.end+'</p><a download="answers_{rater}.csv" href="'+url+'">'+UI.download+'</a>';return}}
 const s=STEPS[k];let h='<h2>'+s.title+'</h2>'+s.html;
 for(const q of s.questions){{h+='<fieldset><legend>'+q.text+(q.mandatory?' *':'')+'</legend>';
  if(q.type==='L'||q.type==='5'){{for(const [c,l] of (q.type==='5'?scale(q):q.options)) h+='<label><input type="radio" name="'+q.code+'" value="'+c+'"> '+l+'</label>'}}
  else if(q.type==='S'){{h+='<input type="text" name="'+q.code+'">'}} else {{h+='<textarea rows="3" name="'+q.code+'"></textarea>'}}
  h+='</fieldset>'}}
 app.innerHTML=h+'<p class="err" id="err"></p><button onclick="next()">'+UI.next+'</button>';window.scrollTo(0,0)}}
function next(){{if(k>=0){{const s=STEPS[k],row=[];for(const q of s.questions){{
  const el=document.querySelector('[name="'+q.code+'"]:checked')||document.querySelector('[name="'+q.code+'"]:not([type=radio])');
  const v=el?el.value.trim():'';if(q.mandatory&&!v){{document.getElementById('err').textContent=UI.required;return}}row.push([q.code,v])}}
  const sec=((Date.now()-t0)/1000).toFixed(1);for(const [c,v] of row) answers.push(['{rater}',s.group,s.case||'',s.arm||'',c,v,sec])}}
 k++;t0=Date.now();render()}}
render();</script></body></html>"""


def write_preview(steps, rater, lang, n_cases, path):
    t = UI[lang]
    ui = {"welcome": html.escape(t["welcome"].format(n=n_cases, minutes=round(10 + 1.6 * n_cases))),
          "end": html.escape(t["end"]), "next": t["next"], "download": t["download"], "required": t["required"]}
    for s in steps:  # the preview hides arm/case codes from the page but keeps them in the CSV
        for q in s["questions"]:
            q["text"] = html.escape(q["text"])
    path.write_text(PREVIEW.format(lang=lang, title=html.escape(t["title"].format(r=rater)), rater=rater,
                                   steps=json.dumps(steps, ensure_ascii=False), ui=json.dumps(ui, ensure_ascii=False)),
                    encoding="utf-8")


def main(opts):
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    pool, threshold = load_pool(opts.inputs, opts.cams, opts.images)
    cases = select_cases(pool, opts.n_per_cell, opts.pilot, opts.seed)
    write_images(cases, opts.cams, opts.images, out)
    with open(out / "case_key.csv", "w", newline="", encoding="utf-8") as fh:  # never uploaded: raters see codes only
        w = csv.writer(fh)
        w.writerow(["case", "isic_id", "label", "prediction", "p_malignant", "correct", "uncertain", "stratum",
                    "source", "mask_quality", "input_hash"])
        for c in cases:
            w.writerow([c["case"], c["isic_id"], c["label"], c["inp"]["prediction"],
                        c["llm_input"]["probability_malignant"], c["correct"], c["inp"]["uncertain"], c["stratum"],
                        c["source"], c["mask_quality"], li.input_hash(c["inp"])])
    assignments = []
    for ri, rater in enumerate(opts.raters):
        steps = {lang: steps_for(cases, rater, ri, lang, opts.img_base) for lang in LANGS}
        write_tsv(limesurvey_tsv(steps, rater, len(cases)), out / f"limesurvey_rater_{rater}.txt")
        for lang in LANGS:
            write_preview(steps_for(cases, rater, ri, lang, "images"), rater, lang, len(cases),
                          out / f"preview_rater_{rater}_{lang}.html")
        assignments += [[rater, pos, s["case"], s["arm"]] for pos, s in
                        enumerate([s for s in steps["en"] if s["group"].endswith("s1")], 1)]
    with open(out / "assignments.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["rater", "position", "case", "arm"]] + assignments)
    manifest = {"study_version": STUDY_VERSION, "llm_input_version": li.LLM_INPUT_VERSION,
                "phrase_bank": li.PHRASE_BANK_VERSION, "fr_version": lang_fr.FR_VERSION,
                "inputs": str(opts.inputs), "inputs_sha256_16": hashlib.sha256(Path(opts.inputs).read_bytes()).hexdigest()[:16],
                "cams": str(opts.cams), "entropy_threshold": threshold, "seed": opts.seed, "pilot": opts.pilot,
                "n_cases": len(cases), "strata": {s: sum(c["stratum"] == s for c in cases) for s in sorted({c["stratum"] for c in cases})},
                "raters": opts.raters, "img_base": opts.img_base, "explanation_arm": "template (render_llm_input / _fr)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(ROOT / "trees/output_v1.0_detv2/explanation_inputs_deferral.jsonl"))
    ap.add_argument("--cams", default=str(ROOT / "eval/output/gradcam_v2/cams.npz"))
    ap.add_argument("--images", default=str(ROOT / "uncertaintyNet-main/datasets_768/ISIC_clinical"))
    ap.add_argument("--n_per_cell", type=int, default=5)
    ap.add_argument("--pilot", type=int, default=0, help="small round-robin case set instead of n_per_cell")
    ap.add_argument("--raters", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-base", dest="img_base", default="IMG_BASE",
                    help="URL prefix of the uploaded images in Poly-Opinion (set after the survey is created)")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
