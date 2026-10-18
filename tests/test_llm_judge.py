"""LLM-as-judge harness (explain/llm_judge.py, T6.4): planted-error controls, scoring, sheet and agreement."""
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explain"))
import llm_judge as J  # noqa: E402


def _row(i, uncertain=False, direction="supports", as_string=False):
    inp = {"prediction": "malignant", "confidence": "low" if uncertain else "high", "uncertain": uncertain,
           "uncertainty_reason": "", "evidence": [{"feature_id": "B_edge_sharpness", "measure": "edge transition",
                                                    "finding": "abrupt", "relative_to_benign": "much more abrupt",
                                                    "direction": direction, "strength": "weak"}]}
    verb = "supports" if direction == "supports" else "argues against"
    final = {"summary": "The model predicts malignant.",
             "reasons": [{"feature_id": "B_edge_sharpness", "direction": direction,
                          "text": f"the edge is much more abrupt than most benign lesions; this weakly {verb} the prediction"}],
             "uncertainty_note": "The model is uncertain; review by a clinician." if uncertain else ""}
    return {"isic_id": f"ISIC_{i}", "row_id": f"ISIC_{i}#base", "variant": "base", "stratum": "correct|confident",
            "llm_input": inp, "final_output": json.dumps(final) if as_string else final,
            "verify": {"pass_all": True, "direction": 1.0}}


def test_controls_plant_the_named_error_and_keep_a_clean_copy():
    rows = [dict(_row(i, uncertain=i % 2 == 0), arm="a") for i in range(40)]
    ctrl = J.make_controls(rows, seed=0)
    kinds = {t["kind"] for _, t in ctrl}
    assert {"clean", "direction_error", "unsupported_concept", "magnitude_error", "uncertainty_dropped"} <= kinds
    for r, t in ctrl:
        text = r["final_output"]["reasons"][0]["text"]
        if t["kind"] == "clean":
            assert r["final_output"] is rows[int(r["isic_id"].split("_")[1])]["final_output"]
        elif t["kind"] == "direction_error":
            assert "argues against" in text
        elif t["kind"] == "unsupported_concept":
            assert J.INVENTED in text
        elif t["kind"] == "uncertainty_dropped":
            assert r["llm_input"]["uncertain"] and r["final_output"]["uncertainty_note"] == ""
    # the original rows are never modified
    assert all("argues against" not in r["final_output"]["reasons"][0]["text"] for r in rows)


def test_control_hit():
    clean = {"labels": ["supported"], "overconfident": False}
    assert J.control_hit(clean, {"kind": "clean", "reason_index": None})
    assert not J.control_hit(clean, {"kind": "direction_error", "reason_index": 0})
    assert J.control_hit({"labels": ["other_error"]}, {"kind": "unsupported_concept", "reason_index": 0})
    assert J.control_hit({"labels": ["supported"], "uncertainty_conveyed": "no"}, {"kind": "uncertainty_dropped", "reason_index": None})
    assert not J.control_hit(None, {"kind": "clean", "reason_index": None})


def test_mock_run_sheet_and_agree(tmp_path):
    d = tmp_path / "arm__v1"
    d.mkdir()
    rows = [_row(i, uncertain=i % 3 == 0, as_string=i % 2 == 0) for i in range(12)]
    (d / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    gens = [str(d / "generations.jsonl")]
    J.run(Namespace(gens=gens, judge="mock", revision=None, backend="mock", control_arms=["arm__v1"], limit=0,
                    batch_size=5, out=str(tmp_path / "judge")))
    summary = json.loads((tmp_path / "judge" / "mock" / "judge_summary.json").read_text())
    assert summary["arms"]["arm__v1"]["parse_ok"] == 1.0  # string outputs are parsed, not judged as raw JSON
    assert summary["controls"]["clean"]["specificity"] == 1.0
    assert summary["controls"]["direction_error"]["recall"] == 0.0  # the mock judge never flags anything

    J.sheet(Namespace(gens=gens, n_per_cell=5, out=str(tmp_path / "human")))
    sheet = list(csv.reader(open(tmp_path / "human" / "rating_sheet.csv", encoding="utf-8")))
    assert sheet[0] == J.SHEET_COLS and len(sheet) == 6 and "arm__v1" not in "".join(map(str, sheet))
    # a human who marks every claim as a direction error: the judge (all supported) has recall 0 and kappa 0
    filled = tmp_path / "filled.csv"
    with open(filled, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(sheet[0])
        for row in sheet[1:]:
            w.writerow([row[0], row[1], row[2], "D"] + row[4:])
    J.agree(Namespace(sheet=str(filled), key=str(tmp_path / "human" / "rating_key.csv"),
                      judged=str(tmp_path / "judge" / "mock" / "judgments.jsonl"), out=str(tmp_path / "agree.json")))
    res = json.loads((tmp_path / "agree.json").read_text())
    assert res["n_claims"] == 5 and res["direction_error"]["judge_recall"] == 0.0


def test_cohen_kappa():
    assert J.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert J.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
