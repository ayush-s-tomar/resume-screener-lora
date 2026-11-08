import json

import pytest

gb = pytest.importorskip("eval_groq_baseline")


def test_build_rubric_uses_min_and_max_per_verdict(tmp_path):
    rows = [("strong_match", 75), ("strong_match", 100), ("moderate_match", 50), ("moderate_match", 74),
            ("weak_match", 8), ("weak_match", 48)]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps({"prompt": "p", "completion": json.dumps({"verdict": v, "ats_score": s})}) for v, s in rows), encoding="utf-8")
    ranges = gb.build_rubric(str(path))
    assert ranges["strong_match"] == (75, 100)
    assert ranges["moderate_match"] == (50, 74)
    assert ranges["weak_match"] == (8, 48)


def test_rubric_instruction_extends_the_base_prompt_with_the_ranges():
    text = gb.rubric_instruction({"strong_match": (75, 100)})
    assert text.startswith(gb.ZERO_SHOT_INSTRUCTION)
    assert "strong_match: ats_score 75-100" in text


def test_score_predictions_counts_valid_json_accuracy_and_error():
    preds = ['{"verdict": "strong_match", "ats_score": 80}', "garbage"]
    truths = [{"verdict": "strong_match", "ats_score": 78}, {"verdict": "weak_match", "ats_score": 20}]
    result = gb.score_predictions(preds, truths)
    assert result["valid_json_rate"] == 0.5
    assert result["verdict_accuracy"] == 0.5
    assert result["mean_absolute_score_error"] == 2.0