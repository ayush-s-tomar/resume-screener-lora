import json
import pathlib

import pytest

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
VERDICTS = {"strong_match", "moderate_match", "weak_match"}
TEST_NAMES = ["Emily Walsh", "Greg Baker", "Lakisha Washington", "Jamal Robinson",
              "Priya Sharma", "Rahul Verma", "Mei Chen", "Mohammed Hassan"]


def rows(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(name + " is not present")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_train_and_eval_prompts_do_not_overlap():
    assert not ({r["prompt"] for r in rows("train.jsonl")} & {r["prompt"] for r in rows("eval.jsonl")})


@pytest.mark.parametrize("name", ["train.jsonl", "eval.jsonl"])
def test_completions_are_valid_and_scores_match_their_verdict_band(name):
    for r in rows(name):
        c = json.loads(r["completion"])
        assert c["verdict"] in VERDICTS
        score = c["ats_score"]
        assert 0 <= score <= 100
        if c["verdict"] == "strong_match":
            assert score >= 75
        elif c["verdict"] == "moderate_match":
            assert 50 <= score <= 74
        else:
            assert score <= 48


def test_handwritten_set_shape():
    rs = rows("handwritten_eval.jsonl")
    assert len(rs) == 36
    assert len({r["id"] for r in rs}) == 36
    for label in VERDICTS:
        assert sum(r["label"] == label for r in rs) == 12


def test_augmented_rows_keep_their_original_labels():
    completions = {r["completion"] for r in rows("train.jsonl")}
    assert all(r["completion"] in completions for r in rows("train_aug.jsonl"))


def test_no_name_swap_test_name_appears_in_augmented_training_data():
    for r in rows("train_aug.jsonl"):
        for name in TEST_NAMES:
            assert name not in r["prompt"]