import pytest

ec = pytest.importorskip("eval_common")

TRUTHS = [("weak_match", None), ("moderate_match", None), ("strong_match", None)]


def test_extract_json_handles_plain_and_wrapped_json():
    assert ec.extract_json('{"a": 1}') == {"a": 1}
    assert ec.extract_json('Sure! {"verdict": "weak_match"} Hope that helps.') == {"verdict": "weak_match"}


def test_extract_json_returns_none_for_garbage():
    assert ec.extract_json("no json here") is None
    assert ec.extract_json("{broken") is None


def test_prompt_of_builds_the_training_prompt_from_role_and_resume():
    prompt = ec.prompt_of({"role": "Backend Engineer", "resume": "Python, Docker."})
    assert prompt.startswith("Screen this resume for a Backend Engineer position")
    assert prompt.endswith("Resume: Python, Docker.")


def test_prompt_of_prefers_an_existing_prompt():
    assert ec.prompt_of({"prompt": "already built"}) == "already built"


def test_truth_of_reads_both_label_formats():
    assert ec.truth_of({"completion": '{"verdict": "strong_match", "ats_score": 90}'}) == ("strong_match", 90)
    assert ec.truth_of({"label": "weak_match"}) == ("weak_match", None)


def test_num_and_verdict_of_are_defensive():
    assert ec.num({"ats_score": 55}) == 55.0
    assert ec.num({"ats_score": "55"}) is None
    assert ec.num({"ats_score": True}) is None
    assert ec.num(None) is None
    assert ec.verdict_of({"verdict": "strong_match"}) == "strong_match"
    assert ec.verdict_of({"verdict": "banana"}) == "invalid"
    assert ec.verdict_of(None) == "invalid"


def test_score_run_perfect_predictions():
    preds = [
        {"verdict": "weak_match", "ats_score": 20},
        {"verdict": "moderate_match", "ats_score": 60},
        {"verdict": "strong_match", "ats_score": 90},
    ]
    result = ec.score_run(preds, TRUTHS)
    assert result["verdict_accuracy"] == 1.0
    assert result["severe_errors"] == 0
    assert result["pairwise_ordering"] == 1.0
    assert result["confusion"]["strong_match"]["strong_match"] == 1


def test_score_run_counts_severe_errors_and_invalid_outputs():
    preds = [{"verdict": "strong_match", "ats_score": 90}, None, {"verdict": "strong_match", "ats_score": 80}]
    result = ec.score_run(preds, TRUTHS)
    assert result["verdict_correct"] == 1
    assert result["severe_errors"] == 1
    assert result["valid_json_rate"] == round(2 / 3, 3)
    assert result["confusion"]["moderate_match"]["invalid"] == 1