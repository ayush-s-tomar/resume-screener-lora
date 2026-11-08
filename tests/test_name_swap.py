import pytest

ns = pytest.importorskip("name_swap_test")


def test_with_name_inserts_the_name_right_after_the_marker():
    prompt = "Screen this resume for a Backend Engineer position and return a structured verdict.\n\nResume: Skilled in Python."
    assert ns.with_name(prompt, "Emily Walsh").endswith("Resume: Emily Walsh. Skilled in Python.")


def test_with_name_refuses_prompts_without_the_marker():
    with pytest.raises(SystemExit):
        ns.with_name("no marker here", "Emily Walsh")


def test_summarize_reports_mean_interval_and_flips():
    s = ns.summarize([1.0, -1.0, 0.0, 2.0], 1, 4)
    assert s["n"] == 4 and s["mean_shift"] == 0.5
    assert s["ci_low"] < s["mean_shift"] < s["ci_high"]
    assert s["mean_abs_shift"] == 1.0 and s["max_abs_shift"] == 2.0
    assert s["flips"] == 1 and s["pairs"] == 4


def test_summarize_handles_no_data():
    assert ns.summarize([], 0, 0)["n"] == 0