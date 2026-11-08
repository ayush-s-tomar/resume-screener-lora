import random

import pytest

aug = pytest.importorskip("augment_dataset")

BODY = "Bachelor's degree in Computer Science. 4 years of experience building distributed backend systems. Skilled in Python, PostgreSQL, Docker, Kubernetes."
PROMPT = "Screen this resume for a Backend Engineer position and return a structured verdict.\n\nResume: " + BODY


def test_sentences_splits_on_sentence_boundaries():
    parts = aug.sentences(BODY)
    assert len(parts) == 3
    assert parts[2].startswith("Skilled in")


def test_preserves_accepts_extra_content_and_rejects_lost_content():
    assert aug.preserves(BODY, BODY + " Fluent in English.")
    assert not aug.preserves(BODY, "Bachelor's degree in Computer Science.")


@pytest.mark.parametrize("style", aug.STYLES)
def test_every_layout_style_keeps_all_content(style):
    text = aug.render(aug.sentences(BODY), style, random.Random(1))
    assert aug.preserves(BODY, text)


def test_augment_one_changes_only_the_resume_text():
    ex = {"prompt": PROMPT, "completion": '{"verdict": "moderate_match", "ats_score": 56}'}
    result = aug.augment_one(ex, random.Random(3), ["Hannah Foster"])
    assert result is not None
    new, _style = result
    assert new["completion"] == ex["completion"]
    assert new["prompt"] != PROMPT
    assert new["prompt"].startswith(PROMPT.split("Resume: ")[0] + "Resume: ")


def test_training_names_do_not_overlap_the_name_swap_test_names():
    banned = {w.lower() for n in aug.TEST_NAMES for w in n.split()}
    used = {w.lower() for w in aug.FIRST + aug.LAST}
    assert not banned & used