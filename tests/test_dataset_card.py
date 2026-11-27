"""Checks that docs/DATASET_CARD.md agrees with the data files."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs" / "DATASET_CARD.md"


def count(rel):
    return sum(1 for line in (ROOT / rel).read_text(encoding="utf-8-sig").splitlines() if line.strip())


@pytest.mark.parametrize("name,rel", [("train", "data/train.jsonl"), ("eval", "data/eval.jsonl"), ("handwritten_eval", "data/handwritten_eval.jsonl")])
def test_row_counts_match_the_data_files(name, rel):
    if not CARD.exists():
        pytest.skip("docs/DATASET_CARD.md is not present")
    m = re.search(r"\| `%s\.jsonl` \| [^|]+ \| (\d+) \|" % re.escape(name), CARD.read_text(encoding="utf-8"))
    assert m, "the card has no row for %s.jsonl" % name
    assert int(m.group(1)) == count(rel), "regenerate the card with: python docs/build_dataset_card.py"


@pytest.mark.parametrize("rel", ["data/train.jsonl", "data/eval.jsonl", "data/handwritten_eval.jsonl"])
def test_every_data_row_parses(rel):
    for line in (ROOT / rel).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            json.loads(line)
