"""Tests for the command-line options of inference.py that do not need a model."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import inference  # noqa: E402


def test_explicit_adapter_wins(monkeypatch):
    monkeypatch.setenv("RESUME_SCREENER_ADAPTER", "from-env")
    assert inference.resolve_adapter("explicit/path") == "explicit/path"


def test_environment_variable_is_used_next(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "resume-screener-lora-adapter").mkdir()
    monkeypatch.setenv("RESUME_SCREENER_ADAPTER", "from-env")
    assert inference.resolve_adapter() == "from-env"


def test_local_folder_is_used_when_it_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("RESUME_SCREENER_ADAPTER", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "resume-screener-lora-adapter").mkdir()
    assert inference.resolve_adapter() == inference.ADAPTER_PATH


def test_hub_is_the_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("RESUME_SCREENER_ADAPTER", raising=False)
    monkeypatch.chdir(tmp_path)
    assert inference.resolve_adapter() == inference.HF_ADAPTER == "Kus-hal/resume-screener-lora"


def test_help_lists_the_adapter_option(capsys):
    with pytest.raises(SystemExit) as exc:
        inference.main(["--help"])
    assert exc.value.code == 0
    assert "--adapter" in capsys.readouterr().out
