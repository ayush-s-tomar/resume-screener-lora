"""Checks that pyproject.toml matches the repository: modules, command-line entry points, version and extras."""
import ast
import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # standard library from Python 3.11
ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
MODULES = CONFIG["tool"]["setuptools"]["py-modules"]


def parse(module):
    return ast.parse((ROOT / (module + ".py")).read_text(encoding="utf-8"))


def test_packaged_modules_exist():
    for module in MODULES:
        assert (ROOT / (module + ".py")).is_file(), module


def test_entry_points_resolve():
    scripts = CONFIG["project"]["scripts"]
    assert scripts
    for name, target in scripts.items():
        module, func = target.split(":")
        assert module in MODULES, "%s points at %s, which is not packaged" % (name, module)
        defined = {n.name for n in parse(module).body if isinstance(n, ast.FunctionDef)}
        assert func in defined, "%s: %s has no function %s" % (name, module, func)


def test_local_imports_are_packaged():
    for module in MODULES:
        for node in ast.walk(parse(module)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for name in names:
                if (ROOT / (name + ".py")).is_file():
                    assert name in MODULES, "%s imports %s, which is not in py-modules" % (module, name)


def test_version_matches_the_citation_file():
    cff = ROOT / "CITATION.cff"
    if not cff.exists():
        pytest.skip("CITATION.cff is not present")
    m = re.search(r'(?m)^version:\s*"?([^"\s]+)', cff.read_text(encoding="utf-8"))
    if not m:
        pytest.skip("CITATION.cff has no version")
    assert CONFIG["project"]["version"] == m.group(1)


def test_extras_exist():
    assert {"files", "model", "demo", "dev"} <= set(CONFIG["project"]["optional-dependencies"])
