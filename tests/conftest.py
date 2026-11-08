"""Test setup: stub the heavy ML libraries when they are not installed, so the pure-Python helpers can be tested anywhere."""
import importlib.util
import sys
import types


def _stub(name):
    module = types.ModuleType(name)
    module.__getattr__ = lambda attr: type(attr, (), {})
    sys.modules[name] = module


for _name in ("torch", "peft", "transformers"):
    if importlib.util.find_spec(_name) is None:
        _stub(_name)