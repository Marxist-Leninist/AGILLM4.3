#!/usr/bin/env python3
"""Compatibility shim for the AGILLM4.1 outbound join worker."""
from pathlib import Path
import importlib.util
import os
import sys


_TARGET = Path(__file__).with_name("agillm41_join_worker.py")
os.environ.setdefault("TOKENIZER_ID", "deepseek-ai/DeepSeek-V3.2")

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, str(_TARGET), *sys.argv[1:]])

_SPEC = importlib.util.spec_from_file_location("agillm41_join_worker", _TARGET)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import AGILLM4.1 join worker from {_TARGET}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("agillm41_join_worker", _MODULE)
_SPEC.loader.exec_module(_MODULE)
globals().update(
    {
        key: value
        for key, value in vars(_MODULE).items()
        if key not in {"__name__", "__file__", "__package__", "__loader__", "__spec__"}
    }
)
