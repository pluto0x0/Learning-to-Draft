"""Deferred import for the base ``SpeculativeDecodingEnv`` classes.

`rl/rl_depth.py` and `rl/rl_total.py` both call ``argparse.parse_args()`` and
``os.makedirs(...)`` at *module-import time*, which makes them inconvenient to
import as libraries. This loader temporarily neutralises ``sys.argv`` and
redirects the import-time ``mkdir`` to a temp directory so the modules pick
up their argparse defaults harmlessly, then returns the env classes for the
phase-aware training launchers to wrap.

The original modules are not modified.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Tuple, Type


def load_base_env_classes() -> Tuple[Type, Type]:
    """Return ``(DepthEnv, SizeEnv)`` from the existing rl_depth/rl_total modules."""
    saved_argv = sys.argv
    saved_cwd = os.getcwd()
    sys.argv = ["__phase_loader__"]
    with tempfile.TemporaryDirectory(prefix="ltd_phase_loader_") as tmp:
        try:
            os.chdir(tmp)
            from rl import rl_depth, rl_total  # type: ignore
        finally:
            os.chdir(saved_cwd)
            sys.argv = saved_argv

    return rl_depth.SpeculativeDecodingEnv, rl_total.SpeculativeDecodingEnv
