"""Phase-detection lexicons and Qwen3 special-token IDs.

Kept as a separate module so callers (env wrapper, classifier, eval scripts)
can override or extend without touching detection logic.
"""

# Qwen3 thinking-mode special tokens (see /workspace/qwen3-example.py).
QWEN3_THINK_OPEN_ID = 151667
QWEN3_THINK_CLOSE_ID = 151668

# Phase-driving lexicons. All matched lowercased on a recent decoded window.
TURN_KEYWORDS = (
    "wait",
    "actually",
    "hmm",
    "but",
    "let me re-check",
    "let me check",
    "on second thought",
    "i made",
    "correction",
)

EXPAND_KEYWORDS = (
    "therefore",
    "thus",
    "let's compute",
    "let's",
    "we have",
    "consider",
    "next",
    "first,",
    "so we",
    "in other words",
)

STEP_MARKERS = ("step", "###")
FINAL_MARKERS = ("\\boxed", "final answer", "answer:")

OP_CHARS = frozenset("=+-*/^<>%")
