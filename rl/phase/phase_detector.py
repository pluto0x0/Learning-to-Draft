"""Phase-aware feature extractor for reasoning LLM speculative decoding.

Per draft-and-verify cycle, the env wrapper calls `update(...)` with the tokens
just accepted by the target model, and `get_features()` to obtain a 32-dim
fixed-length vector to concatenate onto the LTD policy observation:

    [phi_onehot(5) | in_think(1) | history(18) | risk(8)]  -> 32

V1 uses a rule-based detector keyed on Qwen3 `<think>` / `</think>` tokens plus
shallow lexical signals on a small recent decoded window. Designed to be
CPU-only and well under the cost of a single ea_layer forward pass.
"""

from __future__ import annotations

import collections
from typing import Iterable, Optional

import numpy as np

from .keywords import (
    EXPAND_KEYWORDS,
    FINAL_MARKERS,
    OP_CHARS,
    QWEN3_THINK_CLOSE_ID,
    QWEN3_THINK_OPEN_ID,
    STEP_MARKERS,
    TURN_KEYWORDS,
)

PHASE_NON_THINK = 0
PHASE_PROBLEM_SETUP = 1
PHASE_REASONING_EXPAND = 2
PHASE_NUMERIC = 3
PHASE_SELF_CHECK = 4
NUM_PHASES = 5

# Layout of the concatenated phase feature vector.
_PHI_DIM = NUM_PHASES + 1   # 5 phases + in_think flag
_HIST_WINDOW = 6
_HIST_DIM = _HIST_WINDOW * 3  # accept_len, time, fail_pos per slot
_RISK_DIM = 8
PHASE_OBS_DIM = _PHI_DIM + _HIST_DIM + _RISK_DIM  # 32


class PhaseDetector:
    """Stateful, per-episode phase + feature extractor.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Used to decode the recent token window into text once per cycle. May be
        `None` for tests; in that case lexical features degrade to zeros.
    qwen3_mode : bool
        If True, uses 151667/151668 to drive `in_think`. If False, treats every
        cycle as in-think (useful for non-Qwen3 reasoning models that emit
        thoughts without explicit boundary tokens).
    phase_smoothing : int
        Require this many consecutive same-phase cycles before flipping the
        exposed phase, to smooth out transient lexical false positives.
    """

    def __init__(
        self,
        tokenizer=None,
        qwen3_mode: bool = True,
        phase_smoothing: int = 2,
        text_window_chars: int = 256,
        history_window: int = _HIST_WINDOW,
    ):
        self.tokenizer = tokenizer
        self.qwen3_mode = qwen3_mode
        self.phase_smoothing = max(1, phase_smoothing)
        self.text_window_chars = text_window_chars
        self.history_window = history_window
        self.reset()

    def reset(self) -> None:
        self.in_think = False
        self.steps_since_open = 9999
        self.steps_since_close = 9999
        self._text_buffer = ""
        self._recent_accept_lens: collections.deque = collections.deque(maxlen=self.history_window)
        self._recent_times: collections.deque = collections.deque(maxlen=self.history_window)
        self._recent_fail_pos: collections.deque = collections.deque(maxlen=self.history_window)
        self._raw_phase = PHASE_NON_THINK
        self._stable_phase = PHASE_NON_THINK
        self._phase_run_len = 0

    def prime_from_token_ids(self, token_ids: Iterable[int]) -> None:
        """Initialise ``in_think`` from a *prompt* token id sequence.

        Qwen3's chat template with ``add_generation_prompt=True`` writes
        ``<think>`` (151667) into the prompt itself when thinking mode is
        enabled, so the *generation* never re-emits the open tag. Without this
        priming the detector stays in NON_THINK forever and `custom/phase` /
        `custom/in_think` log as 0/False. Call this once per generation episode
        with the prompt + any pre-decoded tokens.
        """
        if not self.qwen3_mode:
            self.in_think = True  # non-Qwen3 fallback already treats all gen as in-think
            return

        last_open = -1
        last_close = -1
        for i, tid in enumerate(token_ids):
            tid_int = int(tid)
            if tid_int == QWEN3_THINK_OPEN_ID:
                last_open = i
            elif tid_int == QWEN3_THINK_CLOSE_ID:
                last_close = i

        if last_open == -1 and last_close == -1:
            self.in_think = False
        elif last_open > last_close:
            # Most recent boundary was an open -> we are inside a think block.
            self.in_think = True
            self.steps_since_open = 0
        else:
            self.in_think = False
            self.steps_since_close = 0

    def update(
        self,
        accepted_token_ids: Iterable[int],
        accept_len: float,
        time_cost: float,
    ) -> None:
        """Call once per draft-and-verify cycle, after verification."""
        ids = list(accepted_token_ids)

        if self.qwen3_mode:
            for tid in ids:
                tid_int = int(tid)
                if tid_int == QWEN3_THINK_OPEN_ID:
                    self.in_think = True
                    self.steps_since_open = 0
                elif tid_int == QWEN3_THINK_CLOSE_ID:
                    self.in_think = False
                    self.steps_since_close = 0
        else:
            self.in_think = True

        self.steps_since_open += 1
        self.steps_since_close += 1

        # Maintain a short decoded text buffer for lexical features.
        if ids and self.tokenizer is not None:
            try:
                new_text = self.tokenizer.decode(ids, skip_special_tokens=False)
            except Exception:
                new_text = ""
            self._text_buffer = (self._text_buffer + new_text)[-self.text_window_chars :]

        self._recent_accept_lens.append(float(accept_len))
        self._recent_times.append(float(time_cost))
        # `accept_len` IS the first-mismatch position by definition (LA accepted, then mismatch).
        self._recent_fail_pos.append(float(accept_len))

        new_phase = self._infer_phase()
        if new_phase == self._raw_phase:
            self._phase_run_len += 1
        else:
            self._raw_phase = new_phase
            self._phase_run_len = 1
        if self._phase_run_len >= self.phase_smoothing:
            self._stable_phase = self._raw_phase

    def current_phase(self) -> int:
        return int(self._stable_phase)

    def get_features(self) -> np.ndarray:
        feats = np.zeros(PHASE_OBS_DIM, dtype=np.float32)

        # ---- phi one-hot (5) + in_think (1) ----
        phi = self._stable_phase
        if 0 <= phi < NUM_PHASES:
            feats[phi] = 1.0
        feats[NUM_PHASES] = 1.0 if self.in_think else 0.0

        # ---- recent-history block (18) ----
        offset = _PHI_DIM
        # most-recent-first ordering: deque preserves insertion order, so iterate reversed.
        for i, (a, t, fp) in enumerate(
            zip(reversed(self._recent_accept_lens), reversed(self._recent_times), reversed(self._recent_fail_pos))
        ):
            base = offset + i * 3
            feats[base] = min(a / 8.0, 2.0)
            feats[base + 1] = min(t / 0.05, 2.0)  # 50ms scale
            feats[base + 2] = min(fp / 8.0, 2.0)

        # ---- risk features (8) ----
        risk_off = _PHI_DIM + _HIST_DIM
        text = self._text_buffer[-64:].lower() if self._text_buffer else ""
        text_len = max(1, len(text))
        if text:
            feats[risk_off + 0] = sum(c.isdigit() for c in text) / text_len
            feats[risk_off + 1] = sum(c in OP_CHARS for c in text) / text_len
            feats[risk_off + 2] = sum(c == "\n" for c in text) / text_len
        feats[risk_off + 3] = 1.0 if any(m in text for m in STEP_MARKERS) else 0.0
        feats[risk_off + 4] = 1.0 if any(kw in text for kw in TURN_KEYWORDS) else 0.0
        feats[risk_off + 5] = 1.0 if any(m in text for m in FINAL_MARKERS) else 0.0
        feats[risk_off + 6] = 1.0 / (1.0 + min(self.steps_since_open, 50))
        feats[risk_off + 7] = 1.0 / (1.0 + min(self.steps_since_close, 50))

        return feats

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _infer_phase(self) -> int:
        if not self.in_think:
            return PHASE_NON_THINK

        # Long window for keyword presence; short window for content density.
        # Older buffer content shouldn't dilute current numeric intent.
        text_long = (
            self._text_buffer[-self.text_window_chars :].lower() if self._text_buffer else ""
        )
        text_short = self._text_buffer[-96:].lower() if self._text_buffer else ""

        if any(kw in text_short for kw in TURN_KEYWORDS):
            return PHASE_SELF_CHECK

        if text_short:
            n = len(text_short)
            digit_ratio = sum(c.isdigit() for c in text_short) / n
            op_ratio = sum(c in OP_CHARS for c in text_short) / n
            has_code = "```" in text_short
            if digit_ratio > 0.10 or op_ratio > 0.04 or has_code:
                return PHASE_NUMERIC

        if any(kw in text_long for kw in EXPAND_KEYWORDS):
            return PHASE_REASONING_EXPAND

        # Right after `<think>` open, default to setup; otherwise expand.
        if self.steps_since_open < 4:
            return PHASE_PROBLEM_SETUP
        return PHASE_REASONING_EXPAND
