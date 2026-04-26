"""Optional reward shapers for phase-aware LTD.

`mode='cycle'` is identity (recovers the original LTD per-cycle throughput
reward); other modes are opt-in and are applied OUTSIDE the env via
`PhaseObsWrapper`, so the underlying env's reward computation is untouched.
"""

from __future__ import annotations

from typing import Optional


class RewardShaper:
    """Wrap the per-cycle throughput reward with optional smoothing/segment logic.

    Modes
    -----
    cycle    : pass-through (default; matches paper's R = L_A / T_total).
    blend    : alpha * R_cycle + (1 - alpha) * EMA(R_cycle), to soften local
               greedy choices without changing episode boundaries.
    segment  : running throughput within the current phase; resets when the
               phase index changes. Penalises locally-greedy decisions that
               hurt the segment-level throughput.

    `cycle` should always be tried first as a safety baseline; `blend` is the
    recommended second variant.
    """

    def __init__(
        self,
        mode: str = "cycle",
        blend_alpha: float = 0.5,
        ema_decay: float = 0.9,
    ):
        if mode not in {"cycle", "blend", "segment"}:
            raise ValueError(f"unknown reward mode: {mode}")
        self.mode = mode
        self.blend_alpha = float(blend_alpha)
        self.ema_decay = float(ema_decay)
        self.reset()

    def reset(self) -> None:
        self._ema: Optional[float] = None
        self._seg_phase: int = -1
        self._seg_acc_len: float = 0.0
        self._seg_acc_time: float = 1e-9

    def shape(
        self,
        cycle_reward: float,
        phase: int,
        accept_len: float = 0.0,
        time_cost: float = 0.0,
    ) -> float:
        if self._ema is None:
            self._ema = float(cycle_reward)
        else:
            self._ema = self.ema_decay * self._ema + (1.0 - self.ema_decay) * float(cycle_reward)

        if self.mode == "cycle":
            return float(cycle_reward)

        if self.mode == "blend":
            return self.blend_alpha * float(cycle_reward) + (1.0 - self.blend_alpha) * float(self._ema)

        # segment
        if phase != self._seg_phase:
            self._seg_phase = int(phase)
            self._seg_acc_len = 0.0
            self._seg_acc_time = 1e-9
        self._seg_acc_len += float(accept_len)
        self._seg_acc_time += float(time_cost) + 1e-9
        # Match the env's reward scale (env uses reward = (LA+1)/(time*100)).
        return self._seg_acc_len / (self._seg_acc_time * 100.0)
