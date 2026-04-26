"""Gym wrapper that bolts phase-aware features and reward shaping onto the
existing LTD ``SpeculativeDecodingEnv`` (depth or size variant) without
changing those classes.

What it does each ``step()``:
    1. Snapshot ``current_input_ids.shape[1]`` before delegating to the inner env.
    2. After the inner env's verification step, read off the newly-accepted
       token ids and call ``PhaseDetector.update(...)``.
    3. Optionally shape the per-cycle reward via ``RewardShaper``.
    4. Concatenate ``PHASE_OBS_DIM`` extra features onto the inner observation.

What it does each ``reset()``:
    - If the underlying env is starting a fresh generation episode (``finished_overall_generation``
      was True before reset), reset the detector + shaper. The "logical generation"
      can span many PPO episodes (the size env terminates every cycle), so we key
      this off the inner env's own flag rather than reset count.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.phase import PHASE_OBS_DIM, PhaseDetector, RewardShaper


class PhaseObsWrapper(gym.Wrapper):
    """Append phase features to obs and (optionally) reshape the cycle reward."""

    def __init__(
        self,
        env: gym.Env,
        phase_detector: PhaseDetector,
        reward_shaper: Optional[RewardShaper] = None,
        extra_dim: int = PHASE_OBS_DIM,
    ):
        super().__init__(env)
        self.detector = phase_detector
        self.shaper = reward_shaper if reward_shaper is not None else RewardShaper(mode="cycle")
        self.extra_dim = int(extra_dim)

        old_space = env.observation_space
        if not isinstance(old_space, spaces.Box) or len(old_space.shape) != 1:
            raise ValueError(
                f"PhaseObsWrapper expects a 1-D Box observation space, got {old_space!r}"
            )
        new_shape = (old_space.shape[0] + self.extra_dim,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float32
        )

        # Re-export action space so PPO sees the wrapped env directly.
        self.action_space = env.action_space

    # ------------------------------------------------------------------
    # gym.Wrapper API
    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Detect the start of a fresh generation episode BEFORE the inner reset
        # flips the flag, so the detector is reset exactly once per generation.
        starting_new_generation = bool(
            getattr(self.env.unwrapped, "finished_overall_generation", True)
        )
        obs, info = self.env.reset(**kwargs)
        if starting_new_generation:
            self.detector.reset()
            self.shaper.reset()
        return self._extend_obs(obs), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        inner = self.env.unwrapped
        prev_seq_len = (
            inner.current_input_ids.shape[1]
            if getattr(inner, "current_input_ids", None) is not None
            else 0
        )

        obs, reward, terminated, truncated, info = self.env.step(action)

        accept_len = float(info.get("token_right", 0.0))
        time_cost = float(info.get("t_draft", 0.0))

        # Only update phase state once verification actually ran; depth env's
        # CONTINUE actions return terminated=False with no accepted tokens.
        new_token_ids = []
        if terminated and getattr(inner, "current_input_ids", None) is not None:
            cur_len = inner.current_input_ids.shape[1]
            if cur_len > prev_seq_len:
                new_token_ids = inner.current_input_ids[0, prev_seq_len:cur_len].tolist()

        if new_token_ids or terminated:
            self.detector.update(new_token_ids, accept_len, time_cost)

        phase = self.detector.current_phase()
        # Reward shaping only applies on cycle-completing steps; intermediate
        # depth-CONTINUE steps already return reward=0 in the base env.
        if terminated:
            shaped = self.shaper.shape(reward, phase, accept_len, time_cost)
        else:
            shaped = reward

        info = dict(info)
        info["phase"] = int(phase)
        info["in_think"] = bool(self.detector.in_think)
        info["cycle_reward"] = float(reward)
        info["shaped_reward"] = float(shaped)

        return self._extend_obs(obs), float(shaped), terminated, truncated, info

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _extend_obs(self, obs: np.ndarray) -> np.ndarray:
        feats = self.detector.get_features()
        if feats.shape[0] != self.extra_dim:
            raise RuntimeError(
                f"PhaseDetector returned {feats.shape[0]} features, expected {self.extra_dim}"
            )
        return np.concatenate([obs.astype(np.float32, copy=False), feats]).astype(np.float32)
