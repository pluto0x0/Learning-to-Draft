"""Phase-aware LTD training launcher for the **size** policy on Qwen3.

Mirrors :mod:`rl.rl_depth_phase` for the verification-size action. Uses the
existing ``rl/rl_total.py`` env unchanged via deferred import.

Example:
    python3 -m rl.rl_total_phase \\
        --base_model_path Qwen/Qwen3-8B \\
        --ea_model_path <eagle3-qwen3-path> \\
        --dataset_train gsm8k \\
        --total_timesteps 100000 \\
        --use_phase --reward_mode blend
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

from eagle.model.ea_model import EaModel
from rl.envs import PhaseObsWrapper, load_base_env_classes
from rl.phase import PhaseDetector, RewardShaper


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-aware LTD: size policy")
    p.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--ea_model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./eagle/data")
    p.add_argument("--dataset_train", type=str, default="gsm8k")
    p.add_argument("--save_path", type=str, default="./checkpoints")
    p.add_argument("--rl_checkpoint_path", type=str, default="")

    p.add_argument("--n_steps", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--total_timesteps", type=int, default=100_000)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--eval_freq", type=int, default=10000)
    p.add_argument("--pi_arch", type=int, nargs="+", default=[1024, 256])
    p.add_argument("--vf_arch", type=int, nargs="+", default=[1024, 256])

    p.add_argument("--use_phase", action="store_true")
    p.add_argument("--reward_mode", choices=["cycle", "blend", "segment"], default="cycle")
    p.add_argument("--blend_alpha", type=float, default=0.5)
    p.add_argument("--phase_smoothing", type=int, default=2)
    p.add_argument("--qwen3_mode", action="store_true", default=True)
    p.add_argument("--no_qwen3_mode", dest="qwen3_mode", action="store_false")
    p.add_argument("--max_input_len", type=int, default=1748)
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Per-generation new-token budget. Raise above 256 for "
                        "Qwen3 / reasoning models so episodes can reach </think>. "
                        "Bounded above by the inner env's KV cache (~1748 - input_len).")
    p.add_argument("--system_prompt", type=str, default="")
    p.add_argument("--wandb_project", type=str, default="speculative-decoding-rl-phase")
    return p


def adawm_schedule(initial_lr: float, warmup_steps: int, total_timesteps: int):
    def func(progress_remaining: float) -> float:
        cur = total_timesteps * (1.0 - progress_remaining)
        if cur < warmup_steps:
            return initial_lr * (cur / warmup_steps) if warmup_steps > 0 else initial_lr
        decay = (cur - warmup_steps) / max(1.0, (total_timesteps - warmup_steps))
        return initial_lr * (1.0 - decay)
    return func


class PhaseTensorboardCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_saved_timestep = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            log = {}
            for k_src, k_dst in (
                ("token_right", "custom/token_right"),
                ("t_draft", "custom/t_draft"),
                ("cycle_reward", "custom/cycle_reward"),
                ("shaped_reward", "custom/shaped_reward"),
                ("total_token_chosen_action", "custom/total_token_chosen_action"),
                ("depth_chosen", "custom/depth_chosen"),
                ("current_seq_len", "custom/current_seq_len"),
                ("phase", "custom/phase"),
                ("in_think", "custom/in_think"),
            ):
                if k_src in info:
                    log[k_dst] = float(info[k_src]) if not isinstance(info[k_src], bool) else int(info[k_src])
            if log:
                wandb.log(log, step=self.num_timesteps)
        if self.save_freq > 0 and self.num_timesteps - self.last_saved_timestep >= self.save_freq:
            ckpt = os.path.join(self.save_path, f"step_{self.num_timesteps}")
            self.model.save(ckpt)
            self.last_saved_timestep = self.num_timesteps
        return True


def main() -> None:
    args = _build_parser().parse_args()
    model_name = os.path.basename(args.base_model_path.rstrip("/"))
    ckpt_dir = os.path.join(args.save_path, model_name, "size_phase")
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(project=args.wandb_project, config=vars(args),
               sync_tensorboard=True, monitor_gym=True, save_code=True)

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        depth=5,
        top_k=10,
        total_token=60,
        use_eagle3=True,
        use_dyn_len=False,
    ).to("cuda")
    model.eval()
    tokenizer = model.get_tokenizer()
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
            "{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>\n' }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
        )

    input_ids_list = []
    dataset_path = os.path.join(args.data_dir, args.dataset_train, "question.jsonl")
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.append({"role": "user", "content": data["turns"][0]})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            if ids.shape[1] <= args.max_input_len:
                input_ids_list.append(ids)
    if not input_ids_list:
        raise RuntimeError(f"no usable prompts found at {dataset_path}")

    _, SizeEnvBase = load_base_env_classes()
    env = SizeEnvBase(model, logits_processor=None, input_ids_list=input_ids_list)
    env.max_new_tokens = args.max_new_tokens
    if args.use_phase:
        detector = PhaseDetector(tokenizer=tokenizer,
                                 qwen3_mode=args.qwen3_mode,
                                 phase_smoothing=args.phase_smoothing)
        shaper = RewardShaper(mode=args.reward_mode, blend_alpha=args.blend_alpha)
        env = PhaseObsWrapper(env, phase_detector=detector, reward_shaper=shaper)

    policy_kwargs = dict(net_arch=dict(pi=args.pi_arch, vf=args.vf_arch))
    warmup = int(0.01 * args.total_timesteps)
    lr_sched = adawm_schedule(args.lr, warmup, args.total_timesteps)

    if args.rl_checkpoint_path and os.path.exists(args.rl_checkpoint_path):
        rl = PPO.load(args.rl_checkpoint_path, env=env,
                      learning_rate=lr_sched, verbose=1,
                      target_kl=args.target_kl,
                      tensorboard_log=os.path.join(ckpt_dir, "tensorboard"))
    else:
        rl = PPO(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs, verbose=1,
            n_steps=args.n_steps, batch_size=args.batch_size,
            n_epochs=args.n_epochs, gamma=args.gamma,
            ent_coef=args.ent_coef, learning_rate=lr_sched,
            tensorboard_log=os.path.join(ckpt_dir, "tensorboard"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    callbacks = CallbackList([
        PhaseTensorboardCallback(save_freq=args.eval_freq, save_path=ckpt_dir),
        WandbCallback(gradient_save_freq=0, verbose=2),
    ])
    rl.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callbacks)
    rl.save(os.path.join(ckpt_dir, "final"))
    wandb.finish()


if __name__ == "__main__":
    main()
