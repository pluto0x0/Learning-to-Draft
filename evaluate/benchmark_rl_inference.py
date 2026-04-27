"""
Benchmark script for comparing baseline, Eagle3, and Eagle3+RL inference performance.
Loads pre-trained RL models and evaluates throughput, acceptance length, and latency.

使用方法:
python evaluate/benchmark_rl_inference.py \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --ea_model_path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --data_dir ./eagle/data \
    --dataset_names humaneval alpaca \
    --num_samples 100 \
    --batch_size 64 \
    --temperature 0.0
"""

import sys
import os
import argparse
import json
import time
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm.rich import tqdm

OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information."
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import (
        initialize_tree,
        tree_decoding,
        evaluate_posterior,
        reset_past_key_values,
        reset_tree_mode,
        prepare_logits_processor,
    )
except ImportError:
    print("⚠️  Note: Eagle models not fully available; some features may be limited.")


def to_jsonable(obj: Any):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_and_sample_data(data_dir: str, dataset_name: str, num_samples: int,
                         tokenizer, max_seq_len: int = 1748) -> List[Dict]:
    """Load and sample from dataset."""
    dataset_path = os.path.join(data_dir, dataset_name, "question.jsonl")

    if not os.path.exists(dataset_path):
        print(f"⚠️  Dataset {dataset_path} not found.")
        return []

    samples = []
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            messages = [
                {
                    "role": "system",
                    "content": OFFICIAL_SYSTEM_PROMPT
                },
                {"role": "user", "content": data["turns"][0]}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

            if input_ids.shape[1] <= max_seq_len:
                samples.append({
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "original_question": data["turns"][0]
                })

    print(f"✓ Loaded {len(samples)} samples from {dataset_name}")
    return samples


def _build_logits_processor(temperature: float):
    if temperature > 1e-5:
        return prepare_logits_processor(temperature=temperature, top_p=0.0, top_k=0)
    return None


def _get_stop_token_ids(model) -> set[int]:
    stop_ids = set()
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return stop_ids
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id >= 0:
            stop_ids.add(int(eot_id))
    except Exception:
        pass
    return stop_ids


def _should_stop(model, current_input_ids: torch.Tensor, input_len: int,
                 num_total_generated: int, max_new_tokens: int,
                 max_seq_len: int = 1748) -> bool:
    if current_input_ids.shape[1] >= max_seq_len or num_total_generated >= max_new_tokens:
        return True

    stop_ids = _get_stop_token_ids(model)
    if not stop_ids:
        return False

    generated_tokens = current_input_ids[0, input_len:].tolist()
    return any(token_id in stop_ids for token_id in generated_tokens)


def _init_kv_cache(model, max_length: int = 2048):
    if hasattr(model, "past_key_values") and model.past_key_values is not None:
        reset_past_key_values(model.past_key_values)
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
            model.base_model, max_length=max_length
        )
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    reset_tree_mode(model)
    if hasattr(model, "ea_layer") and hasattr(model.ea_layer, "reset_kv"):
        model.ea_layer.reset_kv()

    return past_key_values, past_key_values_data, current_length_data


def _copy_selected_kv(past_key_values_data, select_indices: torch.Tensor, prev_len: int):
    for pkv_data in past_key_values_data:
        tgt = pkv_data[..., select_indices.to(pkv_data.device), :]
        dst = pkv_data[..., prev_len: prev_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)


def _policy_predict_discrete(policy, obs_tensor: torch.Tensor) -> int:
    with torch.inference_mode():
        action = policy._predict(obs_tensor.unsqueeze(0), deterministic=True)
    return int(action.squeeze().item())


class EagleRLController:
    def __init__(self, model, current_input_ids: torch.Tensor, past_key_values,
                 past_key_values_data, current_length_data, logits_processor=None,
                 depth_policy=None, size_policy=None, phase_detector=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.current_input_ids = current_input_ids
        self.past_key_values = past_key_values
        self.past_key_values_data = past_key_values_data
        self.current_length_data = current_length_data
        self.logits_processor = logits_processor
        self.depth_policy = depth_policy
        self.size_policy = size_policy
        self.phase_detector = phase_detector
        self.ea_layer_top_k = model.ea_layer.top_k
        self.max_draft_depth = 12

        # When phase-aware checkpoints are loaded, the policy expects a wider
        # obs vector. Extend buffer sizes by PHASE_OBS_DIM (32) so the same
        # controller serves both vanilla and phase-aware policies.
        self._phase_dim = 0
        if self.phase_detector is not None:
            from rl.phase import PHASE_OBS_DIM
            self._phase_dim = PHASE_OBS_DIM
        self.obs_size = 1268 + self._phase_dim
        self.obs_size_depth = 128 + self._phase_dim

        self.cu_scores_for_obs = None
        self.random_depth_this_step = 0
        self.new_token_count = 0
        self.size_actions = []
        self.depth_stop_points = []
        self.depth_policy_calls = 0
        self.phase_history = []
        self._cycle_t0 = None
        self.obs_buffer = torch.zeros(self.obs_size, dtype=torch.float32, device=self.device)
        self.obs_buffer_depth = torch.zeros(self.obs_size_depth, dtype=torch.float32, device=self.device)

        if self.phase_detector is not None:
            try:
                prompt_ids = self.current_input_ids[0].tolist()
            except Exception:
                prompt_ids = []
            if prompt_ids:
                self.phase_detector.prime_from_token_ids(prompt_ids)

    def _append_phase_features(self, buffer: torch.Tensor, base_dim: int) -> None:
        if self.phase_detector is None or self._phase_dim == 0:
            return
        feats_np = self.phase_detector.get_features()
        feats = torch.from_numpy(feats_np).to(self.device, dtype=buffer.dtype)
        buffer[base_dim: base_dim + self._phase_dim] = feats

    def _phase_update_after_verify(self, accepted_token_ids, accept_len: int) -> None:
        if self.phase_detector is None:
            return
        cycle_time = 0.0
        if self._cycle_t0 is not None:
            cycle_time = max(0.0, time.perf_counter() - self._cycle_t0)
        try:
            ids = list(accepted_token_ids)
        except Exception:
            ids = []
        self.phase_detector.update(ids, float(accept_len), cycle_time)
        self.phase_history.append(int(self.phase_detector.current_phase()))

    def _get_obs_depth(self) -> torch.Tensor:
        self.obs_buffer_depth.zero_()
        if self.cu_scores_for_obs is not None:
            scores = self.cu_scores_for_obs.detach().float().flatten()
            self.obs_buffer_depth[0: scores.numel()] = scores
        self.obs_buffer_depth[100:114].fill_(self.current_input_ids.shape[1] / 1000.0)
        self.obs_buffer_depth[114:128].fill_(self.cnet_step / 10.0)
        self._append_phase_features(self.obs_buffer_depth, base_dim=128)
        return self.obs_buffer_depth

    def _get_obs(self) -> torch.Tensor:
        self.obs_buffer.zero_()
        scores = torch.cat(self.scores_list, dim=0).view(-1).detach().float()
        self.obs_buffer[0: scores.numel()] = scores
        self.obs_buffer[1210:1239].fill_(self.current_input_ids.shape[1] / 1000.0)
        self.obs_buffer[1239:1268].fill_(self.cnet_step / 10.0)
        self._append_phase_features(self.obs_buffer, base_dim=1268)
        return self.obs_buffer

    def bootstrap(self) -> int:
        self._cycle_t0 = time.perf_counter()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = initialize_tree(
            self.current_input_ids, self.model, self.past_key_values, self.logits_processor
        )
        self.model.base_model.model.tree_mask = tree_mask.to(self.device)
        logits_verify, hidden_state_new_verify, _ = tree_decoding(
            self.model,
            draft_tokens.to(self.device),
            self.past_key_values,
            tree_position_ids.to(self.device),
            self.current_input_ids,
            retrieve_indices.to(self.device),
        )

        padding = torch.full((1, 1), -1, dtype=torch.long, device=self.device)
        draft_tokens_padded = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens_padded[0, retrieve_indices.to(self.device)]
        best_candidate_idx, accept_length, sample_p = evaluate_posterior(
            logits_verify, candidates, self.logits_processor
        )

        prev_len = self.current_input_ids.shape[1]
        accepted_tokens = candidates[best_candidate_idx, :accept_length + 1]
        self.current_input_ids = torch.cat(
            (self.current_input_ids, accepted_tokens.unsqueeze(0).to(self.current_input_ids.device)),
            dim=-1,
        )

        select_indices = retrieve_indices[best_candidate_idx, :accept_length + 1] + prev_len
        _copy_selected_kv(self.past_key_values_data, select_indices, prev_len)
        self.current_length_data.fill_(self.current_input_ids.shape[1])

        retrieve_hidden_state_new = hidden_state_new_verify[:, retrieve_indices.to(hidden_state_new_verify.device)]
        accepted_hidden_state_base = retrieve_hidden_state_new[:, best_candidate_idx, :accept_length + 1]
        next_token_sampled = torch.argmax(sample_p).unsqueeze(0).unsqueeze(0)

        self.new_token_count += accept_length + 1
        self.accepted_hidden_state_base_for_next_topk = accepted_hidden_state_base
        self.next_token_sampled_for_next_topk = next_token_sampled
        self._phase_update_after_verify(accepted_tokens.tolist(), accept_length + 1)
        return int(accept_length + 1)

    def _prepare_for_drafting(self, accepted_hidden_state_base, next_token_sampled):
        self.hidden_states_for_topk_ea_layer = accepted_hidden_state_base
        self.input_ids_for_topk_first_pass = torch.cat(
            (self.current_input_ids, next_token_sampled.to(self.current_input_ids.device)), dim=1
        )
        self.current_sample_token_for_topk = self.input_ids_for_topk_first_pass[:, -1]

        self.scores_list = []
        self.parents_list = []
        self.ss_token_list = []

        input_ids_for_ea_layer_first_iter = self.input_ids_for_topk_first_pass[:, 1:]
        self.len_posi_for_topk_loop = input_ids_for_ea_layer_first_iter.shape[1]

        self.model.ea_layer.reset()
        kv_len = 0
        if hasattr(self.model.ea_layer, "stable_kv") and self.model.ea_layer.stable_kv is not None:
            kv_len = self.model.ea_layer.stable_kv[0][0].shape[2]
        input_ids_for_forward = input_ids_for_ea_layer_first_iter[:, kv_len:] if kv_len > 0 else input_ids_for_ea_layer_first_iter

        out_hidden, past_key_values_ealayer = self.model.ea_layer(
            self.hidden_states_for_topk_ea_layer,
            input_ids=input_ids_for_forward,
            past_key_values=self.model.ea_layer.stable_kv if kv_len > 0 else None,
            use_cache=True,
        )

        self.model.ea_layer.stable_kv = past_key_values_ealayer
        self.current_past_key_values_ealayer = past_key_values_ealayer

        last_hidden_ea_layer = out_hidden[:, -1]
        last_headout = self.model.ea_layer.lm_head(self.model.ea_layer.norm(last_hidden_ea_layer))
        last_p = self.model.ea_layer.logsoftmax(last_headout)
        top = torch.topk(last_p, self.ea_layer_top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        current_scores = topk_p[0]
        self.scores_list.append(current_scores[None])
        self.current_scores_for_topk_loop_obs = current_scores
        self.parents_list.append(torch.zeros(1, dtype=torch.long, device=current_scores.device))

        if self.model.ea_layer.config.vocab_size == self.model.ea_layer.config.draft_vocab_size:
            self.ss_token_list.append(topk_index)
            input_ids_for_next_depth_iter = topk_index
        else:
            mapped_tokens = topk_index + self.model.ea_layer.d2t[topk_index]
            self.ss_token_list.append(mapped_tokens)
            input_ids_for_next_depth_iter = mapped_tokens

        self.current_input_ids_for_topk_depth_iter = input_ids_for_next_depth_iter
        self.current_input_hidden_for_topk_depth_iter = last_hidden_ea_layer[None].repeat(1, self.ea_layer_top_k, 1)
        self.current_tree_mask_for_topk_loop = self.model.ea_layer.tree_mask_init.clone().to(self.device)
        self.current_topk_cs_index_for_loop = torch.arange(
            self.ea_layer_top_k, device=self.model.ea_layer.embed_tokens.weight.device
        )
        self.cnet_step = 0
        self.cu_scores_for_obs = None

    def _perform_dynamic_depth_expansion(self):
        self.random_depth_this_step = self.max_draft_depth
        for depth_idx in range(self.random_depth_this_step):
            self.model.ea_layer.tree_mask = self.current_tree_mask_for_topk_loop
            current_ea_layer_position_ids = self.len_posi_for_topk_loop + self.model.ea_layer.position_ids.to(self.device)

            out_hidden, past_key_values_ealayer_new = self.model.ea_layer(
                self.current_input_hidden_for_topk_depth_iter,
                input_ids=self.current_input_ids_for_topk_depth_iter,
                past_key_values=self.current_past_key_values_ealayer,
                position_ids=current_ea_layer_position_ids,
                use_cache=True,
            )
            self.len_posi_for_topk_loop += 1
            self.current_past_key_values_ealayer = past_key_values_ealayer_new

            bias1 = self.ea_layer_top_k if self.cnet_step > 0 else 0
            bias2 = max(0, self.cnet_step - 1)
            bias = 1 + self.ea_layer_top_k * self.ea_layer_top_k * bias2 + bias1

            parents = self.current_topk_cs_index_for_loop + bias
            self.parents_list.append(parents)

            last_headout = self.model.ea_layer.lm_head(self.model.ea_layer.norm(out_hidden[0]))
            last_p = self.model.ea_layer.logsoftmax(last_headout)
            top = torch.topk(last_p, self.ea_layer_top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + self.current_scores_for_topk_loop_obs[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), self.ea_layer_top_k, dim=-1)
            topk_cs_index_new, topk_cs_p_new = topk_cs.indices, topk_cs.values
            self.cu_scores_for_obs = cu_scores.flatten()
            self.current_scores_for_topk_loop_obs = topk_cs_p_new
            self.current_topk_cs_index_for_loop = topk_cs_index_new

            out_ids = (topk_cs_index_new // self.ea_layer_top_k).to(self.current_tree_mask_for_topk_loop.device)
            self.current_input_hidden_for_topk_depth_iter = out_hidden[:, out_ids]
            next_input_ids_val = topk_index.view(-1)[topk_cs_index_new][None]

            if self.model.ea_layer.config.vocab_size == self.model.ea_layer.config.draft_vocab_size:
                self.ss_token_list.append(topk_index)
                self.current_input_ids_for_topk_depth_iter = next_input_ids_val
            else:
                mapped_tokens = next_input_ids_val + self.model.ea_layer.d2t[next_input_ids_val.squeeze()].unsqueeze(0)
                self.ss_token_list.append(topk_index + self.model.ea_layer.d2t[topk_index.squeeze()])
                self.current_input_ids_for_topk_depth_iter = mapped_tokens

            self.scores_list.append(cu_scores)

            if self.current_tree_mask_for_topk_loop.shape[2] > 0 and out_ids.max() < self.current_tree_mask_for_topk_loop.shape[2]:
                self.current_tree_mask_for_topk_loop = torch.cat(
                    (
                        self.current_tree_mask_for_topk_loop[:, :, out_ids],
                        self.model.ea_layer.tree_mask_init.clone().to(self.device),
                    ),
                    dim=3,
                )
            self.cnet_step += 1

            if self.depth_policy is not None and depth_idx != self.random_depth_this_step - 1 and depth_idx % 3 == 2:
                self.depth_policy_calls += 1
                action_depth = _policy_predict_discrete(self.depth_policy, self._get_obs_depth())
                if action_depth == 0:
                    self.random_depth_this_step = depth_idx + 1
                    break

        self.depth_stop_points.append(int(self.random_depth_this_step))

    def _finalize_draft_tree(self, total_token_val_action: int):
        scores_cat = torch.cat(self.scores_list, dim=0).view(-1)
        ss_token_cat = torch.cat(self.ss_token_list, dim=0).view(-1)
        actual_total_tokens = min(ss_token_cat.shape[0], total_token_val_action)

        top_scores_indices = torch.topk(scores_cat, actual_total_tokens, dim=-1).indices
        top_scores_indices_sorted = torch.sort(top_scores_indices).values

        draft_tokens_flat = ss_token_cat[top_scores_indices_sorted]
        finalized_draft_tokens = torch.cat(
            (self.current_sample_token_for_topk.to(self.device), draft_tokens_flat), dim=0
        ).unsqueeze(0)

        num_final_draft_plus_sample = finalized_draft_tokens.shape[1]
        draft_parents_flat = torch.cat(self.parents_list, dim=0)[top_scores_indices_sorted // self.ea_layer_top_k].long()
        mask_index = torch.searchsorted(top_scores_indices_sorted, draft_parents_flat - 1, right=False)
        mask_index[draft_parents_flat == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask_bool = torch.eye(num_final_draft_plus_sample, device=self.device).bool()
        tree_mask_bool[:, 0] = True
        for i in range(actual_total_tokens):
            tree_mask_bool[i + 1].add_(tree_mask_bool[mask_index_list[i]])

        finalized_tree_mask = tree_mask_bool.float()[None, None]
        finalized_tree_position_ids = torch.sum(tree_mask_bool.int(), dim=1) - 1

        max_depth = torch.max(finalized_tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        leaf_num = actual_total_tokens - (len(noleaf_index) - 1)
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = finalized_tree_position_ids.tolist()
        for i in range(actual_total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1] if cid > 0 else -1
                rid += 1

        finalized_retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        return finalized_draft_tokens, finalized_tree_mask, finalized_tree_position_ids, finalized_retrieve_indices

    def run_cycle(self) -> int:
        self._cycle_t0 = time.perf_counter()
        self._prepare_for_drafting(
            self.accepted_hidden_state_base_for_next_topk,
            self.next_token_sampled_for_next_topk,
        )
        self._perform_dynamic_depth_expansion()

        if self.size_policy is None:
            action = 5
            total_token_val_action = 60
        else:
            action = _policy_predict_discrete(self.size_policy, self._get_obs())
            total_token_val_action = (action + 1) * 10

        self.size_actions.append(int(action))

        finalized_draft_tokens, finalized_tree_mask, finalized_tree_position_ids, finalized_retrieve_indices = (
            self._finalize_draft_tree(total_token_val_action)
        )

        self.model.base_model.model.tree_mask = finalized_tree_mask.to(self.device)
        logits_verify, hidden_state_new_verify, _ = tree_decoding(
            self.model,
            finalized_draft_tokens.to(self.device),
            self.past_key_values,
            finalized_tree_position_ids.to(self.device),
            self.current_input_ids,
            finalized_retrieve_indices.to(self.device),
        )

        padding = torch.full((1, 1), -1, dtype=torch.long, device=self.device)
        draft_tokens_padded = torch.cat((finalized_draft_tokens, padding), dim=1)
        candidates = draft_tokens_padded[0, finalized_retrieve_indices.to(self.device)]
        best_candidate_idx, accept_length, sample_p = evaluate_posterior(
            logits_verify, candidates, self.logits_processor
        )

        prev_len = self.current_input_ids.shape[1]
        accepted_tokens = candidates[best_candidate_idx, :accept_length + 1]
        self.current_input_ids = torch.cat(
            (self.current_input_ids, accepted_tokens.unsqueeze(0).to(self.current_input_ids.device)),
            dim=-1,
        )

        select_indices = finalized_retrieve_indices[best_candidate_idx, :accept_length + 1] + prev_len
        _copy_selected_kv(self.past_key_values_data, select_indices, prev_len)
        self.current_length_data.fill_(self.current_input_ids.shape[1])

        retrieve_hidden_state_new = hidden_state_new_verify[:, finalized_retrieve_indices.to(hidden_state_new_verify.device)]
        self.accepted_hidden_state_base_for_next_topk = retrieve_hidden_state_new[:, best_candidate_idx, :accept_length + 1]
        self.next_token_sampled_for_next_topk = torch.argmax(sample_p).unsqueeze(0).unsqueeze(0)
        self.new_token_count += accept_length + 1
        self._phase_update_after_verify(accepted_tokens.tolist(), accept_length + 1)
        return int(accept_length + 1)

    def get_stats(self) -> Dict[str, float]:
        avg_size_action = float(np.mean(self.size_actions)) if self.size_actions else 0.0
        avg_size_tokens = float(np.mean([(action + 1) * 10 for action in self.size_actions])) if self.size_actions else 0.0
        avg_depth_stop = float(np.mean(self.depth_stop_points)) if self.depth_stop_points else 0.0
        stats = {
            "avg_size_action": avg_size_action,
            "avg_size_tokens": avg_size_tokens,
            "avg_depth_stop": avg_depth_stop,
            "depth_policy_calls": float(self.depth_policy_calls),
        }
        if self.phase_history:
            from rl.phase import NUM_PHASES
            counts = np.bincount(self.phase_history, minlength=NUM_PHASES).astype(float)
            total = counts.sum()
            for i in range(NUM_PHASES):
                stats[f"phase_frac_{i}"] = float(counts[i] / total) if total > 0 else 0.0
        return stats


def baseline_decoding(model, input_ids: torch.Tensor, max_new_tokens: int = 256,
                      temperature: float = 0.0) -> Tuple[int, float]:
    """
    Vanilla auto-regressive decoding (baseline) aligned with the repo's official
    naivegenerate evaluation path.
    Returns: (num_generated_tokens, elapsed_time_in_seconds)
    """
    start_time = time.time()
    is_llama3 = 128009 in _get_stop_token_ids(model)
    with torch.no_grad():
        _, new_token, _ = model.naivegenerate(
            input_ids.clone(),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            log=True,
            is_llama3=is_llama3,
        )
    elapsed = time.time() - start_time
    return int(new_token), elapsed


def eagle3_decoding(model, input_ids: torch.Tensor,
                    draft_depth: int = 5, verification_size: int = 60,
                    max_new_tokens: int = 256, temperature: float = 0.0,
                    logits_processor=None) -> Dict:
    """
    Eagle3 tree-based speculative decoding with static config, using the repo's
    official eagenerate path so benchmark behavior matches the paper more closely.
    """
    del logits_processor
    start_time = time.time()
    is_llama3 = 128009 in _get_stop_token_ids(model)

    original_depth = model.ea_layer.depth
    original_total_tokens = model.ea_layer.total_tokens
    model.ea_layer.depth = draft_depth
    model.ea_layer.total_tokens = max(1, verification_size - 1)

    try:
        with torch.no_grad():
            _, new_token, _, _, accept_lengths = model.eagenerate(
                input_ids.clone(),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                log=True,
                is_llama3=is_llama3,
                pre_len=True,
            )
    finally:
        model.ea_layer.depth = original_depth
        model.ea_layer.total_tokens = original_total_tokens

    elapsed = time.time() - start_time
    total_cycles = len(accept_lengths)
    total_accepted = sum(accept_lengths)
    return {
        "tokens_generated": int(new_token),
        "elapsed_time": elapsed,
        "throughput": new_token / elapsed if elapsed > 0 else 0,
        "num_cycles": total_cycles,
        "avg_acceptance_len": (total_accepted / total_cycles) if total_cycles > 0 else 0,
        "cycles_per_sec": total_cycles / elapsed if elapsed > 0 else 0,
    }


def eagle3_rl_decoding(model, input_ids: torch.Tensor,
                       size_policy=None, depth_policy=None,
                       max_new_tokens: int = 256, temperature: float = 0.0,
                       logits_processor=None, phase_detector=None) -> Dict:
    """
    Eagle3 with learned RL policies for dynamic depth and verification size.
    This mirrors the reset/step logic in rl_total.py.
    """
    start_time = time.time()
    logits_processor = logits_processor if logits_processor is not None else _build_logits_processor(temperature)

    current_input_ids = input_ids.clone()
    input_len = current_input_ids.shape[1]
    num_total_generated = 0
    total_cycles = 0
    total_accepted = 0

    if phase_detector is not None:
        phase_detector.reset()

    past_key_values, past_key_values_data, current_length_data = _init_kv_cache(model)
    controller = EagleRLController(
        model=model,
        current_input_ids=current_input_ids,
        past_key_values=past_key_values,
        past_key_values_data=past_key_values_data,
        current_length_data=current_length_data,
        logits_processor=logits_processor,
        depth_policy=depth_policy,
        size_policy=size_policy,
        phase_detector=phase_detector,
    )

    with torch.no_grad():
        accepted = controller.bootstrap()
        num_total_generated += accepted
        total_accepted += accepted
        total_cycles += 1

        while num_total_generated < max_new_tokens:
            if _should_stop(model, controller.current_input_ids, input_len, num_total_generated, max_new_tokens):
                break

            accepted = controller.run_cycle()
            num_total_generated += accepted
            total_accepted += accepted
            total_cycles += 1

    elapsed = time.time() - start_time
    stats = controller.get_stats()
    return {
        "tokens_generated": num_total_generated,
        "elapsed_time": elapsed,
        "throughput": num_total_generated / elapsed if elapsed > 0 else 0,
        "num_cycles": total_cycles,
        "avg_acceptance_len": total_accepted / total_cycles if total_cycles > 0 else 0,
        "cycles_per_sec": total_cycles / elapsed if elapsed > 0 else 0,
        **stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark RL-enhanced inference")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ea_model_path", type=str, default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--size_model_path", type=str, default="",
                        help="Path to size RL model (.zip)")
    parser.add_argument("--depth_model_path", type=str, default="",
                        help="Path to depth RL model (.zip)")
    parser.add_argument("--data_dir", type=str, default="./eagle/data")
    parser.add_argument("--dataset_names", nargs="+", default=[],
                        help="Dataset names to evaluate on")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples per dataset")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./evaluate/results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_phase", action="store_true",
                        help="Enable phase-aware obs extension (required for "
                             "checkpoints trained with --use_phase).")
    parser.add_argument("--qwen3_mode", action="store_true", default=True,
                        help="Use Qwen3 <think>/</think> token IDs for phase gating.")
    parser.add_argument("--no_qwen3_mode", dest="qwen3_mode", action="store_false")
    parser.add_argument("--phase_smoothing", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 RL-Enhanced Speculative Decoding Benchmark")
    print("=" * 80)

    print("\n📦 Loading models...")
    print(f"  Base model: {args.base_model_path}")
    print(f"  EA model: {args.ea_model_path}")
    print(f"  Device: {args.device}")

    gpu_name = torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else "N/A"
    print(f"  GPU: {gpu_name}")

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
    ).to(args.device)
    model.eval()

    tokenizer = model.get_tokenizer()

    # Vicuna and similar older models do not have chat_template set in their tokenizer.
    # Manually set the Vicuna v1 conversation format so apply_chat_template works.
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
            "{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>\n' }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
        )

    size_policy = None
    depth_policy = None

    model_name = args.base_model_path.split("/")[-1]

    from pprint import pprint
    # Phase-aware checkpoints live under <model>/{depth_phase,size_phase}; default
    # to those when --use_phase is requested so the same launcher works for both.
    _size_subdir = "size_phase" if args.use_phase else "size"
    _depth_subdir = "depth_phase" if args.use_phase else "depth"
    if args.size_model_path == "":
        args.size_model_path = f"checkpoints/{model_name}/{_size_subdir}/final.zip"
        print(f"  ℹ️ No size model path provided, defaulting to {args.size_model_path}")
    if args.depth_model_path == "":
        args.depth_model_path = f"checkpoints/{model_name}/{_depth_subdir}/final.zip"
        print(f"  ℹ️ No depth model path provided, defaulting to {args.depth_model_path}")
    if args.dataset_names == []:
        # list all subdirectories in data_dir as dataset names
        args.dataset_names = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
        print(f"  ℹ️ No dataset names provided, defaulting to all datasets:")
        pprint(args.dataset_names)

    if args.size_model_path and os.path.exists(args.size_model_path):
        print(f"  Loading size policy: {args.size_model_path}")
        try:
            size_policy = PPO.load(args.size_model_path, device=args.device).policy
            size_policy.to(args.device)
            # UserWarning: You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU when not using a CNN policy (you are using ActorCriticPolicy which should be a MlpPolicy). See https://github.com/DLR-RM/stable-baselines3/issues/1245 for more info. You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force usingut it is primarily intended to run on the CPU w the CPU.Note: The model will train, but the GPU utilization will be poor and the training might take longer than on CPU.
            size_policy.eval()
        except Exception as e:
            print(f"  ⚠️  Could not load size policy: {e}")

    if args.depth_model_path and os.path.exists(args.depth_model_path):
        print(f"  Loading depth policy: {args.depth_model_path}")
        try:
            depth_policy = PPO.load(args.depth_model_path, device=args.device).policy
            depth_policy.to(args.device)
            depth_policy.eval()
        except Exception as e:
            print(f"  ⚠️  Could not load depth policy: {e}")

    phase_detector = None
    if args.use_phase:
        from rl.phase import PhaseDetector
        phase_detector = PhaseDetector(
            tokenizer=tokenizer,
            qwen3_mode=args.qwen3_mode,
            phase_smoothing=args.phase_smoothing,
        )
        print(f"  ℹ️  Phase-aware mode enabled (qwen3_mode={args.qwen3_mode}).")

    print("\n" + "=" * 80)
    print("🔬 Starting benchmark...")
    print("=" * 80)

    all_results = {
        "config": {
            "base_model_path": args.base_model_path,
            "ea_model_path": args.ea_model_path,
            "size_model_path": args.size_model_path,
            "depth_model_path": args.depth_model_path,
            "dataset_names": args.dataset_names,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "device": args.device,
            "gpu_name": gpu_name,
            "use_phase": args.use_phase,
            "qwen3_mode": args.qwen3_mode if args.use_phase else None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    pprint(all_results["config"])

    for dataset_name in args.dataset_names:
        print(f"\n📊 Dataset: {dataset_name}")
        print("-" * 80)

        samples = load_and_sample_data(args.data_dir, dataset_name, args.num_samples, tokenizer)
        if not samples:
            print(f"  ⚠️  Skipping {dataset_name} (no samples)")
            continue

        dataset_results = {
            "baseline": [],
            "eagle3": [],
            "eagle3_rl": [],
        }

        for i, sample in tqdm(enumerate(samples), total=len(samples), desc=f"Evaluating {dataset_name}"):
            input_ids = sample["input_ids"].to(args.device)

            tokens, elapsed = baseline_decoding(model, input_ids, temperature=args.temperature)
            dataset_results["baseline"].append({
                "tokens": tokens,
                "time": elapsed,
                "throughput": tokens / elapsed if elapsed > 0 else 0,
            })

            res = eagle3_decoding(model, input_ids, draft_depth=5, verification_size=60, temperature=args.temperature)
            dataset_results["eagle3"].append(res)

            if size_policy is not None or depth_policy is not None:
                try:
                    res = eagle3_rl_decoding(
                        model,
                        input_ids,
                        size_policy=size_policy,
                        depth_policy=depth_policy,
                        phase_detector=phase_detector,
                        temperature=args.temperature,
                    )
                    dataset_results["eagle3_rl"].append(res)
                except Exception as e:
                    print(f"    ❌ Eagle3+RL error on sample {i}: {e}")

        all_results[dataset_name] = dataset_results

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(to_jsonable(all_results), f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    from evaluate.analyze_results import print_results
    print_results(all_results)


if __name__ == "__main__":
    main()
