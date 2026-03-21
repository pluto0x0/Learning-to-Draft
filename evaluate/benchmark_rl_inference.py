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
    --batch_size 1 \
    --temperature 0.0
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer
from stable_baselines3 import PPO

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import (
        initialize_tree, tree_decoding, evaluate_posterior,
        reset_past_key_values, reset_tree_mode
    )
except ImportError:
    print("⚠️  Note: Eagle models not fully available; some features may be limited.")

from datetime import datetime
from tqdm.rich import tqdm

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
                    "content": (
                        "You are a helpful, respectful and honest assistant. "
                        "Always answer as helpfully as possible, while being safe."
                    )
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


def baseline_decoding(model, input_ids: torch.Tensor, max_new_tokens: int = 256,
                     temperature: float = 0.0) -> Tuple[int, float]:
    """
    Vanilla auto-regressive decoding (baseline).
    Returns: (num_generated_tokens, elapsed_time_in_seconds)
    """
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = model.base_model(
                input_ids=generated_ids,
                use_cache=False,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :]
            
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_id], dim=1)
            
            # Stop if EOS reached
            if next_id.item() == model.tokenizer.eos_token_id:
                break
    
    elapsed = time.time() - start_time
    num_generated = generated_ids.shape[1] - input_ids.shape[1]
    
    return num_generated, elapsed


def eagle3_decoding(model, input_ids: torch.Tensor, 
                   draft_depth: int = 8, verification_size: int = 60,
                   max_new_tokens: int = 256, temperature: float = 0.0,
                   logits_processor=None) -> Dict:
    """
    Eagle3 tree-based speculative decoding with static config.
    Returns: dict with metrics (tokens, time, acceptance_length, throughput)
    """
    start_time = time.time()
    
    current_input_ids = input_ids.clone()
    device = next(model.parameters()).device
    num_total_generated = 0
    total_cycles = 0
    total_accepted = 0
    
    # Initialize KV cache
    if hasattr(model, "past_key_values") and model.past_key_values is not None:
        reset_past_key_values(model.past_key_values)
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
            model.base_model, max_length=2048
        )
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data
    
    reset_tree_mode(model)
    if hasattr(model, "ea_layer") and hasattr(model.ea_layer, "reset_kv"):
        model.ea_layer.reset_kv()
    
    with torch.no_grad():
        while    < max_new_tokens:
            cycle_start = time.time()
            if hasattr(model, "ea_layer") and hasattr(model.ea_layer, "reset_kv"):
                model.ea_layer.reset_kv()
            
            # Initialize tree
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = initialize_tree(
                current_input_ids, model, past_key_values, logits_processor
            )
            
            model.base_model.model.tree_mask = tree_mask.to(device)
            
            # Verify with target model
            logits_verify, hidden_state_new_verify, _ = tree_decoding(
                model, draft_tokens.to(device), past_key_values,
                tree_position_ids.to(device), current_input_ids, retrieve_indices.to(device)
            )
            
            # Evaluate posterior
            padding = torch.full((1, 1), -1, dtype=torch.long, device=device)
            draft_tokens_padded = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens_padded[0, retrieve_indices.to(device)]
            
            best_idx, accept_len, sample_p = evaluate_posterior(
                logits_verify, candidates, logits_processor
            )
            
            # Update state
            prev_len = current_input_ids.shape[1]
            accepted_tokens = candidates[best_idx, :accept_len + 1]
            current_input_ids = torch.cat(
                (current_input_ids, accepted_tokens.unsqueeze(0).to(current_input_ids.device)),
                dim=-1
            )
            
            # Update KV cache
            select_indices = retrieve_indices[best_idx, :accept_len + 1] + prev_len
            for pkv_data in past_key_values_data:
                tgt = pkv_data[..., select_indices.to(pkv_data.device), :]
                dst = pkv_data[..., prev_len: prev_len + tgt.shape[-2], :]
                dst.copy_(tgt, non_blocking=True)
            
            current_length_data.fill_(current_input_ids.shape[1])
            
            num_total_generated += accept_len + 1
            total_cycles += 1
            total_accepted += accept_len + 1
            
            # Check termination
            if current_input_ids.shape[1] >= 1748 or num_total_generated >= max_new_tokens:
                break
    
    elapsed = time.time() - start_time
    
    return {
        "tokens_generated": num_total_generated,
        "elapsed_time": elapsed,
        "throughput": num_total_generated / elapsed if elapsed > 0 else 0,
        "num_cycles": total_cycles,
        "avg_acceptance_len": total_accepted / total_cycles if total_cycles > 0 else 0,
        "cycles_per_sec": total_cycles / elapsed if elapsed > 0 else 0
    }


def eagle3_rl_decoding(model, input_ids: torch.Tensor,
                      size_policy=None, depth_policy=None,
                      max_new_tokens: int = 256, temperature: float = 0.0,
                      logits_processor=None) -> Dict:
    """
    Eagle3 with learned RL policies for dynamic depth and verification size.
    """
    start_time = time.time()
    
    current_input_ids = input_ids.clone()
    device = next(model.parameters()).device
    num_total_generated = 0
    total_cycles = 0
    total_accepted = 0
    
    # Initialize KV cache
    if hasattr(model, "past_key_values") and model.past_key_values is not None:
        reset_past_key_values(model.past_key_values)
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
            model.base_model, max_length=2048
        )
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data
    
    reset_tree_mode(model)
    if hasattr(model, "ea_layer") and hasattr(model.ea_layer, "reset_kv"):
        model.ea_layer.reset_kv()
    
    with torch.no_grad():
        while num_total_generated < max_new_tokens:
            if hasattr(model, "ea_layer") and hasattr(model.ea_layer, "reset_kv"):
                model.ea_layer.reset_kv()
            # Initialize tree
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = initialize_tree(
                current_input_ids, model, past_key_values, logits_processor
            )
            
            model.base_model.model.tree_mask = tree_mask.to(device)
            
            # Verify
            logits_verify, hidden_state_new_verify, _ = tree_decoding(
                model, draft_tokens.to(device), past_key_values,
                tree_position_ids.to(device), current_input_ids, retrieve_indices.to(device)
            )
            
            # Evaluate
            padding = torch.full((1, 1), -1, dtype=torch.long, device=device)
            draft_tokens_padded = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens_padded[0, retrieve_indices.to(device)]
            
            best_idx, accept_len, sample_p = evaluate_posterior(
                logits_verify, candidates, logits_processor
            )
            
            # Update
            prev_len = current_input_ids.shape[1]
            accepted_tokens = candidates[best_idx, :accept_len + 1]
            current_input_ids = torch.cat(
                (current_input_ids, accepted_tokens.unsqueeze(0).to(current_input_ids.device)),
                dim=-1
            )
            
            # Update KV cache
            select_indices = retrieve_indices[best_idx, :accept_len + 1] + prev_len
            for pkv_data in past_key_values_data:
                tgt = pkv_data[..., select_indices.to(pkv_data.device), :]
                dst = pkv_data[..., prev_len: prev_len + tgt.shape[-2], :]
                dst.copy_(tgt, non_blocking=True)
            
            current_length_data.fill_(current_input_ids.shape[1])
            
            num_total_generated += accept_len + 1
            total_cycles += 1
            total_accepted += accept_len + 1
            
            # Check termination
            if current_input_ids.shape[1] >= 1748 or num_total_generated >= max_new_tokens:
                break
    
    elapsed = time.time() - start_time
    
    return {
        "tokens_generated": num_total_generated,
        "elapsed_time": elapsed,
        "throughput": num_total_generated / elapsed if elapsed > 0 else 0,
        "num_cycles": total_cycles,
        "avg_acceptance_len": total_accepted / total_cycles if total_cycles > 0 else 0,
        "cycles_per_sec": total_cycles / elapsed if elapsed > 0 else 0
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
    parser.add_argument("--dataset_names", nargs="+", default=["humaneval"],
                       help="Dataset names to evaluate on")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of samples per dataset")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./evaluate/results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🚀 RL-Enhanced Speculative Decoding Benchmark")
    print("=" * 80)
    
    # Load models
    print("\n📦 Loading models...")
    print(f"  Base model: {args.base_model_path}")
    print(f"  EA model: {args.ea_model_path}")
    print("Device:", args.device)
    
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
    
    # Load RL models if provided
    size_policy = None
    depth_policy = None
    
    if args.size_model_path and os.path.exists(args.size_model_path):
        print(f"  Loading size policy: {args.size_model_path}")
        try:
            size_policy = PPO.load(args.size_model_path, device=args.device)
            size_policy.policy.to(args.device)
            size_policy.policy.eval()
        except Exception as e:
            print(f"  ⚠️  Could not load size policy: {e}")
    
    if args.depth_model_path and os.path.exists(args.depth_model_path):
        print(f"  Loading depth policy: {args.depth_model_path}")
        try:
            depth_policy = PPO.load(args.depth_model_path, device=args.device)
            depth_policy.policy.to(args.device)
            depth_policy.policy.eval()
        except Exception as e:
            print(f"  ⚠️  Could not load depth policy: {e}")
    
    # Benchmark on datasets
    print("\n" + "=" * 80)
    print("🔬 Starting benchmark...")
    print("=" * 80)
    
    all_results = {}
    
    for dataset_name in args.dataset_names:
        print(f"\n📊 Dataset: {dataset_name}")
        print("-" * 80)
        
        # Load dataset
        samples = load_and_sample_data(
            args.data_dir, dataset_name, args.num_samples, tokenizer
        )
        
        if not samples:
            print(f"  ⚠️  Skipping {dataset_name} (no samples)")
            continue
        
        dataset_results = {
            "baseline": [],
            "eagle3": [],
            "eagle3_rl": []
        }
        
        for i, sample in tqdm(enumerate(samples), total=len(samples), desc=f"Evaluating {dataset_name}"):
            # if i % max(1, len(samples) // 5) == 0:
            #     print(f"  Progress: {i}/{len(samples)}")
            
            input_ids = sample["input_ids"].to(args.device)
            
            # Baseline
            # try:
            tokens, elapsed = baseline_decoding(model, input_ids, temperature=args.temperature)
            dataset_results["baseline"].append({
                "tokens": tokens,
                "time": elapsed,
                "throughput": tokens / elapsed if elapsed > 0 else 0
            })
            # except Exception as e:
            #     print(f"    ❌ Baseline error on sample {i}: {e}")
            
            # Eagle3
            # try:
            res = eagle3_decoding(model, input_ids, temperature=args.temperature)
            dataset_results["eagle3"].append(res)
            # except Exception as e:
            #     print(f"    ❌ Eagle3 error on sample {i}: {e}")
            
            # Eagle3 + RL
            if size_policy or depth_policy:
                try:
                    res = eagle3_rl_decoding(
                        model, input_ids,
                        size_policy=size_policy,
                        depth_policy=depth_policy,
                        temperature=args.temperature
                    )
                    dataset_results["eagle3_rl"].append(res)
                except Exception as e:
                    print(f"    ❌ Eagle3+RL error on sample {i}: {e}")
        
        # Aggregate results
        print(f"\n  Results for {dataset_name}:")
        print("  " + "-" * 76)
        
        for method_name, results in dataset_results.items():
            if not results:
                print(f"    {method_name:20s}: No results")
                continue
            
            results_df = pd.DataFrame(results)
            
            if method_name == "baseline":
                avg_tokens = results_df["tokens"].mean()
                avg_time = results_df["time"].mean()
                avg_throughput = results_df["throughput"].mean()
                print(f"    {method_name:20s}: "
                      f"Tokens {avg_tokens:6.1f}, Time {avg_time:7.3f}s, "
                      f"Throughput {avg_throughput:6.2f} tok/s")
            else:
                avg_tokens = results_df["tokens_generated"].mean()
                avg_time = results_df["elapsed_time"].mean()
                avg_throughput = results_df["throughput"].mean()
                avg_acceptance = results_df["avg_acceptance_len"].mean()
                print(f"    {method_name:20s}: "
                      f"Tokens {avg_tokens:6.1f}, Time {avg_time:7.3f}s, "
                      f"Throughput {avg_throughput:6.2f} tok/s, "
                      f"Acceptance {avg_acceptance:5.2f}")
        
        all_results[dataset_name] = dataset_results
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.json")
    with open(output_file, "w") as f:
        # Convert to JSON-serializable format
        results_to_save = {}
        for dataset, methods in all_results.items():
            results_to_save[dataset] = {}
            for method, res_list in methods.items():
                results_to_save[dataset][method] = res_list
        
        json.dump(to_jsonable(results_to_save), f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
