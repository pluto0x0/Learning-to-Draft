#!/usr/bin/env bash
set -euo pipefail

# Phase-aware LTD: size policy training launcher.
# Mirrors train_size.sh but invokes rl/rl_total_phase.py and adds phase flags.

gpu=0
base_model_path="Qwen/Qwen3-8B"
ea_model_path=""
rl_checkpoint_path=""
data_dir="./eagle/data"
dataset_train="gsm8k"
save_path="./checkpoints"

total_timesteps=100000
n_steps=2048
batch_size=256
n_epochs=20
lr=1e-3
ent_coef=0.01
gamma=0.9
pi_arch="1024 256"
vf_arch="1024 256"

use_phase=1
reward_mode="cycle"     # cycle | blend | segment
blend_alpha=0.5
phase_smoothing=2
qwen3_mode=1

usage() {
    cat <<EOF
Usage: bash train_size_phase.sh [OPTIONS]
See train_depth_phase.sh for the full flag list (this script accepts the same options).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)                gpu="$2";                shift 2 ;;
        --base_model_path)    base_model_path="$2";    shift 2 ;;
        --ea_model_path)      ea_model_path="$2";      shift 2 ;;
        --rl_checkpoint_path) rl_checkpoint_path="$2"; shift 2 ;;
        --data_dir)           data_dir="$2";           shift 2 ;;
        --dataset_train)      dataset_train="$2";      shift 2 ;;
        --save_path)          save_path="$2";          shift 2 ;;
        --total_timesteps)    total_timesteps="$2";    shift 2 ;;
        --n_steps)            n_steps="$2";            shift 2 ;;
        --batch_size)         batch_size="$2";         shift 2 ;;
        --n_epochs)           n_epochs="$2";           shift 2 ;;
        --lr)                 lr="$2";                 shift 2 ;;
        --ent_coef)           ent_coef="$2";           shift 2 ;;
        --gamma)              gamma="$2";              shift 2 ;;
        --pi_arch)            pi_arch="$2";            shift 2 ;;
        --vf_arch)            vf_arch="$2";            shift 2 ;;
        --use_phase)          use_phase=1;             shift   ;;
        --no_phase)           use_phase=0;             shift   ;;
        --reward_mode)        reward_mode="$2";        shift 2 ;;
        --blend_alpha)        blend_alpha="$2";        shift 2 ;;
        --phase_smoothing)    phase_smoothing="$2";    shift 2 ;;
        --qwen3_mode)         qwen3_mode=1;            shift   ;;
        --no_qwen3_mode)      qwen3_mode=0;            shift   ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$ea_model_path" ]]; then
    echo "ERROR: --ea_model_path is required." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$gpu

phase_flag=""; [[ "$use_phase" -eq 1 ]] && phase_flag="--use_phase"
qwen3_flag="--qwen3_mode"; [[ "$qwen3_mode" -eq 0 ]] && qwen3_flag="--no_qwen3_mode"

# shellcheck disable=SC2086
python3 -m rl.rl_total_phase \
    --base_model_path "${base_model_path}" \
    --ea_model_path "${ea_model_path}" \
    --rl_checkpoint_path "${rl_checkpoint_path}" \
    --data_dir "${data_dir}" \
    --dataset_train "${dataset_train}" \
    --save_path "${save_path}" \
    --total_timesteps ${total_timesteps} \
    --n_steps ${n_steps} \
    --batch_size ${batch_size} \
    --n_epochs ${n_epochs} \
    --lr ${lr} \
    --ent_coef ${ent_coef} \
    --gamma ${gamma} \
    --pi_arch ${pi_arch} \
    --vf_arch ${vf_arch} \
    --reward_mode ${reward_mode} \
    --blend_alpha ${blend_alpha} \
    --phase_smoothing ${phase_smoothing} \
    ${phase_flag} ${qwen3_flag}
