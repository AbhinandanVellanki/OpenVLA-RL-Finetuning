# OpenVLA-OFT-RL

PPO fine-tuning pipeline for **OpenVLA-7B** on the **LIBERO** benchmark, with LoRA-based adaptation and action-chunked control.

This repository is structured as a research-engineering artifact: clear train/eval entrypoints, reproducibility notes, and scripts for smoke testing and inference-rate benchmarking.

## Status

- Implemented: PPO training entrypoint, LIBERO evaluation entrypoint, OpenVLA actor wrapper, PPO buffers/core algos.
- Implemented: action-chunk generation (8 actions per model query), warmstart logic test, basic unit tests.
- In progress: polished reproducibility workflow for fresh machines and fully validated public quickstart.

## Repository Structure

```text
.
├── OpenVLA_PPO.py                 # Main PPO training entrypoint
├── evaluate_LIBERO.py             # Main LIBERO evaluation entrypoint
├── ppo/                           # PPO configs, losses, buffers
├── libero_rl/                     # LIBERO env wrappers and utilities
├── vla_oft/                       # OpenVLA actor/config and prismatic internals
├── scripts/
│   ├── infer_chunk_benchmark.py   # Inference/action-chunk speed benchmark
│   └── smoke_test.sh              # Lightweight sanity checks
├── start_ppo_training.sh          # Canonical training launcher (foreground + logs)
├── eval_results/                  # Evaluation JSON outputs
├── environment.yml                # Conda environment spec
└── requirements.txt               # Pip requirements
```

## What This Repo Supports

- `train`: PPO fine-tuning via `OpenVLA_PPO.py` (single-task + multi-task IDs).
- `eval`: LIBERO suite/task evaluation via `evaluate_LIBERO.py` with JSON output.
- `infer`: local forward-pass benchmark via `scripts/infer_chunk_benchmark.py` (dummy observation, measured control-rate).

## Environment Setup

### Option A: Conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate oft_rl
```

### Option B: pip

```bash
conda create -n oft_rl python=3.10 -y
conda activate oft_rl
pip install -r requirements.txt
```

## Data and Checkpoint Setup

- LIBERO environment/data prep instructions live in `libero_rl/README.md`.
- Checkpoint loading supports both:
  - local directory checkpoints (`use_local=True`, `pretrained_checkpoint=<local_dir_name_or_path>`)
  - HuggingFace Hub IDs (`use_local=False`, `pretrained_checkpoint=<repo_id>`)
- These are configured in `vla_oft/min_vla/config.py`.
- Evaluation can also override model source directly via `--checkpoint`.
- Training can resume from a prior run checkpoint with:
  - `--resume-checkpoint <path_to_.pt>`
  - `--resume-stage auto|warmup|rl`

## Quickstart (Minimal Pipeline)

1. Check scripts and imports:

```bash
scripts/smoke_test.sh
```

2. Run a short PPO training sanity run:

```bash
python OpenVLA_PPO.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --timesteps 1000 \
  --seed 0 \
  --no-wandb
```

3. Run evaluation (small episode count):

```bash
python evaluate_LIBERO.py \
  --task-suite libero_spatial \
  --task-ids 0 \
  --num-episodes 2 \
  --seed 0 \
  --output-dir eval_results
```

4. Run inference benchmark:

```bash
python scripts/infer_chunk_benchmark.py \
  --runs 20 \
  --warmup 5 \
  --gpu-id 0 \
  --output-json eval_results/infer_chunk_benchmark.json
```

## Training

Primary entrypoint:

```bash
python OpenVLA_PPO.py --help
```

Training is currently a two-phase pipeline:
- Phase 1 (`warmup`): behavior cloning (BC) trains the tokenized/action-chunk head against L1 rollout actions.
- Phase 2 (`rl`): execute tokenized rollouts only and optimize with PPO/GRPO.

Canonical launcher (recommended):

```bash
./start_ppo_training.sh --task-suite libero_spatial --task-id 0 --timesteps 100000 --seed 0
```

Resume from a previous run checkpoint (recommended to skip repeated BC warmup):

```bash
./start_ppo_training.sh \
  --task-suite libero_spatial \
  --task-id 0 \
  --timesteps 10000000 \
  --seed 0 \
  --resume-checkpoint artifacts/checkpoints/<old_run_name>/best_model_stage_warmup.pt \
  --resume-stage rl
```

Notes:
- `--resume-checkpoint` accepts absolute paths or repo-relative paths.
- In the launcher, if `--resume-checkpoint` is set and `--resume-stage` is omitted, it defaults to `rl`.
- Use `--resume-stage warmup` only if you intentionally want more BC before RL.

Direct CLI example:

```bash
python OpenVLA_PPO.py --task-suite libero_spatial --task-ids 0 1 2 3 --num-envs 4 --timesteps 200000 --seed 0
```

Direct CLI resume example:

```bash
python OpenVLA_PPO.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --timesteps 10000000 \
  --seed 0 \
  --checkpoint-dir artifacts/checkpoints/<new_run_name> \
  --resume-checkpoint artifacts/checkpoints/<old_run_name>/best_model_stage_warmup.pt \
  --resume-stage rl
```

Artifact conventions baked into launcher:

```bash
artifacts/logs/<run_name>.log
artifacts/checkpoints/<run_name>/
```

## Evaluation

Primary entrypoint:

```bash
python evaluate_LIBERO.py --help
```

Suite evaluation example:

```bash
python evaluate_LIBERO.py --task-suite libero_spatial --num-episodes 20
```

Explicit device/checkpoint example:

```bash
python evaluate_LIBERO.py \
  --task-suite libero_spatial \
  --num-episodes 20 \
  --device cuda:0 \
  --checkpoint /path/to/checkpoint_or_hf_id \
  --seed 0
```

Output is written to `eval_results/*.json` and includes:

- `overall_success_rate`
- `overall_mean_reward`
- per-task success/reward/episode-length stats

## Inference Demo and Control-Rate Benchmark

Use:

```bash
python scripts/infer_chunk_benchmark.py --runs 30 --warmup 5 --gpu-id 0
```

The script reports measured latency per chunk and derived control rate:

- `mean_latency_ms_per_chunk`
- `chunk_length`
- `derived_control_rate_hz`
- optional machine-readable save via `--output-json <path>`

## Canonical Runbook

Canonical runbook means one deterministic, end-to-end sequence to reproduce the same pipeline shape on a fresh machine.

1. Environment setup:
```bash
conda env create -f environment.yml
conda activate oft_rl
```
2. LIBERO prep:
```bash
# Follow exact setup in:
# libero_rl/README.md
```
3. Train:
```bash
./start_ppo_training.sh --task-suite libero_spatial --task-id 0 --timesteps 100000 --seed 0
```
4. Evaluate:
```bash
python evaluate_LIBERO.py --task-suite libero_spatial --task-ids 0 --num-episodes 20 --seed 0 --output-dir eval_results
```
5. Inference benchmark:
```bash
python scripts/infer_chunk_benchmark.py --runs 30 --warmup 5 --gpu-id 0 --output-json eval_results/infer_chunk_benchmark.json
```

## Reproducibility Notes

- Set deterministic seeds in eval (`--seed`) and keep task IDs explicit.
- Set deterministic seeds in training (`--seed`) and evaluation (`--seed`).
- Keep model checkpoint/source fixed per run and log it alongside outputs.
- Use pinned dependencies from `environment.yml` for closest reproducibility.
- Action chunking uses `NUM_ACTIONS_CHUNK=8` in current LIBERO setup.

### Hardware Notes

- Intended target: single/dual NVIDIA GPUs with >=24GB VRAM for OpenVLA-7B workflows.
- `scripts/infer_chunk_benchmark.py` is GPU-oriented; CPU mode is not practical.

## Results and Metric Provenance

### Computed in-repo

- `eval_results/libero_spatial_eval_20251205_175828.json` reports:
  - `overall_success_rate = 0.195` (19.5%)
  - `num_episodes_per_task = 20`
- `logs/ppo_training_success_bc.log` (phase-1 warmup run) shows:
  - warmup completed and RL handoff gate triggered at `global_step=25088`
  - BC metrics reached `train/bc_loss=2.86813`, `train/bc_accuracy=0.20549`
  - latest logged validation before crash reached `val/l1_success_rate=1.0` and `val/tokenized_success_rate=0.9`
  - run then failed during warmup-to-RL handoff with `ValueError: all input arrays must have the same shape` from `ppo/trajectory_buffer.py` when stacking `l1_actions`

### Reported (not yet reproduced in this repo snapshot)

- From `notes/DRL_final_report.pdf`:
  - `98.2%` success rate (`491/500`) for `moojink/openvla-7b-oft-finetuned-libero-spatial`
  - setup described as `10 tasks x 50 rollouts` using the official evaluation script
  - RL fine-tuning claim: improvement from `80%` to `98%` success
- From `notes/Improving VLA using online RL - DeepRL.pdf`:
  - two-stage training framing: BC warmup then PPO/GRPO fine-tuning
  - `100 Hz` execution target carried over from OpenVLA-OFT parallel decoding

These are currently treated as **reported** goals, not validated claims in this public artifact state.

## Known Gaps / TODO

- TODO: extend CI from `ci-lite` to full dependency + optional GPU smoke coverage.
- TODO: add a lightweight CPU-only smoke path if GPU libraries are unavailable.
- TODO: document recommended checkpoint naming/layout conventions.
- TODO: verify warmup-to-RL handoff stability for trajectory stacking (`l1_actions` shape handling in `ppo/trajectory_buffer.py`).

## Troubleshooting

1. `ModuleNotFoundError` for LIBERO/robosuite:
- Ensure `environment.yml` installation completed in the active env.

2. CUDA OOM during training:
- Reduce batch size / timesteps and disable data parallel.
- Validate no stale processes are holding GPU memory.

3. `flash-attn` build/install failures:
- Match CUDA/PyTorch versions from `environment.yml`.

4. No evaluation output JSON:
- Check `--output-dir` exists or allow script to create it.
- Confirm run was not started with `--no-save`.

5. Inference benchmark too slow:
- Verify GPU utilization and correct `--gpu-id`.
- Use warmup runs before measuring.

## Development Notes

Run lightweight checks before committing:

```bash
scripts/smoke_test.sh
```
