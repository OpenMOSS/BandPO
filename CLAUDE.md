# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BandPO** is a novel reinforcement learning algorithm for LLM post-training that replaces canonical PPO/GRPO clipping with probability-aware dynamic bounds derived from f-divergence trust regions. The core algorithm maps KL, Total Variation, or Pearson χ² constraints into dynamic clipping intervals.

## Environment Setup

### Initial Setup (Required Once)

```bash
# Initialize environment variables, HuggingFace/W&B tokens, and download datasets/models
source init.sh
```

This sets:
- `BandPODir` - Project root path
- `BandPODir_LargeData` - Data directory (defaults to `$BandPODir/data`)
- `HUGGING_FACE_USERNAME`, `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN`
- `WANDB_API_KEY`

### Starting Ray Cluster (Required Before Training)

```bash
# Start Ray head node for distributed training
bash utils/ray/initialization.sh
```

This starts Ray on the current machine with dashboard at `<IP>:8265`.

## Common Development Commands

### Linting

```bash
cd RLtraining/verl

# Check code with ruff
ruff check verl/

# Format code
ruff format verl/
```

Configuration is in `RLtraining/verl/pyproject.toml` (line-length: 120).

### Running Training

```bash
cd RLtraining/verl/baselinescripts

# BandPO with KL divergence, radius=0.05
bash DeepSeek-R1-Distill-Qwen-1.5B-L4k/bandpo__grpo_plus_ray_bandkl005.sh

# GRPO baseline
bash DeepSeek-R1-Distill-Qwen-1.5B-L4k/grpo_official.sh

# BandPO with Total Variation
bash DeepSeek-R1-Distill-Qwen-1.5B-L4k/bandpo__grpo_plus_ray_bandtv005.sh

# BandPO with Pearson Chi²
bash DeepSeek-R1-Distill-Qwen-1.5B-L4k/bandpo__grpo_plus_ray_bandchi2005.sh
```

Training scripts use Ray job submission. Logs are at `/tmp/ray/session_<ID>/logs/job-driver-raysubmit_<ID>.log`.

### Monitoring Training

```bash
# Check Ray status
ray status

# View Ray dashboard (forward port if on remote)
# http://<ray_head_ip>:8265

# Watch job logs
ray job logs <job_id>

# Sync wandb logs
bash utils/wandb/sync_auto_by_dir.sh
```

## Repository Architecture

### Core Algorithm

**`RLtraining/verl/verl/bandpo/band/band.py`** - Main BandPO operator dispatcher
- Entry point: `band(old_log_prob, method="bandkl", delta=0.05, ...)`
- Supported methods: `"bandkl"`, `"bandtv"`, `"bandchi2"`
- Returns dynamic clipping bounds `(lower, upper)` as tensors

**`RLtraining/verl/verl/bandpo/band/solver.py`** - Universal numerical solver
- `universal_bisection_solver()` - Bisection method for f-divergence constraints
- `check_simplex_saturation()` - Proposition 3 saturation checking
- Handles numerical edge cases for p→0 and p→1

### Training Framework (verl)

The training framework is a modified version of [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLM):

```
RLtraining/verl/
├── verl/
│   ├── bandpo/              # BandPO algorithm implementations
│   │   ├── band/            # Core operator (band.py, solver.py)
│   │   ├── baseline_dcpo/   # DCPO baseline
│   │   └── kl2clipbound/    # Legacy KL implementation
│   ├── workers/             # Ray workers for distributed training
│   │   ├── fsdp_workers.py  # Main FSDP worker implementations
│   │   ├── actor/           # Actor implementations
│   │   ├── rollout/         # vLLM/SGLang rollout engines
│   │   └── reward_manager/  # Reward computation
│   ├── trainer/             # Trainer implementations
│   └── protocol.py          # Data protocol definitions
├── recipe/dapo/             # DAPO-style training recipes
│   ├── main_dapo.py         # Entry point
│   └── dapo_ray_trainer.py  # Ray trainer implementation
└── baselinescripts/         # Experiment launch scripts
```

### Key Configuration Options

BandPO-specific options in training scripts:

```bash
+actor_rollout_ref.actor.use_tokenwise_ratio_bounds=True \
+actor_rollout_ref.actor.tokenwise_ratio_bounds_method="bandkl" \
+actor_rollout_ref.actor.band_radius_delta=0.05 \
+actor_rollout_ref.actor.does_relax_high_p_bound=false \
+actor_rollout_ref.actor.upper_bound_max=10.0 \
```

### Data Directory Structure

```
data/
├── dataset/          # Training/evaluation datasets
│   ├── dapo/         # DAPO-Math-17k dataset
│   ├── gsm8k/        # GSM8K benchmark
│   ├── math-500/     # MATH-500 benchmark
│   ├── aime2024_dapo/# AIME 2024
│   └── ...
├── models/           # Base models (HuggingFace cache)
├── ckpts/            # Training checkpoints
├── records/          # Sample traces and validation records
├── wandb/            # Weights & Biases logs
└── RUNTIME_ENV/      # Ray runtime environment YAMLs
```

### Utility Scripts

```
utils/
├── ray/              # Ray cluster management
│   ├── initialization.sh   # Start Ray head
│   ├── status.sh           # Check Ray status
│   └── log.sh              # View Ray logs
├── datasets/         # Dataset processing
├── init/             # Initialization scripts
├── download/         # Model/dataset download helpers
└── apex/             # NVIDIA Apex (for FSDP/Megatron)
```

## Development Patterns

### Adding a New f-Divergence

To add a new divergence constraint:

1. Define generator function `f_new(u)` in `RLtraining/verl/verl/bandpo/band/band.py`
2. Add entry to `SUPPORTED_DIVERGENCES` with `"r_star"` lambda and `"has_analytical"` flag
3. If analytical solution exists, implement `_solve_bandnew_analytical()`
4. Otherwise, the universal bisection solver handles it automatically

### Modifying Training Hyperparameters

Training scripts are bash files that set environment variables and call `ray job submit`. Key parameters:

- `band_radius_delta` - Trust region radius (δ)
- `clip_ratio_high`/`clip_ratio_low` - Standard PPO clipping bounds
- `does_relax_high_p_bound` - Whether to relax bounds for high-probability tokens
- `train_batch_size`/`ppo_mini_batch_size`/`ppo_micro_batch_size` - Batch sizing

### Testing Changes

The codebase relies on end-to-end training runs rather than unit tests for algorithm validation. For quick iteration:

```bash
# Run a short test (1 epoch, small batch)
# Modify a baseline script to reduce:
#   total_epochs=1
#   train_batch_size=64
#   ppo_mini_batch_size=32
```

## Important Notes

- **Ray cluster must be running** before submitting training jobs
- **W&B logging** defaults to offline mode (`WANDB_MODE: offline` in runtime env); set to `"online"` for live syncing
- **CUDA 12.4+** required (see `verl.md` for installation without root)
- **Checkpoints** are saved to `$BandPODir_LargeData/ckpts/`
- The Band operator is designed to be **standalone** - copy `RLtraining/verl/verl/bandpo/band/` to integrate into other RL frameworks
