# Elsa Framework

Elsa Framework is a Python-based simulator workflow for evaluating MoE inference on heterogeneous GPU + PIM/NMP systems. The current main path is the Courier simulator in `courier/`, with Ramulator2 used for memory/PIM timing.

This repository is organized around four daily workflows:

- run a single Courier simulation with `courier_main.py`
- run batches of commands from `configs/evaluations/`
- profile DWAP Phase 2 latency surfaces with `scripts/profile/profile_latency_surface.py`
- aggregate raw logs and CSV outputs with scripts under `scripts/aggregate/`

The project was reorganized from the original `courier_demo` layout. See `docs/STRUCTURE_MIGRATION.md` for a detailed file-by-file migration record.

## Repository Layout

```text
elsa_framework/
  courier/                  # current Courier simulator package
  courier_main.py           # main simulation CLI
  legacy/                   # old baseline src/ and main.py kept for reference
  scripts/
    run/                    # batch runners
    profile/                # DWAP Phase 2 latency profiling
    aggregate/              # log/result aggregation scripts
    setup/                  # Ramulator2 patch/setup helpers
  configs/
    evaluations/            # command-list files used by batch runners
    feasible_region/        # Phase 1 feasible-region JSON files
  data/
    gate_weight/            # local expert stats / gate-weight inputs
  results/
    raw_logs/
      courier/              # Courier internal hardware logs
      process/              # stdout/stderr process logs
    workload/               # per-workload gpu_result / pimoe_result CSVs
    merged/                 # source CSVs for merge scripts
    dwap_phase2/            # latency LUTs and detailed profiling CSVs
    summaries/              # normalized summary CSV outputs
  ramulator_patches/        # patch/source overlays copied into ramulator2
  ramulator2/               # Ramulator2 source/build tree, kept in place
```

`data/` and `results/` are ignored by Git. They are local experiment inputs and outputs, so cloning the repository does not include those files.

## Requirements

- Python 3
- `pandas`
- CMake, `g++`, and `clang++` for building Ramulator2
- SSH access to GitHub if you are pushing to the remote repository

The code has been used on Ubuntu-like Linux environments. Ramulator2 is included as a submodule/tree in this project layout.

## Setup

Clone the repository and initialize submodules if needed:

```bash
git clone git@github.com:wwj281/elsa_framework.git
cd elsa_framework
git submodule update --init --recursive
```

Prepare Ramulator2 with the Courier PIM patch set:

```bash
bash scripts/setup/courier_set_pim_ramulator.sh
cd ramulator2
mkdir -p build
cd build
cmake ..
make -j
cp ramulator2 ../ramulator2
cd ../..
```

For the older AttAcc/HBM3 patch set, use:

```bash
bash scripts/setup/set_pim_ramulator.sh
```

## Data And Results

Input expert/gate-weight data should be placed under:

```text
data/gate_weight/input{seq}_batch{batch}/{model_dir}/
```

For example:

```text
data/gate_weight/input1024_batch16/qwen_3_30b/
  per_layer_expert_stats_t0.23_r0.75.json
  expert_gate_sum_t0.23_r0.75.json
  expert_location_path.json
```

`courier_main.py` accepts short paths such as `input1024_batch16/qwen_3_30b/...` and resolves them under `data/gate_weight/`.

Generated outputs are written under `results/`:

- `results/raw_logs/courier/`: Courier internal hardware logs such as `DDR4_C2_MWEIGHT.log`
- `results/raw_logs/process/`: batch-run stdout/stderr logs
- `results/workload/`: workload-level `gpu_result.csv` and `pimoe_result.csv`
- `results/dwap_phase2/`: latency LUT JSON and detailed profiling CSVs
- `results/summaries/`: normalized aggregate CSVs

## Single Simulation

Run a single Courier simulation from the project root:

```bash
python courier_main.py \
  --tfs_file input1024_batch16/qwen_3_30b/per_layer_expert_stats_t0.23_r0.75.json \
  --gss_file input1024_batch16/qwen_3_30b/expert_gate_sum_t0.23_r0.75.json \
  --elp_file input1024_batch16/qwen_3_30b/expert_location_path.json \
  --model Qwen-3-30B \
  --schedule_strategy FUSION \
  --mapping_strategy WEIGHT \
  --batch 16 \
  --lin 1024 \
  --output_file results/workload/input1024_batch16/qwen_3_30b/fusion_result.csv
```

Useful options:

```bash
python courier_main.py --help
```

Common model names:

- `DeepSeek-16B`
- `Qwen-3-30B`
- `GPT-OSS-120B`
- `Mixtral-8x7B`

Common schedule strategies:

- `FUSION`
- `PIMOE`
- `NOFUSION`
- `FIDDLER`
- `KLOTSKI`

Common mapping strategies:

- `WEIGHT`
- `H2`
- `NAIVE`

For GPT-OSS-120B LPDDR5/H100-style runs, include:

```bash
--pim lpddr5 --gpu H100 --num_channel 16
```

## Batch Runs

Command lists live in:

```text
configs/evaluations/
```

Available lists:

- `evaluation.txt`: DWAP latency-surface profiling commands
- `evaluation_baseline.txt`: baseline Courier commands with log suffixing
- `evaluation_gpu.txt`: GPU result commands
- `evaluation_pimoe.txt`: PIMOE result commands

Run DWAP profiling commands sequentially:

```bash
bash scripts/run/run_eval.sh
```

Run GPU/PIMOE commands and save named process logs:

```bash
bash scripts/run/run_eval_pimoe_gpu.sh
```

Run baseline commands and rename Courier internal logs with workload suffixes:

```bash
python scripts/run/run_eval_baseline_with_suffix.py
```

This runner reads `configs/evaluations/evaluation_baseline.txt` by default and renames logs under `results/raw_logs/courier/`.

## DWAP Phase 2 Latency Profiling

List available data directories for a model:

```bash
python scripts/profile/profile_latency_surface.py \
  --model Qwen-3-30B \
  --list_data_dirs
```

Run a quick smoke test:

```bash
python scripts/profile/profile_latency_surface.py \
  --model Qwen-3-30B \
  --feasible_region_file configs/feasible_region/qwen_3_30b.json \
  --quick_test
```

Run the default DWAP Phase 2 profiling grid:

```bash
python scripts/profile/profile_latency_surface.py \
  --model Qwen-3-30B \
  --mapping_strategy H2 \
  --feasible_region_file configs/feasible_region/qwen_3_30b.json \
  --output_lut results/dwap_phase2/qwen_3_30b/latency_lut_all.json \
  --output_csv results/dwap_phase2/qwen_3_30b/latency_profiling_detailed_all.csv
```

The profiler writes:

- LUT JSON: `latency_lut*.json`
- detailed rows: `latency_profiling_detailed*.csv`

## Aggregation

Aggregation scripts are grouped under `scripts/aggregate/` and default to the new `results/` layout.

Summarize Courier internal hardware logs:

```bash
python scripts/aggregate/aggregate_root_log_summary.py
```

Summarize GPU/PIMOE process logs:

```bash
python scripts/aggregate/aggregate_log_gpu_pimoe.py
```

Extract best average latency configs from profiling process logs:

```bash
python scripts/aggregate/aggregate_log_shadow.py
```

Collect best latency rows from DWAP Phase 2 CSVs:

```bash
python scripts/aggregate/aggregate_best_latency_rows.py
```

Collect first data row from workload result CSVs:

```bash
python scripts/aggregate/aggregate_gpu_pimoe_first_row.py
```

Merge normalized result summaries:

```bash
python scripts/aggregate/merge_result_merge_summaries.py
```

All summary outputs are written to:

```text
results/summaries/
```

## Legacy Baseline

The original baseline code is preserved under:

```text
legacy/
```

Use it only when comparing with the old simulator path. The actively maintained path is:

```text
courier/
courier_main.py
scripts/
```

## Verification

Basic checks:

```bash
python -m compileall courier courier_main.py scripts legacy
python courier_main.py --help
python scripts/profile/profile_latency_surface.py --help
bash -n scripts/run/run_eval.sh scripts/run/run_eval_pimoe_gpu.sh
bash -n scripts/setup/set_pim_ramulator.sh scripts/setup/courier_set_pim_ramulator.sh
```

Data discovery check:

```bash
python scripts/profile/profile_latency_surface.py --model Qwen-3-30B --list_data_dirs
```

## Notes

- `data/` and `results/` are intentionally ignored by Git.
- `ramulator2/` remains in the project root and was not moved during the structure cleanup.
- `docs/STRUCTURE_MIGRATION.md` contains the detailed migration map from the old project layout.
