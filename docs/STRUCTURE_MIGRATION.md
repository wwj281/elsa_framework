# Project Structure Migration

This document records the cleanup applied to the test copy at:

```text
/home/wwj/code/inhouse/elsa_framework
```

`ramulator2/` was intentionally kept in its original location.

## New Top-Level Layout

```text
elsa_framework/
  courier/                  # current simulator package
  courier_main.py           # current CLI entrypoint
  legacy/                   # old src/ + old main.py kept for reference
  scripts/
    run/                    # batch runners
    profile/                # latency surface profiler
    aggregate/              # result/log aggregation scripts
    setup/                  # ramulator patch/setup scripts
  configs/
    evaluations/            # command lists
    feasible_region/        # phase-1 feasible-region JSON files
  data/
    gate_weight/            # expert stats / gate-weight inputs
  results/
    raw_logs/
      courier/              # courier internal hardware CSV logs
      process/              # stdout/stderr process logs
    workload/               # per-workload gpu_result / pimoe_result CSVs
    merged/                 # source files used by merge scripts
    dwap_phase2/            # latency LUT and profiling CSV outputs
    summaries/              # normalized summary CSVs
  ramulator_patches/        # patch/source overlays for ramulator2
  ramulator2/               # unchanged
```

## Main File Moves

| Old location | New location |
|---|---|
| `courier_src/*.py` | `courier/*.py` |
| `src/` | `legacy/src/` |
| `main.py` | `legacy/main.py` |
| `run_eval.sh` | `scripts/run/run_eval.sh` |
| `run_eval_pimoe_gpu.sh` | `scripts/run/run_eval_pimoe_gpu.sh` |
| `run_eval_baseline_with_suffix.py` | `scripts/run/run_eval_baseline_with_suffix.py` |
| `profile_latency_surface.py` | `scripts/profile/profile_latency_surface.py` |
| `aggregate_root_log_summary.py` | `scripts/aggregate/aggregate_root_log_summary.py` |
| `logs/aggregate_log_gpu_pimoe.py` | `scripts/aggregate/aggregate_log_gpu_pimoe.py` |
| `logs/aggregate_log_shadow.py` | `scripts/aggregate/aggregate_log_shadow.py` |
| `result/aggregate_gpu_pimoe_first_row.py` | `scripts/aggregate/aggregate_gpu_pimoe_first_row.py` |
| `result_merge/merge_result_merge_summaries.py` | `scripts/aggregate/merge_result_merge_summaries.py` |
| `dwap_phase2_results/aggregate_best_latency_rows.py` | `scripts/aggregate/aggregate_best_latency_rows.py` |
| `evaluation*.txt` | `configs/evaluations/` |
| `feasible_region/*.json` | `configs/feasible_region/` |
| `gate_weight_data/` | `data/gate_weight/` |
| root `DDR4_*.log`, `LPDDR5_*.log`, `courier_output.log` | `results/raw_logs/courier/` |
| `logs/*.log`, `logs/old/` | `results/raw_logs/process/` |
| `result/` contents | `results/workload/` |
| `result_merge/` contents | `results/merged/` |
| `dwap_phase2_results/` contents | `results/dwap_phase2/` |
| `root_log_summary.csv`, `output.csv` | `results/summaries/` |
| `pim_ramulator_src/` | `ramulator_patches/pim_ramulator_src/` |
| `courier_pim_ramulator_src/` | `ramulator_patches/courier_pim_ramulator_src/` |

## Code Path Updates

- Imports were updated from `courier_src.*` to `courier.*`.
- Legacy imports were updated from `src.*` to `legacy.src.*`.
- `courier_main.py` now resolves input files under `data/gate_weight/`.
- `courier_main.py` maps old command-list outputs like `result/...` to `results/workload/...`.
- Courier internal hardware logs are now written under `results/raw_logs/courier/`.
- Batch runner logs are now written under `results/raw_logs/process/`.
- Aggregation scripts default to reading from `results/` and writing normalized CSVs into `results/summaries/`.
- Profiling accepts old-style paths for compatibility, but defaults now point to `configs/feasible_region/` and `results/dwap_phase2/`.
- Setup scripts now copy overlays from `ramulator_patches/` into the unchanged `ramulator2/` tree.

## Current Entrypoints

```bash
python courier_main.py --help
python scripts/profile/profile_latency_surface.py --help
python scripts/run/run_eval_baseline_with_suffix.py --help
bash scripts/run/run_eval.sh
bash scripts/run/run_eval_pimoe_gpu.sh
```

Aggregation examples:

```bash
python scripts/aggregate/aggregate_root_log_summary.py
python scripts/aggregate/aggregate_log_gpu_pimoe.py
python scripts/aggregate/aggregate_log_shadow.py
python scripts/aggregate/aggregate_best_latency_rows.py
python scripts/aggregate/aggregate_gpu_pimoe_first_row.py
python scripts/aggregate/merge_result_merge_summaries.py
```

## Verification Performed

The following checks passed after the migration:

```bash
python3 -m compileall courier courier_main.py scripts legacy
python3 courier_main.py --help
python3 scripts/profile/profile_latency_surface.py --help
bash -n scripts/run/run_eval.sh scripts/run/run_eval_pimoe_gpu.sh scripts/setup/set_pim_ramulator.sh scripts/setup/courier_set_pim_ramulator.sh
python3 scripts/profile/profile_latency_surface.py --model Qwen-3-30B --list_data_dirs
python3 scripts/aggregate/aggregate_root_log_summary.py
python3 scripts/aggregate/aggregate_log_gpu_pimoe.py
python3 scripts/aggregate/aggregate_log_shadow.py
python3 scripts/aggregate/aggregate_best_latency_rows.py
python3 scripts/aggregate/aggregate_gpu_pimoe_first_row.py
python3 scripts/aggregate/merge_result_merge_summaries.py
```

The full evaluation runners were not executed because they would launch the complete experiment batch.
