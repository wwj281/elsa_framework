import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_LOG_FILES = (
    "deepseek_16b_h2.log",
    "gpt_oss_120b_h2.log",
    "qwen_3_30b_h2.log",
)

MODEL_RE = re.compile(r"^Model:\s+(?P<model>.+)$")
BUCKET_RE = re.compile(r"^Profiling Bucket: Bucket\(B=\((?P<b_start>\d+),\s*(?P<b_end>\d+)\), S=\((?P<s_start>\d+),\s*(?P<s_end>\d+)\)\)")
BATCH_REP_RE = re.compile(r"^\s*Batch representatives:\s*\[(?P<batch>\d+)\]")
SEQ_REP_RE = re.compile(r"^\s*Seq representatives:\s*\[(?P<seq>\d+)\]")
CONFIG_RE = re.compile(r"^\s*Config\s+\d+/\d+:\s+T=(?P<t>[0-9.]+),\s+M=(?P<m>[0-9.]+)")
WORKLOAD_RE = re.compile(r"^\s*B=(?P<batch>\d+),\s+S=(?P<seq>\d+)\s+\(dir:\s+(?P<data_dir>[^)]+)\)\.\.\.")
GPU_EXPERT_RE = re.compile(r"^gpu_expert_ids\s+\((?P<count>\d+)\):")
TOTAL_TIME_RE = re.compile(r"^(?P<kind>gpu|acc)_total_time:\s+(?P<value>[0-9.eE+-]+)\s+s$")
BREAKDOWN_RE = re.compile(r"^\s*(?P<name>[cml]_time):\s+(?P<value>[0-9.eE+-]+)\s+s$")
LATENCY_RE = re.compile(r"^\s*Latency=(?P<latency_ms>[0-9.]+)ms$")
AVG_LATENCY_RE = re.compile(r"^\s*-> Avg latency for \(T=(?P<t>[0-9.]+), M=(?P<m>[0-9.]+)\):\s+(?P<latency_ms>[0-9.]+)ms$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse model log files and collect the best T/M configuration for "
            "each batch-size and sequence-length combination."
        )
    )
    parser.add_argument(
        "--logs-dir",
        default=str(PROJECT_ROOT / "results" / "raw_logs" / "process"),
        help="Directory containing the model log files.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "summaries" / "best_avg_latency_configs.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--log-files",
        nargs="*",
        default=list(DEFAULT_LOG_FILES),
        help="Specific log filenames to parse.",
    )
    return parser.parse_args()


def build_log_paths(logs_dir: Path, log_files: List[str]) -> List[Path]:
    paths = [logs_dir / name for name in log_files]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing log files: {missing_list}")
    return paths


def new_config(model_name: str, source_log: Path, batch_size: Optional[int], seq_length: Optional[int]) -> Dict[str, object]:
    return {
        "model_name": model_name,
        "source_log": source_log.name,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "T": None,
        "M": None,
        "data_dir": None,
        "gpu_expert_count": None,
        "gpu_total_time_s": None,
        "gpu_c_time_s": None,
        "gpu_m_time_s": None,
        "gpu_l_time_s": None,
        "acc_total_time_s": None,
        "acc_c_time_s": None,
        "acc_m_time_s": None,
        "acc_l_time_s": None,
        "latency_ms": None,
        "avg_latency_ms": None,
    }


def finalize_config(config: Dict[str, object], best_by_workload: Dict[Tuple[str, int, int], Dict[str, object]]) -> None:
    batch_size = config.get("batch_size")
    seq_length = config.get("seq_length")
    latency_ms = config.get("latency_ms")
    if batch_size is None or seq_length is None or latency_ms is None:
        return

    key = (str(config["model_name"]), int(batch_size), int(seq_length))
    previous = best_by_workload.get(key)
    if previous is None or float(latency_ms) < float(previous["latency_ms"]):
        best_by_workload[key] = dict(config)


def finalize_pending_workloads(
    pending_workloads: List[Dict[str, object]],
    best_by_workload: Dict[Tuple[str, int, int], Dict[str, object]],
    avg_latency_ms: Optional[float] = None,
) -> None:
    for workload in pending_workloads:
        if avg_latency_ms is not None:
            workload["avg_latency_ms"] = avg_latency_ms
        finalize_config(workload, best_by_workload)
    pending_workloads.clear()


def parse_log_file(log_path: Path) -> List[Dict[str, object]]:
    model_name: Optional[str] = None
    batch_rep: Optional[int] = None
    seq_rep: Optional[int] = None
    current_config_template: Optional[Dict[str, object]] = None
    current_workload: Optional[Dict[str, object]] = None
    pending_workloads: List[Dict[str, object]] = []
    breakdown_target: Optional[str] = None
    breakdown_index = 0
    best_by_workload: Dict[Tuple[str, int, int], Dict[str, object]] = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            model_match = MODEL_RE.match(line)
            if model_match and model_name is None:
                model_name = model_match.group("model")
                continue

            if BUCKET_RE.match(line):
                if current_workload is not None:
                    pending_workloads.append(current_workload)
                    current_workload = None
                finalize_pending_workloads(pending_workloads, best_by_workload)
                current_config_template = None
                batch_rep = None
                seq_rep = None
                continue

            batch_match = BATCH_REP_RE.match(line)
            if batch_match:
                batch_rep = int(batch_match.group("batch"))
                continue

            seq_match = SEQ_REP_RE.match(line)
            if seq_match:
                seq_rep = int(seq_match.group("seq"))
                continue

            config_match = CONFIG_RE.match(line)
            if config_match:
                if model_name is None:
                    raise ValueError(f"Model name not found before config in {log_path}")
                if current_workload is not None:
                    pending_workloads.append(current_workload)
                    current_workload = None
                finalize_pending_workloads(pending_workloads, best_by_workload)

                current_config_template = new_config(model_name, log_path, batch_rep, seq_rep)
                current_config_template["T"] = float(config_match.group("t"))
                current_config_template["M"] = float(config_match.group("m"))
                breakdown_target = None
                breakdown_index = 0
                continue

            if current_config_template is None:
                continue

            workload_match = WORKLOAD_RE.match(line)
            if workload_match:
                if current_workload is not None:
                    pending_workloads.append(current_workload)

                current_workload = dict(current_config_template)
                current_workload["batch_size"] = int(workload_match.group("batch"))
                current_workload["seq_length"] = int(workload_match.group("seq"))
                current_workload["data_dir"] = workload_match.group("data_dir")
                breakdown_target = None
                breakdown_index = 0
                continue

            if current_workload is None:
                continue

            gpu_expert_match = GPU_EXPERT_RE.match(line)
            if gpu_expert_match and current_workload["gpu_expert_count"] is None:
                current_workload["gpu_expert_count"] = int(gpu_expert_match.group("count"))
                continue

            total_time_match = TOTAL_TIME_RE.match(line)
            if total_time_match:
                kind = total_time_match.group("kind")
                value = float(total_time_match.group("value"))

                if kind == "gpu" and current_workload["gpu_total_time_s"] is None:
                    current_workload["gpu_total_time_s"] = value
                    breakdown_target = "gpu"
                    breakdown_index = 0
                    continue

                if kind == "acc" and current_workload["acc_total_time_s"] is None:
                    current_workload["acc_total_time_s"] = value
                    breakdown_target = "acc"
                    breakdown_index = 0
                    continue

            breakdown_match = BREAKDOWN_RE.match(line)
            if breakdown_match and breakdown_target is not None and breakdown_index < 3:
                name = breakdown_match.group("name")
                value = float(breakdown_match.group("value"))
                current_workload[f"{breakdown_target}_{name}_s"] = value
                breakdown_index += 1
                if breakdown_index == 3:
                    breakdown_target = None
                continue

            latency_match = LATENCY_RE.match(line)
            if latency_match:
                current_workload["latency_ms"] = float(latency_match.group("latency_ms"))
                continue

            avg_latency_match = AVG_LATENCY_RE.match(line)
            if avg_latency_match:
                if current_workload is not None:
                    pending_workloads.append(current_workload)
                    current_workload = None
                finalize_pending_workloads(
                    pending_workloads,
                    best_by_workload,
                    avg_latency_ms=float(avg_latency_match.group("latency_ms")),
                )
                breakdown_target = None
                breakdown_index = 0

    if current_workload is not None:
        pending_workloads.append(current_workload)

    finalize_pending_workloads(pending_workloads, best_by_workload)

    return [best_by_workload[key] for key in sorted(best_by_workload)]


def write_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "model_name",
        "B",
        "S",
        "T",
        "M",
        "latency_ms",
        "avg_latency_ms",
        "gpu_total_time_s",
        "gpu_c_time_s",
        "gpu_m_time_s",
        "gpu_l_time_s",
        "acc_total_time_s",
        "acc_c_time_s",
        "acc_m_time_s",
        "acc_l_time_s",
        "gpu_expert_count",
        "data_dir",
        "source_log",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_name": row["model_name"],
                    "B": row["batch_size"],
                    "S": row["seq_length"],
                    "T": row["T"],
                    "M": row["M"],
                    "latency_ms": row["latency_ms"],
                    "avg_latency_ms": row["avg_latency_ms"],
                    "gpu_total_time_s": row["gpu_total_time_s"],
                    "gpu_c_time_s": row["gpu_c_time_s"],
                    "gpu_m_time_s": row["gpu_m_time_s"],
                    "gpu_l_time_s": row["gpu_l_time_s"],
                    "acc_total_time_s": row["acc_total_time_s"],
                    "acc_c_time_s": row["acc_c_time_s"],
                    "acc_m_time_s": row["acc_m_time_s"],
                    "acc_l_time_s": row["acc_l_time_s"],
                    "gpu_expert_count": row["gpu_expert_count"],
                    "data_dir": row["data_dir"],
                    "source_log": row["source_log"],
                }
            )


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir).resolve()
    output_path = Path(args.output).resolve()
    log_paths = build_log_paths(logs_dir, args.log_files)

    all_rows: List[Dict[str, object]] = []
    for log_path in log_paths:
        all_rows.extend(parse_log_file(log_path))

    all_rows.sort(key=lambda row: (str(row["model_name"]), int(row["batch_size"]), int(row["seq_length"])))
    write_csv(output_path, all_rows)

    print(f"Parsed {len(log_paths)} log file(s).")
    print(f"Wrote {len(all_rows)} best configuration rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
