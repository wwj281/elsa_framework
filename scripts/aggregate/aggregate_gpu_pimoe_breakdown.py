import argparse
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


FILENAME_RE = re.compile(
    r"^(?P<mode>gpu|pimoe)_b(?P<batch>\d+)i(?P<seq>\d+)_(?P<model>ds|gpt|qw|q3)\.log$"
)
GPU_EXPERT_RE = re.compile(r"^gpu_expert_ids\s+\((?P<count>\d+)\):")
TOTAL_TIME_RE = re.compile(r"^(?P<kind>gpu|acc)_total_time:\s+(?P<value>[0-9.eE+-]+)\s+s$")
BREAKDOWN_RE = re.compile(r"^\s*(?P<name>[cml]_time):\s+(?P<value>[0-9.eE+-]+)\s+s$")

MODEL_NAME_MAP = {
    "ds": "DeepSeek-16B",
    "gpt": "GPT-OSS-120B",
    "qw": "Qwen-3-30B",
    "q3": "Qwen-3-30B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect summary rows from gpu_*.log and pimoe_*.log files under a "
            "logs directory."
        )
    )
    parser.add_argument(
        "--logs-dir",
        default=str(PROJECT_ROOT / "results" / "raw_logs" / "process"),
        help="Directory containing gpu_*.log and pimoe_*.log files.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "summaries" / "mode_log_summary.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def find_log_files(logs_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in logs_dir.iterdir()
        if path.is_file() and FILENAME_RE.match(path.name)
    )


def parse_metadata(log_path: Path) -> dict[str, object]:
    match = FILENAME_RE.match(log_path.name)
    if match is None:
        raise ValueError(f"Unsupported log filename: {log_path.name}")

    model_key = match.group("model")
    return {
        "mode": match.group("mode"),
        "B": int(match.group("batch")),
        "S": int(match.group("seq")),
        "model_name": MODEL_NAME_MAP[model_key],
        "source_log": log_path.name,
    }


def parse_log_summary(log_path: Path) -> dict[str, object]:
    row = {
        **parse_metadata(log_path),
        "gpu_expert_count": None,
        "gpu_total_time_s": None,
        "gpu_c_time_s": None,
        "gpu_m_time_s": None,
        "gpu_l_time_s": None,
        "acc_total_time_s": None,
        "acc_c_time_s": None,
        "acc_m_time_s": None,
        "acc_l_time_s": None,
    }

    breakdown_target: str | None = None
    breakdown_index = 0

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            gpu_expert_match = GPU_EXPERT_RE.match(line)
            if gpu_expert_match and row["gpu_expert_count"] is None:
                row["gpu_expert_count"] = int(gpu_expert_match.group("count"))
                continue

            total_time_match = TOTAL_TIME_RE.match(line)
            if total_time_match:
                kind = total_time_match.group("kind")
                value = float(total_time_match.group("value"))

                if kind == "gpu" and row["gpu_total_time_s"] is None:
                    row["gpu_total_time_s"] = value
                    breakdown_target = "gpu"
                    breakdown_index = 0
                    continue

                if kind == "acc" and row["acc_total_time_s"] is None:
                    row["acc_total_time_s"] = value
                    breakdown_target = "acc"
                    breakdown_index = 0
                    continue

            breakdown_match = BREAKDOWN_RE.match(line)
            if breakdown_match and breakdown_target is not None and breakdown_index < 3:
                row[f"{breakdown_target}_{breakdown_match.group('name')}_s"] = float(
                    breakdown_match.group("value")
                )
                breakdown_index += 1
                if breakdown_index == 3:
                    breakdown_target = None
                continue

    required_fields = [
        "gpu_expert_count",
        "gpu_total_time_s",
        "gpu_c_time_s",
        "gpu_m_time_s",
        "gpu_l_time_s",
        "acc_total_time_s",
        "acc_c_time_s",
        "acc_m_time_s",
        "acc_l_time_s",
    ]
    missing = [field for field in required_fields if row[field] is None]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required fields in {log_path.name}: {missing_text}")

    return row


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model_name",
        "mode",
        "B",
        "S",
        "gpu_total_time_s",
        "gpu_c_time_s",
        "gpu_m_time_s",
        "gpu_l_time_s",
        "acc_total_time_s",
        "acc_c_time_s",
        "acc_m_time_s",
        "acc_l_time_s",
        "gpu_expert_count",
        "source_log",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir).resolve()
    output_path = Path(args.output).resolve()

    log_files = find_log_files(logs_dir)
    if not log_files:
        raise FileNotFoundError(f"No gpu/pimoe log files found under {logs_dir}")

    rows = [parse_log_summary(log_path) for log_path in log_files]
    rows.sort(key=lambda row: (str(row["model_name"]), str(row["mode"]), int(row["B"]), int(row["S"])))
    write_csv(output_path, rows)

    print(f"Parsed {len(log_files)} log file(s).")
    print(f"Wrote {len(rows)} summary rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
