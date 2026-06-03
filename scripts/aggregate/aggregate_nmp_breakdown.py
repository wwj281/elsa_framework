#!/usr/bin/env python3
import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Counter as CounterType, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURIER_LOG_DIR = PROJECT_ROOT / "results" / "raw_logs" / "courier"
PROCESS_LOG_DIR = PROJECT_ROOT / "results" / "raw_logs" / "process"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"


LOG_PATTERN = re.compile(
    r"^(?P<memory_type>DDR4|LPDDR5)_C(?P<num_channel>\d+)_(?P<mapping_name>MH2|MWEIGHT)"
    r"_(?P<schedule_type>PIMOE|FUSION)_(?P<batch_size>\d+)_(?P<input_length>\d+)"
    r"_(?P<model>qw|ds|gpt)\.log$"
)
BLOCK_HEADER_PATTERN = re.compile(r"B=(?P<batch_size>\d+),\s*S=(?P<input_length>\d+)")
ACC_TOKEN_PATTERNS = [
    re.compile(r"^Expert\s+\d+\s+acc\s+tokens:\s+(\d+)$"),
    re.compile(r"^Channel\s+\d+\s+Expert\s+\d+\s+acc\s+tokens:\s+(\d+)$"),
]
MODEL_PROCESS_LOG_FILES = {
    "ds": "deepseek_16b.log",
    "qw": "qwen_3_30b.log",
    "gpt": "gpt_oss_120b.log",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 courier_demo 根目录日志并导出汇总 CSV。"
    )
    parser.add_argument(
        "--workdir",
        default=str(COURIER_LOG_DIR),
        help="courier 内部硬件日志目录，默认 results/raw_logs/courier",
    )
    parser.add_argument(
        "--output",
        default=str(SUMMARY_DIR / "root_log_summary.csv"),
        help="输出 CSV 路径，默认 results/summaries/root_log_summary.csv",
    )
    return parser.parse_args()


def list_target_logs(workdir: Path) -> List[Path]:
    matched_logs: List[Path] = []
    for path in sorted(workdir.iterdir()):
        if not path.is_file() or path.suffix != ".log":
            continue
        if path.name.endswith("_old.log"):
            continue
        if LOG_PATTERN.match(path.name):
            matched_logs.append(path)
    return matched_logs


def get_scale_factors(memory_type: str) -> Dict[str, int]:
    if memory_type == "DDR4":
        return {"mac": 4, "wrgb": 2, "af": 4}
    return {"mac": 4, "wrgb": 2, "af": 4}


def parse_process_log_counts(process_log_path: Path) -> CounterType[int]:
    token_counter: CounterType[int] = Counter()

    if not process_log_path.exists():
        return token_counter

    with process_log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            for pattern in ACC_TOKEN_PATTERNS:
                match = pattern.match(line)
                if match is not None:
                    token_counter[int(match.group(1))] += 1
                    break

    return token_counter


def parse_profile_log_blocks(
    process_log_path: Path,
    batch_size: int,
    input_length: int,
) -> List[CounterType[int]]:
    candidate_blocks: List[CounterType[int]] = []
    current_counter: Optional[CounterType[int]] = None
    in_target_block = False

    if not process_log_path.exists():
        return candidate_blocks

    with process_log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            header_match = BLOCK_HEADER_PATTERN.search(line)
            if header_match is not None:
                if in_target_block and current_counter is not None:
                    candidate_blocks.append(current_counter)
                in_target_block = (
                    int(header_match.group("batch_size")) == batch_size
                    and int(header_match.group("input_length")) == input_length
                )
                current_counter = Counter() if in_target_block else None
                continue

            if not in_target_block or current_counter is None:
                continue

            for pattern in ACC_TOKEN_PATTERNS:
                match = pattern.match(line)
                if match is not None:
                    current_counter[int(match.group(1))] += 1
                    break

    if in_target_block and current_counter is not None:
        candidate_blocks.append(current_counter)

    return candidate_blocks


def choose_best_expected_counts(
    observed_counts: CounterType[int],
    candidates: List[CounterType[int]],
) -> CounterType[int]:
    if not candidates:
        return Counter()

    observed_tokens = set(observed_counts.keys())

    def score(counter: CounterType[int]) -> tuple[int, int, int]:
        shared_tokens = observed_tokens & set(counter.keys())
        overlap = len(shared_tokens)
        matched_occurrences = sum(min(observed_counts[token], counter[token]) for token in shared_tokens)
        total_occurrences = sum(counter[token] for token in shared_tokens)
        return overlap, matched_occurrences, total_occurrences

    return max(candidates, key=score)


def get_expected_counts(workdir: Path, metadata: Dict[str, str], observed_counts: CounterType[int]) -> CounterType[int]:
    logs_dir = PROCESS_LOG_DIR

    if metadata["schedule_type"] == "PIMOE":
        process_log_path = logs_dir / (
            f"pimoe_b{metadata['batch_size']}i{metadata['input_length']}_{metadata['model']}.log"
        )
        return parse_process_log_counts(process_log_path)

    process_log_name = MODEL_PROCESS_LOG_FILES.get(metadata["model"])
    if process_log_name is None:
        return Counter()

    process_log_path = logs_dir / process_log_name
    candidates = parse_profile_log_blocks(
        process_log_path,
        int(metadata["batch_size"]),
        int(metadata["input_length"]),
    )
    return choose_best_expected_counts(observed_counts, candidates)


def get_row_multiplier(
    token_count: int,
    observed_counts: CounterType[int],
    expected_counts: CounterType[int],
) -> float:
    observed = observed_counts.get(token_count, 0)
    expected = expected_counts.get(token_count, 0)

    if observed == 0 or expected == 0 or expected <= observed:
        return 1.0
    if expected % observed == 0:
        return float(expected // observed)
    return expected / observed


def summarize_log(log_path: Path, workdir: Path) -> Dict[str, object]:
    match = LOG_PATTERN.match(log_path.name)
    if match is None:
        raise ValueError(f"Unexpected log name: {log_path.name}")

    metadata = match.groupdict()
    cycle_sum = 0
    mac_sum = 0
    mvgb_sum = 0
    wrgb_sum = 0
    acc_sum = 0
    af_sum = 0
    ewmul_sum = 0

    observed_counts: CounterType[int] = Counter()

    factors = get_scale_factors(metadata["memory_type"])
    rows = []
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_row in reader:
            rows.append(csv_row)
            observed_counts[int(csv_row["t"])] += 1

    expected_counts = get_expected_counts(workdir, metadata, observed_counts)
    if expected_counts:
        overlap = len(set(observed_counts.keys()) & set(expected_counts.keys()))
        if overlap == 0:
            print(f"[Warning] No token overlap for {log_path.name}; keeping unmatched rows as-is")

    for csv_row in rows:
        token_count = int(csv_row["t"])
        multiplier = get_row_multiplier(token_count, observed_counts, expected_counts)
        cycle_sum += int(csv_row["cycle"]) * multiplier
        mac_sum += int(csv_row["mac"]) * factors["mac"] * multiplier
        mvgb_sum += int(csv_row["mvgb"]) * multiplier
        wrgb_sum += int(csv_row["wrgb"]) * multiplier
        acc_sum += int(csv_row["acc"]) * multiplier
        af_sum += int(csv_row["af"]) * multiplier
        ewmul_sum += int(csv_row["ewmul"]) * multiplier

    if metadata['schedule_type'] == "FUSION":
        activation_time = (
        cycle_sum
        - mac_sum
        - wrgb_sum * factors["wrgb"]
        - af_sum * factors["af"]
        )
    else:
        activation_time = (
            cycle_sum
            - mac_sum
            - wrgb_sum * factors["wrgb"]
            - af_sum * factors["af"]
            - mvgb_sum * factors["wrgb"]
            - acc_sum * factors["af"]
            - ewmul_sum * factors["af"]
        )

    return {
        "file_name": log_path.name,
        "mapping_type": metadata["schedule_type"],
        "model": metadata["model"],
        "batch_size": int(metadata["batch_size"]),
        "input_length": int(metadata["input_length"]),
        "execution_time": cycle_sum,
        "compute_time": mac_sum,
        "activation_time": activation_time,
    }


def write_summary(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "file_name",
        "mapping_type",
        "model",
        "batch_size",
        "input_length",
        "execution_time",
        "compute_time",
        "activation_time",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    output_arg = Path(args.output)
    output_path = output_arg if output_arg.is_absolute() else PROJECT_ROOT / output_arg

    logs = list_target_logs(workdir)
    if not logs:
        raise SystemExit("No matching logs found in the root directory.")

    rows = [summarize_log(path, workdir) for path in logs]
    write_summary(rows, output_path)

    print(f"[Summary] logs counted: {len(rows)}")
    print(f"[Summary] output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
