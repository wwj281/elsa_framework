# 从DWAP2产生的Log文件中收集并聚合运行结果产生结果汇总文件
import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


REQUIRED_COLUMNS = {"batch_size", "seq_length", "total_latency_ms"}
PREFERRED_FILENAMES = (
    "latency_profiling_detailed_all.csv",
    "latency_profiling_detailed.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the best row for each batch_size+seq_length workload from "
            "every latency_profiling_detailed.csv found under a result directory."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(PROJECT_ROOT / "results" / "dwap_phase2"),
        help="Root directory containing per-model result folders.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "summaries" / "best_latency_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--fallback-filename",
        default="latency_profiling_detailed.csv",
        help=(
            "Filename used for recursive fallback when a model directory does not "
            "contain a preferred top-level CSV."
        ),
    )
    return parser.parse_args()


def find_csv_files(input_dir: Path, fallback_filename: str) -> list[Path]:
    csv_files: list[Path] = []

    for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        selected_path = None
        for filename in PREFERRED_FILENAMES:
            candidate = model_dir / filename
            if candidate.is_file():
                selected_path = candidate
                break

        if selected_path is not None:
            csv_files.append(selected_path)
            continue

        fallback_paths = sorted(
            path for path in model_dir.rglob(fallback_filename) if path.is_file()
        )
        csv_files.extend(fallback_paths)

    return csv_files


def derive_metadata(input_dir: Path, csv_path: Path) -> dict[str, str]:
    relative_path = csv_path.relative_to(input_dir)
    parts = relative_path.parts
    model_name = parts[0] if parts else csv_path.parent.name
    parent_relative = relative_path.parent

    if len(parent_relative.parts) <= 1:
        variant = "root"
    else:
        variant = "/".join(parent_relative.parts[1:])

    return {
        "model_name": model_name,
        "variant": variant,
        "source_csv": relative_path.as_posix(),
    }


def collect_best_rows(input_dir: Path, csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    metadata = derive_metadata(input_dir, csv_path)
    best_by_workload: dict[tuple[int, int], tuple[float, dict[str, str]]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV missing required columns ({missing}): {csv_path}")

        for row in reader:
            batch_size = int(row["batch_size"])
            seq_length = int(row["seq_length"])
            total_latency_ms = float(row["total_latency_ms"])
            workload = (batch_size, seq_length)

            previous = best_by_workload.get(workload)
            if previous is None or total_latency_ms < previous[0]:
                merged_row = {**metadata, **row}
                best_by_workload[workload] = (total_latency_ms, merged_row)

        rows = [
            best_by_workload[key][1]
            for key in sorted(best_by_workload, key=lambda item: (item[0], item[1]))
        ]
        return rows, list(reader.fieldnames)


def write_summary(output_path: Path, rows: list[dict[str, str]], source_fields: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "variant", "source_csv", *source_fields]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()

    csv_files = find_csv_files(input_dir, args.fallback_filename)
    if not csv_files:
        raise FileNotFoundError(
            f"No matching latency profiling CSV files found under {input_dir}"
        )

    all_rows: list[dict[str, str]] = []
    source_fields: list[str] | None = None

    for csv_path in csv_files:
        rows, fieldnames = collect_best_rows(input_dir, csv_path)
        if source_fields is None:
            source_fields = fieldnames
        elif source_fields != fieldnames:
            raise ValueError(
                "CSV headers are inconsistent across input files: "
                f"{csv_path}"
            )
        all_rows.extend(rows)

    all_rows.sort(
        key=lambda row: (
            row["model_name"],
            row["variant"],
            int(row["batch_size"]),
            int(row["seq_length"]),
        )
    )
    write_summary(output_path, all_rows, source_fields or [])

    print(f"Scanned {len(csv_files)} file(s).")
    print(f"Wrote {len(all_rows)} best-row entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
