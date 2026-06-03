import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


TARGET_FILENAMES = ("gpu_result.csv", "pimoe_result.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the first data row from every gpu_result.csv and "
            "pimoe_result.csv under a result directory."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(PROJECT_ROOT / "results" / "workload"),
        help="Root directory containing workload/model result folders.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "summaries" / "first_row_summary.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def find_result_csv_files(input_dir: Path) -> list[Path]:
    csv_files: list[Path] = []

    for filename in TARGET_FILENAMES:
        csv_files.extend(path for path in input_dir.rglob(filename) if path.is_file())

    return sorted(csv_files)


def derive_metadata(input_dir: Path, csv_path: Path) -> dict[str, str]:
    relative_path = csv_path.relative_to(input_dir)
    parts = relative_path.parts

    workload = parts[0] if len(parts) >= 1 else "unknown"
    model_name = parts[1] if len(parts) >= 2 else csv_path.parent.name
    result_type = csv_path.stem.replace("_result", "")

    return {
        "workload": workload,
        "model_name": model_name,
        "result_type": result_type,
        "source_csv": relative_path.as_posix(),
    }


def read_first_data_row(input_dir: Path, csv_path: Path) -> tuple[dict[str, str], list[str]]:
    metadata = derive_metadata(input_dir, csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"CSV has no data rows: {csv_path}")

        return {**metadata, **first_row}, list(reader.fieldnames)


def write_summary(output_path: Path, rows: list[dict[str, str]], source_fields: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["workload", "model_name", "result_type", "source_csv", *source_fields]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()

    csv_files = find_result_csv_files(input_dir)
    if not csv_files:
        raise FileNotFoundError(f"No target result CSV files found under {input_dir}")

    all_rows: list[dict[str, str]] = []
    source_fields: list[str] | None = None

    for csv_path in csv_files:
        row, fieldnames = read_first_data_row(input_dir, csv_path)
        if source_fields is None:
            source_fields = fieldnames
        elif source_fields != fieldnames:
            raise ValueError(
                "CSV headers are inconsistent across input files: "
                f"{csv_path}"
            )
        all_rows.append(row)

    all_rows.sort(
        key=lambda row: (
            row["workload"],
            row["model_name"],
            row["result_type"],
        )
    )
    write_summary(output_path, all_rows, source_fields or [])

    print(f"Scanned {len(csv_files)} file(s).")
    print(f"Wrote {len(all_rows)} first-row entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
