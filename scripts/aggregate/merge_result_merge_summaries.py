import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODEL_NAME_MAP = {
    "deepseek_16b": "DeepSeek-16B",
    "gpt_oss_120b": "GPT-OSS-120B",
    "qwen_3_30b": "Qwen-3-30B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge result_merge breakdown files and all_result files into two "
            "normalized summary CSV files."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "results" / "merged"),
        help="Directory containing the four input CSV files.",
    )
    parser.add_argument(
        "--breakdown-output",
        default=str(PROJECT_ROOT / "results" / "summaries" / "merged_breakdown_summary.csv"),
        help="Output path for the merged breakdown summary CSV.",
    )
    parser.add_argument(
        "--result-output",
        default=str(PROJECT_ROOT / "results" / "summaries" / "merged_result_summary.csv"),
        help="Output path for the merged result summary CSV.",
    )
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader)


def write_csv(output_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_breakdown_rows(input_dir: Path) -> list[dict[str, str]]:
    shadow_rows = read_csv_rows(input_dir / "all_result_shadow_breakdown.csv")
    gpu_pimoe_rows = read_csv_rows(input_dir / "all_result_gpu_pimoe_breakdown.csv")

    merged_rows: list[dict[str, str]] = []

    for row in shadow_rows:
        merged_rows.append(
            {
                "model_name": row["model_name"],
                "mode": "shadow",
                "B": row["B"],
                "S": row["S"],
                "gpu_total_time_s": row["gpu_total_time_s"],
                "gpu_c_time_s": row["gpu_c_time_s"],
                "gpu_m_time_s": row["gpu_m_time_s"],
                "gpu_l_time_s": row["gpu_l_time_s"],
                "acc_total_time_s": row["acc_total_time_s"],
                "acc_c_time_s": row["acc_c_time_s"],
                "acc_m_time_s": row["acc_m_time_s"],
                "acc_l_time_s": row["acc_l_time_s"],
                "gpu_expert_count": row["gpu_expert_count"],
            }
        )

    for row in gpu_pimoe_rows:
        merged_rows.append(
            {
                "model_name": row["model_name"],
                "mode": row["mode"],
                "B": row["B"],
                "S": row["S"],
                "gpu_total_time_s": row["gpu_total_time_s"],
                "gpu_c_time_s": row["gpu_c_time_s"],
                "gpu_m_time_s": row["gpu_m_time_s"],
                "gpu_l_time_s": row["gpu_l_time_s"],
                "acc_total_time_s": row["acc_total_time_s"],
                "acc_c_time_s": row["acc_c_time_s"],
                "acc_m_time_s": row["acc_m_time_s"],
                "acc_l_time_s": row["acc_l_time_s"],
                "gpu_expert_count": row["gpu_expert_count"],
            }
        )

    merged_rows.sort(key=lambda row: (row["model_name"], row["mode"], int(row["B"]), int(row["S"])))
    return merged_rows


def normalize_result_rows(input_dir: Path) -> list[dict[str, str]]:
    shadow_rows = read_csv_rows(input_dir / "all_result_shadow.csv")
    gpu_pimoe_rows = read_csv_rows(input_dir / "all_result_gpu_pimoe.csv")

    merged_rows: list[dict[str, str]] = []

    for row in shadow_rows:
        merged_rows.append(
            {
                "model_name": MODEL_NAME_MAP.get(row["model_name"], row["model_name"]),
                "result_type": "shadow",
                "Lin": row["seq_length"],
                "Lout": "2",
                "bs": row["batch_size"],
                "s_time": row["s_time_ms"],
                "s_matmul": "",
                "s_fc": row["s_fc_time_ms"],
                "s_comm": row["s_comm_time_ms"],
                "s_softmax": "",
                "g_time": row["g_time_ms"],
                "g_matmul": "",
                "g_fc": row["g_fc_time_ms"],
                "g_comm": row["g_comm_time_ms"],
                "g_softmax": "",
                "g_ff_time": row["g_ff_time_ms"],
            }
        )

    for row in gpu_pimoe_rows:
        merged_rows.append(
            {
                "model_name": row.get("model", row["model_name"]),
                "result_type": row["result_type"],
                "Lin": row["Lin"],
                "Lout": row["Lout"],
                "bs": row["bs"],
                "s_time": row["s_time"],
                "s_matmul": row["s_matmul"],
                "s_fc": row["s_fc"],
                "s_comm": row["s_comm"],
                "s_softmax": row["s_softmax"],
                "g_time": row["g_time (ms)"],
                "g_matmul": row["g_matmul"],
                "g_fc": row["g_fc"],
                "g_comm": row["g_comm"],
                "g_softmax": row["g_softmax"],
                "g_ff_time": row["g_ff_time"],
            }
        )

    merged_rows.sort(
        key=lambda row: (
            row["model_name"],
            row["result_type"],
            int(row["bs"]) if row["bs"] else -1,
            int(row["Lin"]) if row["Lin"] else -1,
        )
    )
    return merged_rows


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    breakdown_output = Path(args.breakdown_output).resolve()
    result_output = Path(args.result_output).resolve()

    breakdown_rows = normalize_breakdown_rows(input_dir)
    result_rows = normalize_result_rows(input_dir)

    write_csv(
        breakdown_output,
        [
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
        ],
        breakdown_rows,
    )

    write_csv(
        result_output,
        [
            "model_name",
            "result_type",
            "Lin",
            "Lout",
            "bs",
            "s_time",
            "s_matmul",
            "s_fc",
            "s_comm",
            "s_softmax",
            "g_time",
            "g_matmul",
            "g_fc",
            "g_comm",
            "g_softmax",
            "g_ff_time",
        ],
        result_rows,
    )

    print(f"Wrote {len(breakdown_rows)} rows to {breakdown_output}")
    print(f"Wrote {len(result_rows)} rows to {result_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
