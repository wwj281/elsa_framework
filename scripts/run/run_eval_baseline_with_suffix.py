#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURIER_LOG_DIR = PROJECT_ROOT / "results" / "raw_logs" / "courier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="顺序执行命令文件，并为 courier 生成的记录日志追加后缀。"
    )
    parser.add_argument(
        "--cmd-file",
        default="configs/evaluations/evaluation_baseline.txt",
        help="命令文件路径，默认使用 configs/evaluations/evaluation_baseline.txt",
    )
    parser.add_argument(
        "--workdir",
        default=str(PROJECT_ROOT),
        help="执行命令时使用的工作目录，默认项目根目录",
    )
    parser.add_argument(
        "--log-dir",
        default=str(COURIER_LOG_DIR),
        help="courier 内部生成日志所在目录，默认 results/raw_logs/courier",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单条命令失败后继续执行后续命令",
    )
    return parser.parse_args()


def load_commands(cmd_file: Path) -> List[str]:
    commands = []
    for raw_line in cmd_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        commands.append(line)
    return commands


def extract_cli_value(tokens: List[str], option: str, default: Optional[str] = None) -> Optional[str]:
    for idx, token in enumerate(tokens):
        if token == option and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(f"{option}="):
            return token.split("=", 1)[1]
    return default


def build_base_log_name(tokens: List[str]) -> str:
    pim = extract_cli_value(tokens, "--pim", "ddr4")
    num_channel = extract_cli_value(tokens, "--num_channel", "2")
    mapping_strategy = extract_cli_value(tokens, "--mapping_strategy", "WEIGHT")
    return f"{pim.upper()}_C{num_channel}_M{mapping_strategy.upper()}.log"


def build_renamed_log_name(tokens: List[str]) -> str:
    base_name = build_base_log_name(tokens)
    stem, suffix = base_name.rsplit(".", 1)
    schedule_strategy = extract_cli_value(tokens, "--schedule_strategy", "FUSION")
    batch = extract_cli_value(tokens, "--batch", "unknown")
    lin = extract_cli_value(tokens, "--lin", "unknown")
    return f"{stem}_{schedule_strategy}_{batch}_{lin}.{suffix}"


def rename_generated_log(log_dir: Path, tokens: List[str]) -> Optional[Path]:
    source_path = log_dir / build_base_log_name(tokens)
    if not source_path.exists():
        return None

    target_path = log_dir / build_renamed_log_name(tokens)
    if target_path.exists():
        target_path.unlink()
    source_path.rename(target_path)
    return target_path


def run_command(command: str, workdir: Path) -> int:
    completed = subprocess.run(command, cwd=workdir, shell=True)
    return completed.returncode


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    log_dir = Path(args.log_dir).resolve()
    cmd_file = (workdir / args.cmd_file).resolve() if not Path(args.cmd_file).is_absolute() else Path(args.cmd_file)

    if not cmd_file.exists():
        print(f"命令文件不存在: {cmd_file}", file=sys.stderr)
        return 1

    commands = load_commands(cmd_file)
    if not commands:
        print(f"命令文件为空: {cmd_file}", file=sys.stderr)
        return 1

    print(f"[Runner] workdir = {workdir}")
    print(f"[Runner] log_dir = {log_dir}")
    print(f"[Runner] cmd_file = {cmd_file}")
    print(f"[Runner] total commands = {len(commands)}")

    for index, command in enumerate(commands, start=1):
        tokens = shlex.split(command)
        print(f"\n[Runner] ({index}/{len(commands)}) running:")
        print(command)

        return_code = run_command(command, workdir)
        renamed_log = rename_generated_log(log_dir, tokens)

        if renamed_log is not None:
            print(f"[Runner] renamed log -> {renamed_log.name}")
        else:
            print("[Runner] no generated log found to rename")

        if return_code != 0:
            print(f"[Runner] command failed with exit code {return_code}", file=sys.stderr)
            if not args.continue_on_error:
                return return_code

    print("\n[Runner] all commands finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
