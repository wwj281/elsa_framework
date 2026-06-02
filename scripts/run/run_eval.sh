#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CMD_FILE="${PROJECT_ROOT}/configs/evaluations/evaluation.txt"     # 存放命令的文件
LOG_DIR="${PROJECT_ROOT}/results/raw_logs/process"                # 日志目录

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}" || exit 1

i=1
while IFS= read -r cmd || [[ -n "$cmd" ]]; do
    # 跳过空行
    [[ -z "$cmd" ]] && continue

    log_file="${LOG_DIR}/job_${i}.log"

    echo "[$(date)] Running command ${i}"
    echo "Command: ${cmd}"
    echo "Log: ${log_file}"
    echo "----------------------------------------"

    # 顺序执行：前一个不结束，后一个不会开始
    eval "$cmd" > "${log_file}" 2>&1

    echo "[$(date)] Finished command ${i}"
    echo

    i=$((i + 1))
done < "${CMD_FILE}"
