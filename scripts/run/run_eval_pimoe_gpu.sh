#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CMD_FILE="${PROJECT_ROOT}/configs/evaluations/evaluation_gpu.txt"     # 存放命令的文件
LOG_DIR="${PROJECT_ROOT}/results/raw_logs/process"                # 日志目录

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}" || exit 1

extract_arg() {
    local cmd="$1"
    local arg_name="$2"
    local value

    value=$(awk -v key="$arg_name" '
        {
            for (i = 1; i <= NF; i++) {
                if ($i == key && i + 1 <= NF) {
                    print $(i + 1)
                    exit
                }
            }
        }
    ' <<< "$cmd")

    printf '%s' "$value"
}

get_hardware_tag() {
    local cmd="$1"
    local hardware

    hardware=$(extract_arg "$cmd" "--schedule_strategy")
    hardware=${hardware,,}

    if [[ -n "$hardware" ]]; then
        printf '%s' "$hardware"
        return
    fi

    if [[ "$cmd" == *"dgx "* ]]; then
        printf '%s' "gpu"
        return
    fi

    if [[ "$cmd" == *"--pim "* ]]; then
        printf '%s' "pimoe"
        return
    fi

    printf '%s' "job"
}

get_model_tag() {
    local model_name="$1"
    local normalized_model

    normalized_model=${model_name,,}

    case "$normalized_model" in
        deepseek*) printf '%s' "ds" ;;
        gpt*|gpt-oss*) printf '%s' "gpt" ;;
        qwen*) printf '%s' "qw" ;;
        *)
            normalized_model=${normalized_model//[^a-z0-9]/_}
            printf '%s' "$normalized_model"
            ;;
    esac
}

build_log_name() {
    local cmd="$1"
    local hardware
    local batch
    local input_len
    local model_name
    local model_tag

    hardware=$(get_hardware_tag "$cmd")
    batch=$(extract_arg "$cmd" "--batch")
    input_len=$(extract_arg "$cmd" "--lin")
    model_name=$(extract_arg "$cmd" "--model")
    model_tag=$(get_model_tag "$model_name")

    [[ -z "$batch" ]] && batch="x"
    [[ -z "$input_len" ]] && input_len="x"
    [[ -z "$model_tag" ]] && model_tag="model"

    printf '%s_b%si%s_%s.log' "$hardware" "$batch" "$input_len" "$model_tag"
}

i=1
while IFS= read -r cmd || [[ -n "$cmd" ]]; do
    # 跳过空行
    [[ -z "$cmd" ]] && continue

    log_file="${LOG_DIR}/$(build_log_name "$cmd")"

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
