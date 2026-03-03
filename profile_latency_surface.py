#!/usr/bin/env python3
"""
DWAP Phase 2: Hardware Latency Surface Mapping (硬件负载曲面测绘)

This script performs offline profiling for the second phase of DWAP:
1. Load the feasible region (R_safe) from Phase 1 accuracy profiling
2. Build workload grid (Batch Size x Seq Length buckets)
3. For each workload bucket (B, S):
   - Dynamically select data directory based on (batch, seq, model)
   - Traverse (T, P) combinations within R_safe
   - Call the actual simulation system to measure end-to-end latency
   - Record system latency max(T_GPU, T_NMP)
   - Find optimal configuration (T_opt, P_opt) that minimizes latency
4. Finalize the results into a static Look-Up Table (LUT) as JSON

Usage:
    # List available data directories for a model
    python scripts/profile_latency_surface.py --model Mixtral-8x7B --list_data_dirs

    # Quick test mode
    python scripts/profile_latency_surface.py --model Mixtral-8x7B --quick_test

    # Default profiling
    python scripts/profile_latency_surface.py --model Mixtral-8x7B

    # Full grid profiling
    python scripts/profile_latency_surface.py --model Mixtral-8x7B --full_grid

    # With Phase 1 feasible region file
    python scripts/profile_latency_surface.py \
        --model Mixtral-8x7B \
        --feasible_region_file results/feasible_region.json \
        --output_lut results/latency_lut.json
"""

import argparse
import csv
import json
import os
import sys
import time
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import courier system components (same as courier_main.py)
from courier_src.system import System
from courier_src.type import (
    DataType, GPUType, PIMType, InterfaceType,
    DeviceType, MappingStrategyType, ScheduleStrategyType
)
from courier_src.config import make_model_config, make_xpu_config, make_pim_config


# ==============================================================================
# Default Configuration
# ==============================================================================

# Workload buckets definition (as specified in DWAP)
DEFAULT_BATCH_BUCKETS = [
    (1, 4),      # Small batch
    (5, 16),     # Medium batch
    (17, 32),    # Large batch
]

DEFAULT_SEQ_LENGTH_BUCKETS = [
    (0, 512),     # Short sequence
    (513, 1024),  # Medium sequence
    (1025, 2048), # Long sequence
]

# Representative values for each bucket (used for profiling)
DEFAULT_BATCH_REPRESENTATIVES = {
    (1, 4): [1, 2, 4],
    (5, 16): [8, 12, 16],
    (17, 32): [20, 24, 32],
}

DEFAULT_SEQ_REPRESENTATIVES = {
    (0, 512): [128, 256, 512],
    (513, 1024): [640, 768, 1024],
    (1025, 2048): [1280, 1536, 2048],
}


# ==============================================================================
# Dynamic Data Directory Functions
# ==============================================================================

def get_model_dir_name(model_name: str) -> str:
    """
    将模型名称转换为数据目录中使用的名称格式。

    例如:
        "Mixtral-8x7B" -> "mixtral_8x7b"
        "Qwen-2.7B" -> "qwen_1.5_2.7b"
        "DeepSeek-16B" -> "deepseek_16b"
        "Qwen-3-30B" -> "qwen_3_30b"
    """
    model_dir_mapping = {
        "Mixtral-8x7B": "mixtral_8x7b",
        "Qwen-2.7B": "qwen_1.5_2.7b",
        "DeepSeek-16B": "deepseek_16b",
        "Qwen-3-30B": "qwen_3_30b",
    }

    if model_name in model_dir_mapping:
        return model_dir_mapping[model_name]

    # 默认转换：转小写，替换 - 为 _
    return model_name.lower().replace("-", "_")


def list_available_data_dirs(model_name: str) -> List[Dict]:
    """
    列出指定模型所有可用的数据目录。

    Returns:
        List of dicts with keys: dir_name, seq, batch, path
    """
    model_dir = get_model_dir_name(model_name)
    gate_weight_base = "gate_weight_data"
    available = []

    if not os.path.exists(gate_weight_base):
        return available

    for dir_name in os.listdir(gate_weight_base):
        dir_path = os.path.join(gate_weight_base, dir_name)
        if not os.path.isdir(dir_path):
            continue

        if dir_name.startswith("input") and "_batch" in dir_name:
            try:
                parts = dir_name.replace("input", "").split("_batch")
                seq = int(parts[0])
                batch = int(parts[1])

                model_subdir = os.path.join(dir_path, model_dir)
                if os.path.exists(model_subdir):
                    available.append({
                        "dir_name": dir_name,
                        "seq": seq,
                        "batch": batch,
                        "path": f"{dir_name}/{model_dir}"
                    })
            except (ValueError, IndexError):
                continue

    return sorted(available, key=lambda x: (x["seq"], x["batch"]))


def find_best_matching_data_dir(model_name: str,
                                target_seq: int,
                                target_batch: int) -> str:
    """
    查找最接近目标参数的可用数据目录。

    搜索策略:
    1. 优先匹配序列长度 >= target_seq 的目录
    2. 其次匹配批次大小 >= target_batch 的目录
    3. 如果没有更大的，使用最接近的较小值
    """
    model_dir = get_model_dir_name(model_name)
    available_dirs = list_available_data_dirs(model_name)

    if not available_dirs:
        print(f"[Warning] No data directory found for model {model_name}")
        return f"input{target_seq}_batch{target_batch}/{model_dir}"

    # 排序: 优先选择 seq >= target_seq 且 batch >= target_batch 的目录
    def score_dir(d):
        seq_diff = d["seq"] - target_seq
        batch_diff = d["batch"] - target_batch

        # 优先选择 >= 目标值的目录
        if seq_diff >= 0 and batch_diff >= 0:
            return (0, seq_diff + batch_diff)  # 最优: 都大于等于目标
        elif seq_diff >= 0:
            return (1, seq_diff + abs(batch_diff))  # 次优: seq满足
        elif batch_diff >= 0:
            return (2, abs(seq_diff) + batch_diff)  # 再次: batch满足
        else:
            return (3, abs(seq_diff) + abs(batch_diff))  # 最后: 都不满足

    available_dirs.sort(key=score_dir)
    best_match = available_dirs[0]

    return best_match["path"]


def get_data_dir(model_name: str, seq_length: int, batch_size: int) -> str:
    """
    根据模型名称、序列长度和批次大小动态生成数据目录路径。

    目录命名规则: input{seq_length}_batch{batch_size}/{model_dir_name}

    例如:
        model="Mixtral-8x7B", seq=1024, batch=16
        -> "input1024_batch16/mixtral_8x7b"

    如果精确匹配的目录不存在，会尝试查找最接近的可用目录。
    """
    model_dir = get_model_dir_name(model_name)

    # 首先尝试精确匹配
    exact_path = f"input{seq_length}_batch{batch_size}/{model_dir}"
    full_path = os.path.join("gate_weight_data", exact_path)

    if os.path.exists(full_path):
        return exact_path
    else:
        # 如果精确匹配不存在，查找可用的目录
        print(f"    [Error] Exact data directory not found: {exact_path}. Searching for best match...")
        # raise SystemExit()
        return find_best_matching_data_dir(model_name, seq_length, batch_size)


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class ProfilingResult:
    """Stores profiling result for a specific configuration."""
    token_fusion_threshold: float  # T
    expert_merge_percent: float    # P
    batch_size: int
    seq_length: int
    data_dir: str = ""            # 使用的数据目录
    # Timing results (in ms)
    s_time: float = 0.0           # Summarization (prefill) time
    g_time: float = 0.0           # Generation (decode) time
    total_latency: float = 0.0    # Total end-to-end latency
    # Breakdown (for analysis)
    s_fc_time: float = 0.0
    s_comm_time: float = 0.0
    g_fc_time: float = 0.0
    g_ff_time: float = 0.0
    g_comm_time: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorkloadBucket:
    """Represents a workload bucket with batch size and sequence length ranges."""
    batch_range: Tuple[int, int]
    seq_range: Tuple[int, int]

    @property
    def batch_min(self) -> int:
        return self.batch_range[0]

    @property
    def batch_max(self) -> int:
        return self.batch_range[1]

    @property
    def seq_min(self) -> int:
        return self.seq_range[0]

    @property
    def seq_max(self) -> int:
        return self.seq_range[1]

    def __repr__(self):
        return f"Bucket(B={self.batch_range}, S={self.seq_range})"

    def to_dict(self) -> Dict:
        return {
            "batch_range": list(self.batch_range),
            "seq_range": list(self.seq_range),
        }

    def get_key(self) -> str:
        return f"B{self.batch_min}-{self.batch_max}_S{self.seq_min}-{self.seq_max}"


# ==============================================================================
# CSV Writer (compatible with courier_main.py)
# ==============================================================================

def write_csv(logfile: str, perfs: List):
    """Write simulation results to CSV file (compatible with courier_main.py format)."""
    if logfile is None:
        return

    firstrow = not os.path.exists(logfile)

    with open(logfile, 'a', newline='') as f:
        wrt = csv.writer(f)
        if firstrow:
            col_name = [
                'model', 'dtype', 'xpu', 'cap', 'bw', 'sys_opb', 'hw', 'cores',
                'pipe_level', 'is parallel', 'power constraint', 'gqa_size',
                'Lin', 'Lout', 'bs', 'required_cap', 's_flops',
                'g_flops', 's_time', 's_matmul', 's_fc', 's_comm', 's_softmax',
                's_act', 's_lnorm', 'g_time (ms)', 'g_matmul', 'g_fc', 'g_comm',
                'g_etc', 'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g2g_comm',
                'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm', 'g_energy (nJ)',
                'g_dram_energy', 'g_l2_energy', 'g_l1_energy', 'g_reg_energy',
                'g_alu_energy', 'g_fc_mem_energy', 'g_fc_comp_energy',
                'g_attn_mem_energy', 'g_attn_comp_energy', 'g_etc_mem_energy',
                'g_etc_comp_energy', 'g_comm_energy'
            ]
            wrt.writerow(col_name)

        for perf in perfs:
            tag, config, time_data, energy = perf
            info = tag + config + time_data + energy
            wrt.writerow(info)


# ==============================================================================
# Latency Surface Profiler
# ==============================================================================

class LatencySurfaceProfiler:
    """
    Hardware Latency Surface Profiler for DWAP Phase 2.

    This class implements the constrained optimization within the feasible region
    to find optimal (T, P) configurations for each workload bucket.

    It directly uses the simulation system from courier_main.py.
    """

    def __init__(self,
                 model_name: str = "Mixtral-8x7B",
                 gpu_type: str = "RTX4090",
                 num_gpu: int = 1,
                 gmemcap: int = 80,
                 pim_type: str = "ddr4",
                 num_acc: int = 1,
                 num_channel: int = 2,
                 schedule_strategy: str = "FUSION",
                 mapping_strategy: str = "WEIGHT",
                 word_size: int = 2,
                 power_limit: bool = False,
                 pipe_opt: bool = False,
                 ff_opt: bool = False):
        """
        Initialize the profiler with system configuration.

        Args:
            model_name: Model to profile (e.g., "Mixtral-8x7B", "DeepSeek-16B", "Qwen-2.7B")
            gpu_type: GPU type ("A100a", "H100", "RTX4090")
            num_gpu: Number of GPUs
            gmemcap: GPU memory capacity in GB
            pim_type: PIM/NMP type ("bank", "bg", "buffer", "ddr4")
            num_acc: Number of accelerators
            num_channel: Number of memory channels
            schedule_strategy: Scheduling strategy
            mapping_strategy: Expert mapping strategy
            word_size: Precision (1=INT8, 2=FP16)
            power_limit: Apply power constraint for PIM
            pipe_opt: Apply pipeline optimization
            ff_opt: Apply feedforward parallel optimization
        """
        self.model_name = model_name
        self.gpu_type_str = gpu_type
        self.num_gpu = num_gpu
        self.gmemcap = gmemcap
        self.pim_type_str = pim_type
        self.num_acc = num_acc
        self.num_channel = num_channel
        self.schedule_strategy_str = schedule_strategy
        self.mapping_strategy_str = mapping_strategy
        self.word_size = word_size
        self.power_limit = power_limit
        self.pipe_opt = pipe_opt
        self.ff_opt = ff_opt

        # Parse enums (same logic as courier_main.py)
        self.gpu_type = self._parse_gpu_type(gpu_type)
        self.pim_type = self._parse_pim_type(pim_type)
        self.mapping_strategy = self._parse_mapping_strategy(mapping_strategy)
        self.schedule_strategy = self._parse_schedule_strategy(schedule_strategy)
        self.dtype = DataType.W16A16 if word_size == 2 else DataType.W8A8

        # Workload configuration
        self.batch_buckets = DEFAULT_BATCH_BUCKETS
        self.seq_buckets = DEFAULT_SEQ_LENGTH_BUCKETS
        self.batch_representatives = DEFAULT_BATCH_REPRESENTATIVES
        self.seq_representatives = DEFAULT_SEQ_REPRESENTATIVES

        # Results storage
        self.profiling_results: Dict[str, List[ProfilingResult]] = {}
        self.optimal_configs: Dict[str, Dict[str, Any]] = {}

    def _parse_gpu_type(self, gpu_str: str) -> GPUType:
        mapping = {
            "A100a": GPUType.A100a,
            "H100": GPUType.H100,
            "RTX4090": GPUType.RTX4090,
        }
        if gpu_str not in mapping:
            raise ValueError(f"Unknown GPU type: {gpu_str}. Valid: {list(mapping.keys())}")
        return mapping[gpu_str]

    def _parse_pim_type(self, pim_str: str) -> PIMType:
        mapping = {
            "bank": PIMType.BA,
            "bg": PIMType.BG,
            "buffer": PIMType.BUFFER,
            "ddr4": PIMType.DDR4,
        }
        return mapping.get(pim_str.lower(), PIMType.DDR4)

    def _parse_mapping_strategy(self, strategy_str: str) -> MappingStrategyType:
        mapping = {
            "NAIVE": MappingStrategyType.NAIVE,
            "H2": MappingStrategyType.H2,
            "WEIGHT": MappingStrategyType.WEIGHT,
        }
        return mapping.get(strategy_str.upper(), MappingStrategyType.WEIGHT)

    def _parse_schedule_strategy(self, strategy_str: str) -> ScheduleStrategyType:
        mapping = {
            "FUSION": ScheduleStrategyType.FUSION,
            "NOFUSION": ScheduleStrategyType.NOFUSION,
            "PIMOE": ScheduleStrategyType.PIMOE,
            "FIDDLER": ScheduleStrategyType.FIDDLER,
            "KLOTSKI": ScheduleStrategyType.KLOTSKI,
        }
        return mapping.get(strategy_str.upper(), ScheduleStrategyType.FUSION)

    def _create_system(self,
                       tfs_file: str,
                       gss_file: str,
                       elp_file: str) -> System:
        """
        Create and configure the simulation system.
        This follows the exact same logic as courier_main.py main() function.
        """
        # Model configuration
        moe = self.model_name in ['DeepSeek-16B', 'Qwen-2.7B', 'Mixtral-8x7B', 'Qwen-3-30B']
        modelinfos = make_model_config(self.model_name, self.dtype, moe=moe)

        # GPU configuration
        gmem_cap = self.gmemcap * 1024 * 1024 * 1024
        xpu_config = make_xpu_config(self.gpu_type, num_gpu=self.num_gpu, mem_cap=gmem_cap)

        # Data file paths
        tfs_path = os.path.join("gate_weight_data", tfs_file)
        print('tfs_path:', tfs_path)
        gss_path = os.path.join("gate_weight_data", gss_file)
        print('gss_path:', gss_path)
        elp_path = os.path.join("gate_weight_data", elp_file)
        print('elp_path:', elp_path)

        # Create system
        system = System(
            xpu_config['GPU'],
            modelinfos,
            expert_token_fusion_stats_path=tfs_path,
            expert_gate_sum_stats_path=gss_path,
            expert_location_path=elp_path
        )

        # Configure PIM/NMP accelerator (dgx-attacc mode)
        pim_config = make_pim_config(
            self.pim_type,
            self.mapping_strategy,
            InterfaceType.NVLINK3,
            num_attacc=self.num_acc,
            num_hbm=self.num_channel,
            power_constraint=self.power_limit
        )
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config)

        return system

    def run_single_simulation(self,
                              system: System,
                              batch_size: int,
                              seq_length: int,
                              lout: int = 2) -> List:
        """
        Run a single simulation and return performance results.
        This follows the run() function logic from courier_main.py.
        """
        perfs = []

        try:
            system.simulate(
                batch_size=batch_size,
                lin=seq_length,
                lout=lout,
                perfs=perfs,
                pipe=self.pipe_opt,
                parallel_ff=self.ff_opt,
                power_constraint=self.power_limit,
                schedule_strategy=self.schedule_strategy,
                attn_on_hetero=False,
                act_on_hetero=False,
                moe_on_hetero=True
            )
            return perfs

        except Exception as e:
            print(f"    [Error] Simulation failed: {e}")
            return []

    def parse_perf_results(self, perfs: List) -> Optional[Dict[str, float]]:
        """
        Parse the performance results from simulation.
        """
        if not perfs or len(perfs) == 0:
            return None

        try:
            tag, config, time_perf, energy = perfs[0]

            result = {
                's_time': time_perf[0] if len(time_perf) > 0 else 0.0,
                's_matmul': time_perf[1] if len(time_perf) > 1 else 0.0,
                's_fc': time_perf[2] if len(time_perf) > 2 else 0.0,
                's_comm': time_perf[3] if len(time_perf) > 3 else 0.0,
                's_softmax': time_perf[4] if len(time_perf) > 4 else 0.0,
                's_act': time_perf[5] if len(time_perf) > 5 else 0.0,
                's_norm': time_perf[6] if len(time_perf) > 6 else 0.0,
                'g_time': time_perf[7] if len(time_perf) > 7 else 0.0,
                'g_matmul': time_perf[8] if len(time_perf) > 8 else 0.0,
                'g_fc': time_perf[9] if len(time_perf) > 9 else 0.0,
                'g_comm': time_perf[10] if len(time_perf) > 10 else 0.0,
                'g_etc': time_perf[11] if len(time_perf) > 11 else 0.0,
                'g_qkv': time_perf[12] if len(time_perf) > 12 else 0.0,
                'g_prj': time_perf[13] if len(time_perf) > 13 else 0.0,
                'g_ff': time_perf[14] if len(time_perf) > 14 else 0.0,
                'total_latency': (time_perf[0] if len(time_perf) > 0 else 0.0) +
                                 (time_perf[7] if len(time_perf) > 7 else 0.0),
            }
            return result

        except Exception as e:
            print(f"  [Warning] Failed to parse perf results: {e}")
            return None

    def load_feasible_region(self, filepath: str) -> List[Tuple[float, float]]:
        """
        Load the feasible region R_safe from Phase 1 accuracy profiling.
        """
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [(item['T'], item['P']) for item in data.get('feasible_region', [])]
            except Exception as e:
                print(f"[Warning] Failed to load feasible region: {e}")

        print("[Info] Using default parameter grid")
        return self._generate_default_feasible_region()

    def _generate_default_feasible_region(self) -> List[Tuple[float, float]]:
        """
        Generate a default feasible region based on available data files.
        """
        # Based on available data files in the repository
        T_values = [0.49, 0.80]
        P_values = [0.50, 1.00]

        return list(product(T_values, P_values))

    def get_data_files_for_params(self,
                                  T: float,
                                  P: float,
                                  data_dir: str) -> Tuple[str, str, str]:
        """
        Get the data file paths for given (T, P) parameters.
        """
        t_str = f"{T:.2f}"
        r_str = f"{P:.2f}"

        tfs_file = f"{data_dir}/per_layer_expert_stats_t{t_str}_r{r_str}.json"
        gss_file = f"{data_dir}/expert_gate_sum_t{t_str}_r{r_str}.json"
        elp_file = f"{data_dir}/expert_location_path.json"

        return tfs_file, gss_file, elp_file

    def profile_configuration(self,
                              T: float,
                              P: float,
                              batch_size: int,
                              seq_length: int,
                              lout: int = 2) -> Optional[ProfilingResult]:
        """
        Profile a single (T, P, batch, seq) configuration.
        Data directory is dynamically determined based on parameters.
        """
        # 动态获取数据目录
        data_dir = get_data_dir(self.model_name, seq_length, batch_size)
        print(f"    B={batch_size}, S={seq_length} (dir: {data_dir})...")
        # 获取该 (T, P) 对应的数据文件
        tfs_file, gss_file, elp_file = self.get_data_files_for_params(T, P, data_dir)

        # 检查文件是否存在
        tfs_path = os.path.join("gate_weight_data", tfs_file)
        if not os.path.exists(tfs_path):
            print(f"    [Error] File does not exist: {tfs_path}")
            return None

        gss_path = os.path.join("gate_weight_data", gss_file)
        if not os.path.exists(gss_path):
            print(f"    [Error] File does not exist: {gss_path}")
            return None

        elp_path = os.path.join("gate_weight_data", elp_file)
        if not os.path.exists(elp_path):
            print(f"    [Error] File does not exist: {elp_path}")
            return None

        # 创建系统
        try:
            system = self._create_system(tfs_file, gss_file, elp_file)
        except Exception as e:
            print(f"  [Error] Failed to create system: {e}")
            return None

        # 运行模拟
        perfs = self.run_single_simulation(system, batch_size, seq_length, lout)

        # 解析结果
        parsed = self.parse_perf_results(perfs)
        if parsed is None:
            return None

        # 创建结果对象
        result = ProfilingResult(
            token_fusion_threshold=T,
            expert_merge_percent=P,
            batch_size=batch_size,
            seq_length=seq_length,
            data_dir=data_dir,
            s_time=parsed.get('s_time', 0.0),
            g_time=parsed.get('g_time', 0.0),
            total_latency=parsed.get('total_latency', 0.0),
            s_fc_time=parsed.get('s_fc', 0.0),
            s_comm_time=parsed.get('s_comm', 0.0),
            g_fc_time=parsed.get('g_fc', 0.0),
            g_ff_time=parsed.get('g_ff', 0.0),
            g_comm_time=parsed.get('g_comm', 0.0),
        )

        return result

    def profile_workload_bucket(self,
                                bucket: WorkloadBucket,
                                feasible_region: List[Tuple[float, float]],
                                verbose: bool = True) -> Dict[str, Any]:
        """
        Profile a single workload bucket to find optimal (T, P) configuration.
        Data directories are dynamically selected for each (batch, seq) sample.
        """
        bucket_key = bucket.get_key()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling Bucket: {bucket}")
            print(f"Model: {self.model_name}")
            print(f"{'='*60}")

        # 获取该桶的代表值
        batch_reps = self.batch_representatives.get(bucket.batch_range, [bucket.batch_min])
        seq_reps = self.seq_representatives.get(bucket.seq_range, [bucket.seq_min])

        if verbose:
            print(f"  Batch representatives: {batch_reps}")
            print(f"  Seq representatives: {seq_reps}")

            # 显示将使用的数据目录示例
            sample_batch = batch_reps[0]
            sample_seq = seq_reps[0]
            sample_dir = get_data_dir(self.model_name, sample_seq, sample_batch)
            print(f"  Sample data dir: {sample_dir}")

        # 存储结果
        all_results: List[ProfilingResult] = []
        config_latencies: Dict[Tuple[float, float], List[float]] = {}

        total_configs = len(feasible_region)
        config_count = 0

        # 遍历可行域 (约束优化)
        for T, P in feasible_region:
            config_count += 1

            if verbose:
                print(f"\n  Config {config_count}/{total_configs}: T={T:.2f}, P={P:.2f}")

            latencies = []

            # 对每个代表性 (batch, seq) 组合进行profiling
            for batch_size in batch_reps:
                for seq_length in seq_reps:
                    # 调用模拟
                    result = self.profile_configuration(T, P, batch_size, seq_length)

                    if result is not None:
                        all_results.append(result)
                        latencies.append(result.total_latency)
                        if verbose:
                            print(f"    Latency={result.total_latency:.3f}ms")
                    else:
                        if verbose:
                            print("    Skipped (no data)")

            if latencies:
                config_latencies[(T, P)] = latencies
                if verbose:
                    avg = sum(latencies) / len(latencies)
                    print(f"    -> Avg latency for (T={T:.2f}, P={P:.2f}): {avg:.3f}ms")

        # 找最优配置 (最小化平均延迟)
        best_config = None
        best_avg_latency = float('inf')

        for (T, P), latencies in config_latencies.items():
            avg_latency = sum(latencies) / len(latencies)
            if avg_latency < best_avg_latency:
                best_avg_latency = avg_latency
                best_config = {
                    'T': T,
                    'P': P,
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min(latencies),
                    'max_latency_ms': max(latencies),
                    'num_samples': len(latencies),
                }

        # 保存结果
        self.profiling_results[bucket_key] = all_results
        self.optimal_configs[bucket_key] = best_config

        if verbose and best_config:
            print(f"\n  >>> Optimal Config for {bucket}:")
            print(f"      T={best_config['T']:.2f}, P={best_config['P']:.2%}")
            print(f"      Avg Latency: {best_config['avg_latency_ms']:.3f} ms")

        return {
            'bucket': bucket.to_dict(),
            'optimal_config': best_config,
            'num_results': len(all_results),
        }

    def profile_all_buckets(self,
                            feasible_region: List[Tuple[float, float]],
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Profile all workload buckets and generate the complete LUT.
        """
        print("\n" + "="*80)
        print("DWAP Phase 2: Hardware Latency Surface Mapping")
        print("="*80)
        print(f"Model: {self.model_name}")
        print(f"GPU: {self.gpu_type_str} x {self.num_gpu}")
        print(f"NMP: {self.pim_type_str}, {self.num_channel} channels")
        print(f"Schedule Strategy: {self.schedule_strategy_str}")
        print(f"Mapping Strategy: {self.mapping_strategy_str}")
        print(f"Feasible Region Size: {len(feasible_region)} configurations")
        print(f"Workload Buckets: {len(self.batch_buckets)} x {len(self.seq_buckets)} = "
              f"{len(self.batch_buckets) * len(self.seq_buckets)}")

        # 显示可用的数据目录
        available_dirs = list_available_data_dirs(self.model_name)
        if available_dirs:
            print(f"\nAvailable data directories for {self.model_name}:")
            for d in available_dirs:
                print(f"  - {d['path']} (seq={d['seq']}, batch={d['batch']})")
        else:
            print(f"\n[Warning] No data directories found for {self.model_name}")

        print("="*80)

        start_time = time.time()

        # Profile每个桶
        bucket_results = []
        for batch_bucket in self.batch_buckets:
            for seq_bucket in self.seq_buckets:
                bucket = WorkloadBucket(batch_bucket, seq_bucket)
                result = self.profile_workload_bucket(
                    bucket, feasible_region, verbose
                )
                bucket_results.append(result)

        elapsed_time = time.time() - start_time

        # 生成LUT
        lut = self.generate_lut()

        print("\n" + "="*80)
        print("Profiling Complete!")
        print(f"Total Time: {elapsed_time:.2f} seconds")
        print("="*80)

        return lut

    def generate_lut(self) -> Dict[str, Any]:
        """
        Generate the final Look-Up Table (LUT) from profiling results.
        """
        lut = {
            "metadata": {
                "model": self.model_name,
                "gpu_type": self.gpu_type_str,
                "num_gpu": self.num_gpu,
                "pim_type": self.pim_type_str,
                "num_channel": self.num_channel,
                "schedule_strategy": self.schedule_strategy_str,
                "mapping_strategy": self.mapping_strategy_str,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description": "DWAP Phase 2 - Hardware Latency Surface LUT",
            },
            "workload_buckets": {
                "batch_buckets": [list(b) for b in self.batch_buckets],
                "seq_buckets": [list(s) for s in self.seq_buckets],
            },
            "optimal_configs": {},
            "detailed_results": {},
        }

        # Add optimal configs (the main LUT data)
        for bucket_key, config in self.optimal_configs.items():
            if config:
                lut["optimal_configs"][bucket_key] = config

        # Add detailed results (for analysis/debugging)
        for bucket_key, results in self.profiling_results.items():
            lut["detailed_results"][bucket_key] = {
                "num_samples": len(results),
                "samples": [r.to_dict() for r in results[:20]],
            }

        return lut

    def save_lut(self, lut: Dict[str, Any], filepath: str):
        """Save the LUT to a JSON file."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(lut, f, indent=2, ensure_ascii=False)

        print(f"\nLUT saved to: {filepath}")

    def save_detailed_csv(self, filepath: str):
        """Save all profiling results to a detailed CSV file."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'bucket_key', 'T', 'P', 'batch_size', 'seq_length', 'data_dir',
                's_time_ms', 'g_time_ms', 'total_latency_ms',
                's_fc_time_ms', 's_comm_time_ms',
                'g_fc_time_ms', 'g_ff_time_ms', 'g_comm_time_ms'
            ])

            for bucket_key, results in self.profiling_results.items():
                for r in results:
                    writer.writerow([
                        bucket_key,
                        r.token_fusion_threshold,
                        r.expert_merge_percent,
                        r.batch_size,
                        r.seq_length,
                        r.data_dir,
                        r.s_time,
                        r.g_time,
                        r.total_latency,
                        r.s_fc_time,
                        r.s_comm_time,
                        r.g_fc_time,
                        r.g_ff_time,
                        r.g_comm_time,
                    ])

        print(f"Detailed results saved to: {filepath}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DWAP Phase 2: Hardware Latency Surface Mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and system configuration
    parser.add_argument(
        "--model", type=str, default="Mixtral-8x7B",
        help="Model to profile (Mixtral-8x7B, DeepSeek-16B, Qwen-2.7B, Qwen-3-30B)"
    )
    parser.add_argument(
        "--gpu", type=str, default="RTX4090",
        help="GPU type (A100a, H100, RTX4090)"
    )
    parser.add_argument(
        "--ngpu", type=int, default=1,
        help="Number of GPUs"
    )
    parser.add_argument(
        "--gmemcap", type=int, default=80,
        help="GPU memory capacity in GB"
    )
    parser.add_argument(
        "--pim", type=str, default="ddr4",
        help="PIM/NMP type (bank, bg, buffer, ddr4)"
    )
    parser.add_argument(
        "--num_acc", type=int, default=1,
        help="Number of accelerators"
    )
    parser.add_argument(
        "--num_channel", type=int, default=2,
        help="Number of memory channels"
    )
    parser.add_argument(
        "--schedule_strategy", type=str, default="FUSION",
        help="Schedule strategy (FUSION, NOFUSION, PIMOE, FIDDLER, KLOTSKI)"
    )
    parser.add_argument(
        "--mapping_strategy", type=str, default="WEIGHT",
        help="Mapping strategy (NAIVE, H2, WEIGHT)"
    )
    parser.add_argument(
        "--word", type=int, default=2,
        help="Word size/precision (1=INT8, 2=FP16)"
    )
    parser.add_argument(
        "--powerlimit", action="store_true",
        help="Apply power constraint for PIM"
    )
    parser.add_argument(
        "--pipeopt", action="store_true",
        help="Apply pipeline optimization"
    )
    parser.add_argument(
        "--ffopt", action="store_true",
        help="Apply feedforward parallel optimization"
    )

    # Input/Output files
    parser.add_argument(
        "--feasible_region_file", type=str, default="",
        help="Path to feasible region JSON from Phase 1 (optional)"
    )
    parser.add_argument(
        "--output_lut", type=str, default="results/latency_lut.json",
        help="Output path for the Look-Up Table JSON"
    )
    parser.add_argument(
        "--output_csv", type=str, default="results/latency_profiling_detailed.csv",
        help="Output path for detailed profiling results CSV"
    )

    # Utility options
    parser.add_argument(
        "--list_data_dirs", action="store_true",
        help="List available data directories for the specified model and exit"
    )

    # Profiling options
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed progress"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed output"
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Run quick test with minimal configuration"
    )
    parser.add_argument(
        "--full_grid", action="store_true",
        help="Use full workload grid (all 3x3 buckets)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 如果用户只想查看可用的数据目录
    if args.list_data_dirs:
        print(f"\nAvailable data directories for model '{args.model}':")
        print(f"Model directory name: {get_model_dir_name(args.model)}")
        print()
        available = list_available_data_dirs(args.model)
        if available:
            for d in available:
                print(f"  - {d['path']}")
                print(f"      seq_length: {d['seq']}, batch_size: {d['batch']}")
        else:
            print(f"  No data directories found for {args.model}")
            print(f"  Expected pattern: gate_weight_data/input{{seq}}_batch{{batch}}/{{model_dir}}")
        return 0

    # 处理verbose/quiet
    verbose = args.verbose and not args.quiet

    # 创建profiler
    profiler = LatencySurfaceProfiler(
        model_name=args.model,
        gpu_type=args.gpu,
        num_gpu=args.ngpu,
        gmemcap=args.gmemcap,
        pim_type=args.pim,
        num_acc=args.num_acc,
        num_channel=args.num_channel,
        schedule_strategy=args.schedule_strategy,
        mapping_strategy=args.mapping_strategy,
        word_size=args.word,
        power_limit=args.powerlimit,
        pipe_opt=args.pipeopt,
        ff_opt=args.ffopt,
    )

    # 加载可行域
    if args.feasible_region_file and os.path.exists(args.feasible_region_file):
        feasible_region = profiler.load_feasible_region(args.feasible_region_file)
    else:
        feasible_region = profiler._generate_default_feasible_region()

    # 配置运行模式
    if args.quick_test:
        print("[Info] Quick test mode: minimal configuration")
        # 使用可用的数据文件
        feasible_region = [(0.49, 1.00), (0.80, 1.00)]
        # 单个桶，单个代表值
        profiler.batch_buckets = [(1, 4)]
        profiler.seq_buckets = [(0, 512)]
        profiler.batch_representatives = {(1, 4): [16]}
        profiler.seq_representatives = {(0, 512): [1024]}

    elif args.full_grid:
        print("[Info] Full grid mode: all 3x3 buckets")
        # 使用默认的完整配置

    else:
        # 默认: 中等规模配置
        profiler.batch_buckets = [(1, 4), (5, 16)]
        profiler.seq_buckets = [(0, 512), (513, 1024)]
        profiler.batch_representatives = {
            (1, 4): [1],
            (5, 16): [16],
        }
        profiler.seq_representatives = {
            (0, 512): [512],
            (513, 1024): [1024],
        }

    # 运行profiling
    lut = profiler.profile_all_buckets(
        feasible_region=feasible_region,
        verbose=verbose,
    )

    # 保存结果
    profiler.save_lut(lut, args.output_lut)
    profiler.save_detailed_csv(args.output_csv)

    # 打印汇总
    print("\n" + "="*80)
    print("Summary: Optimal Configurations per Workload Bucket")
    print("="*80)
    for bucket_key, config in profiler.optimal_configs.items():
        if config:
            print(f"\n  {bucket_key}:")
            print(f"    T = {config['T']:.2f}")
            print(f"    P = {config['P']:.2%}")
            print(f"    Avg Latency = {config['avg_latency_ms']:.3f} ms")
            print(f"    Min Latency = {config['min_latency_ms']:.3f} ms")
            print(f"    Max Latency = {config['max_latency_ms']:.3f} ms")
        else:
            print(f"\n  {bucket_key}: No valid configuration found")
    print("\n" + "="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())