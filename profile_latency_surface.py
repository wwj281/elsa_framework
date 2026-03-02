#!/usr/bin/env python3
"""
DWAP Phase 2: Hardware Latency Surface Mapping (硬件负载曲面测绘)

This script performs offline profiling for the second phase of DWAP:
1. Load the feasible region (R_safe) from Phase 1 accuracy profiling
2. Build workload grid (Batch Size x Seq Length buckets)
3. For each workload bucket (B, S):
   - Traverse (T, P) combinations within R_safe
   - Call the actual simulation system to measure end-to-end latency
   - Record system latency max(T_GPU, T_NMP)
   - Find optimal configuration (T_opt, P_opt) that minimizes latency
4. Finalize the results into a static Look-Up Table (LUT) as JSON

Usage:
    python scripts/profile_latency_surface.py \
        --model Mixtral-8x7B \
        --feasible_region_file results/feasible_region.json \
        --output_lut results/latency_lut.json

    # Quick test mode
    python scripts/profile_latency_surface.py --quick_test

    # Full profiling with all workload buckets
    python scripts/profile_latency_surface.py --model Mixtral-8x7B --full_grid
"""

import argparse
import csv
import json
import os
import sys
import time
import copy
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict

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
# These are the actual values we will run simulations on
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
# Data Classes
# ==============================================================================

@dataclass
class ProfilingResult:
    """Stores profiling result for a specific configuration."""
    token_fusion_threshold: float  # T
    expert_merge_percent: float    # P
    batch_size: int
    seq_length: int
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
    # GPU vs NMP (from expert scheduling)
    gpu_expert_count: int = 0
    nmp_expert_count: int = 0
    
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
# CSV Writer (same as courier_main.py)
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
            schedule_strategy: Scheduling strategy ("FUSION", "NOFUSION", "PIMOE", "FIDDLER", "KLOTSKI")
            mapping_strategy: Expert mapping strategy ("NAIVE", "H2", "WEIGHT")
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
        
        # System will be created per simulation to handle different data files
        self.system: Optional[System] = None
        
    def _parse_gpu_type(self, gpu_str: str) -> GPUType:
        mapping = {
            "A100a": GPUType.A100a,
            "H100": GPUType.H100,
            "RTX4090": GPUType.RTX4090,
        }
        if gpu_str not in mapping:
            raise ValueError(f"Unknown GPU type: {gpu_str}. Valid options: {list(mapping.keys())}")
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
        gss_path = os.path.join("gate_weight_data", gss_file)
        elp_path = os.path.join("gate_weight_data", elp_file)
        
        # Validate files exist
        for path, name in [(tfs_path, "TFS"), (gss_path, "GSS"), (elp_path, "ELP")]:
            if not os.path.exists(path):
                print(f"  [Warning] {name} file not found: {path}")
        
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
                              lout: int = 2) -> Tuple[List, Optional[Dict]]:
        """
        Run a single simulation and return performance results.
        This follows the run() function logic from courier_main.py.
        
        Args:
            system: Configured System instance
            batch_size: Batch size
            seq_length: Input sequence length (lin)
            lout: Output tokens to generate
            
        Returns:
            Tuple of (perfs list, expert_schedule dict or None)
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
            return perfs, None
            
        except Exception as e:
            print(f"  [Error] Simulation failed for B={batch_size}, S={seq_length}: {e}")
            return [], None
    
    def parse_perf_results(self, perfs: List) -> Optional[Dict[str, float]]:
        """
        Parse the performance results from simulation.
        
        The perfs structure is: [tag, config, time_perf, energy]
        - tag: [model, dtype, xpu, cap, bw, opb]
        - config: [hw, cores, pipe, parallel, power, gqa, lin, lout, bs, cap_usage, s_flops, g_flops]
        - time_perf: [s_all, s_matmul, s_fc, s_comm, s_softmax, s_act, s_norm,
        #             g_all, g_matmul, g_fc, g_comm, g_etc, g_qkv, g_prj, g_ff, g2g, x2g, g_softmax, g_act, g_norm]
        - energy: [...]
        """
        if not perfs or len(perfs) == 0:
            return None
            
        try:
            tag, config, time_perf, energy = perfs[0]
            
            # Extract timing information
            # Index mapping based on courier_main.py write_csv col_name
            result = {
                # Summarization phase (prefill)
                's_time': time_perf[0] if len(time_perf) > 0 else 0.0,
                's_matmul': time_perf[1] if len(time_perf) > 1 else 0.0,
                's_fc': time_perf[2] if len(time_perf) > 2 else 0.0,
                's_comm': time_perf[3] if len(time_perf) > 3 else 0.0,
                's_softmax': time_perf[4] if len(time_perf) > 4 else 0.0,
                's_act': time_perf[5] if len(time_perf) > 5 else 0.0,
                's_norm': time_perf[6] if len(time_perf) > 6 else 0.0,
                # Generation phase (decode)
                'g_time': time_perf[7] if len(time_perf) > 7 else 0.0,
                'g_matmul': time_perf[8] if len(time_perf) > 8 else 0.0,
                'g_fc': time_perf[9] if len(time_perf) > 9 else 0.0,
                'g_comm': time_perf[10] if len(time_perf) > 10 else 0.0,
                'g_etc': time_perf[11] if len(time_perf) > 11 else 0.0,
                'g_qkv': time_perf[12] if len(time_perf) > 12 else 0.0,
                'g_prj': time_perf[13] if len(time_perf) > 13 else 0.0,
                'g_ff': time_perf[14] if len(time_perf) > 14 else 0.0,
                # Total
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
        
        Expected JSON format:
        {
            "feasible_region": [
                {"T": 0.05, "P": 0.20},
                {"T": 0.05, "P": 0.25},
                ...
            ]
        }
        
        Args:
            filepath: Path to the feasible region JSON file
            
        Returns:
            List of (T, P) tuples representing safe parameter combinations
        """
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [(item['T'], item['P']) for item in data.get('feasible_region', [])]
            except Exception as e:
                print(f"[Warning] Failed to load feasible region: {e}")
        
        print("[Info] Using default parameter grid for demonstration")
        return self._generate_default_feasible_region()
    
    def _generate_default_feasible_region(self) -> List[Tuple[float, float]]:
        """
        Generate a default feasible region for demonstration purposes.
        
        Based on DWAP specification:
        - T (Token Fusion Threshold): 0.01 ~ 0.15, step 0.02
        - P (Expert Merge Percent): 10% ~ 50%, step 5%
        """
        # Token fusion thresholds to test
        T_values = [0.49, 0.60, 0.70, 0.80]  # Based on available data files
        # Expert merge percentages to test  
        P_values = [0.50, 0.75, 1.00]  # r values from file naming
        
        return list(product(T_values, P_values))
    
    def get_data_files_for_params(self, 
                                   T: float, 
                                   P: float,
                                   base_dir: str = "input1024_batch16/mixtral_8x7b") -> Tuple[str, str, str]:
        """
        Get the data file paths for given (T, P) parameters.
        
        The file naming convention is:
        - per_layer_expert_stats_t{T}_r{P}.json
        - expert_gate_sum_t{T}_r{P}.json
        - expert_location_path.json (shared)
        """
        # Format T and P for filename
        t_str = f"{T:.2f}"
        r_str = f"{P:.2f}"
        
        tfs_file = f"{base_dir}/per_layer_expert_stats_t{t_str}_r{r_str}.json"
        gss_file = f"{base_dir}/expert_gate_sum_t{t_str}_r{r_str}.json"
        elp_file = f"{base_dir}/expert_location_path.json"
        
        return tfs_file, gss_file, elp_file
    
    def profile_configuration(self,
                              T: float,
                              P: float,
                              batch_size: int,
                              seq_length: int,
                              tfs_file: str,
                              gss_file: str,
                              elp_file: str,
                              lout: int = 2) -> Optional[ProfilingResult]:
        """
        Profile a single (T, P, batch, seq) configuration.
        
        Returns:
            ProfilingResult or None if simulation failed
        """
        # Create system with specified data files
        try:
            system = self._create_system(tfs_file, gss_file, elp_file)
        except Exception as e:
            print(f"  [Error] Failed to create system: {e}")
            return None
        
        # Run simulation
        perfs, _ = self.run_single_simulation(system, batch_size, seq_length, lout)
        
        # Parse results
        parsed = self.parse_perf_results(perfs)
        if parsed is None:
            return None
        
        # Create result object
        result = ProfilingResult(
            token_fusion_threshold=T,
            expert_merge_percent=P,
            batch_size=batch_size,
            seq_length=seq_length,
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
                                data_base_dir: str,
                                verbose: bool = True) -> Dict[str, Any]:
        """
        Profile a single workload bucket to find optimal (T, P) configuration.
        
        This is the core of DWAP Phase 2: Constrained Optimization.
        For each bucket, we:
        1. Only search within the feasible region R_safe
        2. Test representative (batch, seq) values from the bucket
        3. Find (T, P) that minimizes average latency
        """
        bucket_key = bucket.get_key()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling Bucket: {bucket}")
            print(f"{'='*60}")
        
        # Get representative values for this bucket
        batch_reps = self.batch_representatives.get(bucket.batch_range, [bucket.batch_min])
        print('batch_reps:', batch_reps)
        seq_reps = self.seq_representatives.get(bucket.seq_range, [bucket.seq_min])
        print('seq_reps:', seq_reps)
        print(1/0)
        
        # Storage
        all_results: List[ProfilingResult] = []
        config_latencies: Dict[Tuple[float, float], List[float]] = {}
        
        total_configs = len(feasible_region)
        config_count = 0
        
        # Iterate through feasible region (constrained optimization)
        for T, P in feasible_region:
            config_count += 1
            
            if verbose:
                print(f"\n  Config {config_count}/{total_configs}: T={T:.2f}, P={P:.2f}")
            
            # Get data files for this (T, P) combination
            tfs_file, gss_file, elp_file = self.get_data_files_for_params(T, P, data_base_dir)
            
            # Check if files exist
            tfs_path = os.path.join("gate_weight_data", tfs_file)
            if not os.path.exists(tfs_path):
                if verbose:
                    print(f"    [Skip] Data file not found: {tfs_file}")
                continue
            
            latencies = []
            
            # Profile for each representative (batch, seq) combination
            for batch_size in batch_reps:
                for seq_length in seq_reps:
                    if verbose:
                        print(f"    Running B={batch_size}, S={seq_length}...", end=" ")
                    
                    result = self.profile_configuration(
                        T, P, batch_size, seq_length,
                        tfs_file, gss_file, elp_file
                    )
                    
                    if result is not None:
                        all_results.append(result)
                        latencies.append(result.total_latency)
                        if verbose:
                            print(f"Latency={result.total_latency:.3f}ms")
                    else:
                        if verbose:
                            print("Failed")
            
            if latencies:
                config_latencies[(T, P)] = latencies
        
        # Find optimal configuration (minimize average latency)
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
        
        # Store results
        self.profiling_results[bucket_key] = all_results
        self.optimal_configs[bucket_key] = best_config
        
        if verbose and best_config:
            print(f"\n  >>> Optimal Config for {bucket}:")
            print(f"      T={best_config['T']:.2f}, P={best_config['P']:.2%}")
            print(f"      Avg Latency: {best_config['avg_latency_ms']:.3f} ms")
            print(f"      Min Latency: {best_config['min_latency_ms']:.3f} ms")
            print(f"      Max Latency: {best_config['max_latency_ms']:.3f} ms")
        
        return {
            'bucket': bucket.to_dict(),
            'optimal_config': best_config,
            'num_results': len(all_results),
        }
    
    def profile_all_buckets(self,
                            feasible_region: List[Tuple[float, float]],
                            data_base_dir: str = "input1024_batch16/mixtral_8x7b",
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
        print("="*80)
        
        start_time = time.time()
        
        # Profile each bucket
        bucket_results = []
        for batch_bucket in self.batch_buckets:
            for seq_bucket in self.seq_buckets:
                bucket = WorkloadBucket(batch_bucket, seq_bucket)
                result = self.profile_workload_bucket(
                    bucket, feasible_region, data_base_dir, verbose
                )
                bucket_results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Generate LUT
        lut = self.generate_lut()
        
        print("\n" + "="*80)
        print("Profiling Complete!")
        print(f"Total Time: {elapsed_time:.2f} seconds")
        print("="*80)
        
        return lut
    
    def generate_lut(self) -> Dict[str, Any]:
        """
        Generate the final Look-Up Table (LUT) from profiling results.
        
        This is the static configuration file used at runtime.
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
                "samples": [r.to_dict() for r in results[:20]],  # Keep first 20
            }
        
        return lut
    
    def save_lut(self, lut: Dict[str, Any], filepath: str):
        """Save the LUT to a JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(lut, f, indent=2, ensure_ascii=False)
        
        print(f"\nLUT saved to: {filepath}")
    
    def save_detailed_csv(self, filepath: str):
        """Save all profiling results to a detailed CSV file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'bucket_key', 'T', 'P', 'batch_size', 'seq_length',
                's_time_ms', 'g_time_ms', 'total_latency_ms',
                's_fc_time_ms', 's_comm_time_ms',
                'g_fc_time_ms', 'g_ff_time_ms', 'g_comm_time_ms'
            ])
            
            # Data rows
            for bucket_key, results in self.profiling_results.items():
                for r in results:
                    writer.writerow([
                        bucket_key,
                        r.token_fusion_threshold,
                        r.expert_merge_percent,
                        r.batch_size,
                        r.seq_length,
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
    
    # Model and system configuration (same as courier_main.py)
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
    
    # Data directory
    parser.add_argument(
        "--data_dir", type=str,
        default="input1024_batch16/mixtral_8x7b",
        help="Base directory for gate weight data files"
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
    
    # Profiling options
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed progress"
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Run quick test with reduced parameter grid and single bucket"
    )
    parser.add_argument(
        "--full_grid", action="store_true",
        help="Use full workload grid (all buckets)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create profiler
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
    
    # Load feasible region
    if args.feasible_region_file and os.path.exists(args.feasible_region_file):
        feasible_region = profiler.load_feasible_region(args.feasible_region_file)
    else:
        feasible_region = profiler._generate_default_feasible_region()
    
    # Quick test mode: minimal configuration
    if args.quick_test:
        print("[Info] Quick test mode: using minimal configuration")
        # Use only available data files
        feasible_region = [(0.49, 1.00), (0.80, 1.00)]
        # Single bucket with single representative
        profiler.batch_buckets = [(1, 4)]
        profiler.seq_buckets = [(0, 512)]
        profiler.batch_representatives = {(1, 4): [1]}
        profiler.seq_representatives = {(0, 512): [512]}
    
    # Full grid mode (default)
    elif not args.full_grid:
        # Default: use medium-sized configuration
        profiler.batch_buckets = [(1, 4), (5, 16)]
        profiler.seq_buckets = [(0, 512), (513, 1024)]
        profiler.batch_representatives = {
            (1, 4): [1, 4],
            (5, 16): [8, 16],
        }
        profiler.seq_representatives = {
            (0, 512): [256, 512],
            (513, 1024): [768, 1024],
        }
    
    # Run profiling
    lut = profiler.profile_all_buckets(
        feasible_region=feasible_region,
        data_base_dir=args.data_dir,
        verbose=args.verbose,
    )
    
    # Save results
    profiler.save_lut(lut, args.output_lut)
    profiler.save_detailed_csv(args.output_csv)
    
    # Print summary
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