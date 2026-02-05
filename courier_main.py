import argparse
import csv
import os
from courier_src.system import *
from courier_src.type import *
from courier_src.config import *
from courier_src.ramulator_wrapper import *

RAMULATOR = False


def write_csv(logfile, perfs):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        f = open(logfile, 'a')
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
            tag, config, time, energy = perf
            info = tag + config + time + energy
            wrt.writerow(info)
        f.close()


def run(system: System,
        batch,
        lin,
        lout,
        power_constraint=False,
        pipe=0,
        parallel=False,
        output_file=None,
        schedule_strategy=ScheduleStrategyType.FUSION,
        attn_on_hetero=False,
        act_on_hetero=False,
        moe_on_hetero=True):
    print("---Run simple mode Batch {} Lin {} Lout {} pipe {} parall {}---".
          format(batch, lin, lout, pipe, parallel))
    assert system.model_set, "Need to SetModel"
    perfs = []
    system.simulate(batch,
                    lin,
                    lout,
                    perfs=perfs,
                    pipe=pipe,
                    parallel_ff=parallel,
                    power_constraint=power_constraint,
                    schedule_strategy=schedule_strategy,
                    attn_on_hetero=attn_on_hetero,
                    act_on_hetero=act_on_hetero,
                    moe_on_hetero=moe_on_hetero)
    if output_file is not None:
        write_csv(output_file, perfs)


def main():
    parser = argparse.ArgumentParser(
        description="Model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## set system configuration
    parser.add_argument(
        "--system",
        type=str,
        default="dgx-attacc",
        help="dgx (each GPU has 80GB HBM), \
              dgx-cpu (In dgx, offloading the attention layer to cpu), \
              dgx-attacc (dgx + attacc)")
    parser.add_argument(
        "--gpu",
        type=str,
        default='RTX4090',
        help="GPU type (A100a , H100 and RTX4090), A100a is A100 with HBM3")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="number of GPUs in DGX system. default=1")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB). default=80")
    parser.add_argument("--attn_on_hetero",
                        type=bool,
                        default=False,
                        help="whether to do attention on heterogeneous devices")
    parser.add_argument("--act_on_hetero",
                        type=bool,
                        default=False,
                        help="whether to activate on heterogeneous devices")
    parser.add_argument(
        "--tfs_file",
        type=str,
        default="input1024_batch16/mixtral_8x7b/per_layer_expert_stats_t0.49_r1.00.json",
        help="expert-token-fusion-stats file name")
    parser.add_argument(
        "--gss_file",
        type=str,
        default="input1024_batch16/mixtral_8x7b/expert_gate_sum_t0.49_r1.00.json",
        help="expert0-gate-sum-stats file name")
    parser.add_argument(
        "--elp_file",
        type=str,
        default="input1024_batch16/mixtral_8x7b/expert_location_path.json",
        help="expert-location-path file name")


    ## set attacc configuration
    parser.add_argument("--pim",
                        type=str,
                        default='ddr4',
                        help="pim mode. list: bank, bg, buffer, ddr4")
    parser.add_argument("--powerlimit",
                        action='store_true',
                        help="power constraint for PIM ")
    parser.add_argument("--ffopt",
                        action='store_true',
                        help="apply feedforward parallel optimization")
    parser.add_argument("--pipeopt",
                        action='store_true',
                        help="apply pipeline optimization ")
    parser.add_argument(
        "--schedule_strategy",
        type=str,
        default='FUSION',
        help="Schedule strategy type (FUSION, NOFUSION, PIMOE, FIDDLER)")
    parser.add_argument(
        "--mapping_strategy",
        type=str,
        default='WEIGHT',
        help="Mapping strategy type (NAIVE, H2, WEIGHT)")
    parser.add_argument("--num_acc",
                        type=int,
                        default=1,
                        help="number of accelerator in DGX system. default=1")
    parser.add_argument("--num_channel",
                        type=int,
                        default=2,
                        help="number of channel in accelerator. default=4")

    ## set model and service environment
    parser.add_argument(
        "--model",
        type=str,
        default='Mixtral-8x7B',
        help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B, DeepSeek-16B, Qwen-2.7B, Mixtral-8x7B")
    parser.add_argument("--word",
                        type=int,
                        default='2',
                        help="word size (precision): 1(INT8), 2(FP16)")
    parser.add_argument("--lin",
                        type=int,
                        default=1024,
                        help="input sequence length")
    parser.add_argument("--lout",
                        type=int,
                        default=2,
                        help="number of generated tokens")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help=
        "batch size, default = 1"
    )

    args = parser.parse_args()

    global RAMULATOR
    if RAMULATOR:
        print("The Ramulator {}".format(RAMULATOR))

    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    elif args.gpu == 'RTX4090':
        gpu_device = GPUType.RTX4090
    else:
        assert 0

    if args.system == 'dgx-attacc':
        print("{}: ({} x {}), PIM:{}, [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu, args.pim,
            [args.lin, args.lout, args.batch]))
    else:
        print("{}: ({} x {}), [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu,
            [args.lin, args.lout, args.batch]))
    num_gpu = args.ngpu
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024
    num_acc = args.num_acc
    num_channel = args.num_channel
    output_path = "output.csv"
    if os.path.exists(output_path):
        os.system("rm " + output_path)

    # set system
    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    moe = True if args.model in ['DeepSeek-16B', 'Qwen-2.7B', 'Mixtral-8x7B', 'Qwen-3-30B'] else False
    modelinfos = make_model_config(args.model, dtype, moe=moe)
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap)
    expert_token_fusion_stats_path = os.path.join("gate_weight_data", args.tfs_file)
    expert_gate_sum_stats_path = os.path.join("gate_weight_data", args.gss_file)
    expert_location_path = os.path.join("gate_weight_data", args.elp_file)
    system = System(xpu_config['GPU'],
                    modelinfos,
                    expert_token_fusion_stats_path=expert_token_fusion_stats_path,
                    expert_gate_sum_stats_path=expert_gate_sum_stats_path,
                    expert_location_path=expert_location_path)
    if args.system in ['dgx-attacc']:
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        elif args.pim == "ddr4":
            pim_type = PIMType.DDR4
        else:
            pim_type = PIMType.BA
        if args.mapping_strategy == 'NAIVE':
            mapping_strategy = MappingStrategyType.NAIVE
        elif args.mapping_strategy == 'WEIGHT':
            mapping_strategy = MappingStrategyType.WEIGHT
        else:
            mapping_strategy = MappingStrategyType.H2
        pim_config = make_pim_config(pim_type,  
                                     mapping_strategy,
                                     InterfaceType.NVLINK3,
                                     num_attacc=num_acc,
                                     num_hbm=num_channel,
                                     power_constraint=args.powerlimit)
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config)

    elif args.system in ['dgx-cpu']:
        xpu_config = make_xpu_config(gpu_device)
        system.set_xpu(xpu_config['GPU'])
        system.set_accelerator(modelinfos, DeviceType.CPU, xpu_config['CPU'])


    if args.schedule_strategy == 'NOFUSION':  
        schedule_strategy = ScheduleStrategyType.NOFUSION
    elif args.schedule_strategy == 'PIMOE':
        schedule_strategy = ScheduleStrategyType.PIMOE
    elif args.schedule_strategy == 'FIDDLER':
        schedule_strategy = ScheduleStrategyType.FIDDLER
    elif args.schedule_strategy == 'KLOTSKI':
        schedule_strategy = ScheduleStrategyType.KLOTSKI
    else:
        schedule_strategy = ScheduleStrategyType.FUSION

    run(system,
        args.batch,
        args.lin,
        args.lout,
        pipe=args.pipeopt,
        parallel=args.ffopt,
        output_file=output_path,
        power_constraint=args.powerlimit,
        schedule_strategy=schedule_strategy,
        attn_on_hetero=args.act_on_hetero,
        act_on_hetero=args.act_on_hetero,
        moe_on_hetero=args.system not in ['dgx'])


if __name__ == "__main__":
    main()
