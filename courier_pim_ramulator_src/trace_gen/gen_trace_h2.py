import argparse
import math
import copy
import numpy as np

model = "deepseek-moe-16B"

num_experts = 64
token_experts = 6
shared_experts = 2
hidden_size = 2048
moe_intermediate_size = 1408
shared_moe_intermediate_size = 1408
batch_size = 1
data_size = 2  # FP 16

n_channel = 2
n_dimm = 2
n_rank = 2
n_bg = 8
n_bank = 4
n_row = pow(2, 16)
n_col = pow(2, 6)
n_chip = 8
prefetch_size = 8 
n_mac = 8  # mac的数量就是一个物理bank一次能处理的操作数数量，在16bit DQ下，也等于BL

# Granularity size
DIMM_GS = {}
DIMM_GS['col'] = n_chip * data_size * prefetch_size 
DIMM_GS['row'] = n_col * DIMM_GS['col']
DIMM_GS['ba'] = n_row * DIMM_GS['row']  # 这里的bank是指逻辑bank，即多个chip中相同idx的所有bank
DIMM_GS['bg'] = n_bank * DIMM_GS['ba']
DIMM_GS['rank'] = n_bg * DIMM_GS['bg']
DIMM_GS['dimm'] = n_rank * DIMM_GS['rank']
DIMM_GS['ch'] = n_dimm * DIMM_GS['dimm']
DIMM_GS['courier'] = n_channel * DIMM_GS['ch']

## To do!!! 一共512GB的内存空间，单根DIMM 32GB
## --------------------------------------  DIMM memory space -----------------------------------------##
## ------|  legacy CH  |  dimm  |  rank  |   BG   |  BA  |  row index  |  column index  |  access granularity  |------ ##
## bits  |     2       |   2    |   1    |    3   |   2  |      14     |       8        |           7          |       ##

## ----------------------------  Commands -------------------------------##
## ACT: Activate all banks in parallel
## ACTASYNC: Activate a bank through AESPA
## MACAB: Perform MAC (Multiply-and-Accumulate) in all banks in parallel
## AF: Compute Activation Function (Non-linear function) in all banks
## ACC: Merge intermediate results
## EWMUL: Perform element-wise multiplication
## RDCP: Copy data from a bank to Global Buffer
## WRCP: Copy data from Global Buffer to a bank
## WRGB: Write Global Buffer


total_cmd = []
# cmd_gate_wrgb = []
# cmd_gate_mac = []
# cmd_af = []
# cmd_gate_mvgb = []
# cmd_gate_acc = []
# cmd_up_mac = []
# cmd_up_mvgb = []
# cmd_up_acc = []
# cmd_ewmul = []
# cmd_ewmul_mvgb = []
# cmd_down_mac = []
# cmd_down_mvgb = []
# cmd_down_acc = []

valid_dimms = []


def cmd_list_reset():
    total_cmd = []
    # cmd_gate_wrgb = []
    # cmd_gate_mac = []
    # cmd_af = []
    # cmd_gate_mvgb = []
    # cmd_gate_acc = []
    # cmd_up_mac = []
    # cmd_up_mvgb = []
    # cmd_up_acc = []
    # cmd_ewmul = []
    # cmd_ewmul_mvgb = []
    # cmd_down_mac = []
    # cmd_down_mvgb = []
    # cmd_down_acc = []

    valid_dimms = []


def Attention(gate_addr, up_addr, down_addr, itr, tiling_size_gate_up, tiling_size_down, valid_dimm=n_dimm):
    total_cmd.append([])
    k_ch_gate_up = tiling_size_gate_up['k_ch_gate_up']
    n_ch_gate_up = tiling_size_gate_up['n_ch_gate_up']
    k_ch_down = tiling_size_down['k_ch_down']
    n_ch_down = tiling_size_down['n_ch_down']
    # cmd_gate_wrgb.append([])
    # cmd_gate_mac.append([])
    # cmd_af.append([])
    # cmd_gate_mvgb.append([])
    # cmd_gate_acc.append([])
    # cmd_up_mac.append([])
    # cmd_up_mvgb.append([])
    # cmd_up_acc.append([])
    # cmd_ewmul.append([])
    # cmd_ewmul_mvgb.append([])
    # cmd_down_mac.append([])
    # cmd_down_mvgb.append([])
    # cmd_down_acc.append([])

    valid_dimms.append(valid_dimm)

    # Attacc中每个Channel处理的head数量是相同的，因此源代码中每次迭代需要向所有Channel发送相同的指令，
    # 但是当前映射模式下，各DIMM处理的专家数可能不同，因此只需要向被激活的DIMM发送指令，下面的代码可能需要修改
    def gate_cpvec(addr_offset):
        for col_idx in range(math.ceil(k_ch_gate_up / n_chip / n_mac)):
            for dimm_idx in range(math.ceil(valid_dimm)):
                addr = addr_offset + dimm_idx * DIMM_GS['dimm'] + col_idx * DIMM_GS['col']
                hex_addr = hex(addr)[2:]
                total_cmd[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

    def gate_mac(addr_offset):
        for n_idx in range(math.ceil(n_ch_gate_up / n_dimm / n_rank / n_bg / n_bank)):
            for k_idx in range(math.ceil(k_ch_gate_up / n_chip / n_mac)):
                idx = k_idx + n_idx * math.ceil(k_ch_gate_up / n_chip / n_mac)

                for dimm_idx in range(math.ceil(valid_dimm)):
                    addr = addr_offset + dimm_idx * DIMM_GS['dimm'] + idx * DIMM_GS['col']
                    hex_addr = hex(addr)[2:]
                    total_cmd[itr].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
                    if k_idx == (math.ceil(k_ch_gate_up / n_chip / n_mac) - 1):
                        total_cmd[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))
                        total_cmd[itr].append("PIM_ACC 0x{0:0>8}".format(hex_addr))

        # gate mac计算完后，以DIMM为单位合并中间结果，然后计算激活函数
        for n_idx in range(math.ceil(n_ch_gate_up / n_chip / n_mac / 16)):  # 假设DIMM NMP的算力是Bank NMP的16倍
            for dimm_idx in range(math.ceil(valid_dimm)):
                addr = addr_offset + dimm_idx * DIMM_GS['dimm'] + n_idx * DIMM_GS['col']
                hex_addr = hex(addr)[2:]
                total_cmd[itr].append("PIM_AF 0x{0:0>8}".format(hex_addr))

    def up_mac(addr_offset):
        for n_idx in range(math.ceil(n_ch_gate_up / n_dimm / n_rank / n_bg / n_bank)):
            for k_idx in range(math.ceil(k_ch_gate_up / n_chip / n_mac)):
                idx = k_idx + n_idx * math.ceil(k_ch_gate_up / n_chip / n_mac)

                for dimm_idx in range(math.ceil(valid_dimm)):
                    addr = addr_offset + dimm_idx * DIMM_GS['dimm'] + idx * DIMM_GS['col']
                    hex_addr = hex(addr)[2:]
                    total_cmd[itr].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
                    if k_idx == (math.ceil(k_ch_gate_up / n_chip / n_mac) - 1):
                        total_cmd[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))
                        total_cmd[itr].append("PIM_ACC 0x{0:0>8}".format(hex_addr))

    def ewmul(addr_offset):
        for k_idx in range(math.ceil(n_ch_gate_up / n_chip / n_mac / 16)):
            for dimm_idx in range(math.ceil(valid_dimm)):
                addr = addr_offset + dimm_idx * DIMM_GS['dimm'] + k_idx * DIMM_GS['col']
                hex_addr = hex(addr)[2:]
                total_cmd[itr].append("PIM_EWMUL 0x{0:0>8}".format(hex_addr))

    def down_cpvec(addr_offset):
        for col_idx in range(math.ceil(k_ch_down / n_chip / n_mac)):
            for dimm_idx in range(math.ceil(valid_dimm)):
                addr = addr_offset + dimm_idx * DIMM_GS['dimm'] + col_idx * DIMM_GS['col']
                hex_addr = hex(addr)[2:]
                total_cmd[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

    def down_mac(addr_offset):
        for n_idx in range(math.ceil(n_ch_down / n_dimm / n_rank / n_bg / n_bank)):
            for k_idx in range(math.ceil(k_ch_down / n_chip / n_mac)):
                idx = k_idx + n_idx * math.ceil(k_ch_down / n_chip / n_mac)

                for dimm_idx in range(math.ceil(valid_dimm)):
                    addr = addr_offset + dimm_idx * DIMM_GS['dimm'] + idx * DIMM_GS['col']
                    hex_addr = hex(addr)[2:]
                    total_cmd[itr].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
                    if k_idx == (math.ceil(k_ch_down / n_chip / n_mac) - 1):
                        total_cmd[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))
                        total_cmd[itr].append("PIM_ACC 0x{0:0>8}".format(hex_addr))

    def barrier():
        for dimm_idx in range(n_dimm):
            addr = dimm_idx * DIMM_GS['dimm']
            hex_addr = hex(addr)[2:]
            total_cmd[itr].append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

    # 目前假设没有流水线，所有操作都是顺序执行的，每一阶段计算完成后强制用barrier进行数据同步
    gate_cpvec(gate_addr)

    gate_mac(gate_addr)

    barrier()

    up_mac(up_addr)

    barrier()

    ewmul(up_addr)

    barrier()

    down_cpvec(down_addr)

    down_mac(down_addr)

    barrier()


# 暂时假设以DIMM为单位分配专家，可能导致DIMM之间负载不均，某些DIMM可能处于闲置状态
def run_attention(token_num, trace_file_name):
    # 暂时假设共享专家和普通专家的形状相同
    weight_offset = math.ceil(hidden_size * moe_intermediate_size / (n_channel * n_dimm * n_rank * n_bg * n_bank))
    t_k_gate_up = max(int(math.sqrt(n_channel * hidden_size / moe_intermediate_size)), 1)
    t_k_gate_up = 2 if t_k_gate_up == 1 and n_channel > 2 else t_k_gate_up
    while n_channel % t_k_gate_up != 0:
        t_k_gate_up -= 1
    t_n_gate_up = n_channel / t_k_gate_up
    t_k_down = max(int(math.sqrt(n_channel * moe_intermediate_size / hidden_size)), 1)
    t_k_down = 2 if t_k_down == 1 and n_channel > 2 else t_k_down
    while n_channel % t_k_down != 0:
        t_k_down -= 1
    t_n_down = n_channel / t_k_down
    tiling_size_gate_up = {'k_ch_gate_up': hidden_size//t_k_gate_up, 'n_ch_gate_up': moe_intermediate_size//t_n_gate_up}
    tiling_size_down = {'k_ch_down': moe_intermediate_size//t_k_down, 'n_ch_down': hidden_size//t_n_down}

    cmd_list_reset()
    ##-- Generate Commands --##
    # num_itr = math.ceil(n_expert_per_channel / (n_dimm))
    # 这里以处理专家数最多的DIMM作为衡量延迟的标准
    num_itr = token_num
    gate_addr = 0
    up_addr = gate_addr + weight_offset
    down_addr = gate_addr + weight_offset * 2
    for itr in range(num_itr):
        Attention(gate_addr, up_addr, down_addr, itr, tiling_size_gate_up, tiling_size_down)

    trace_file = open(trace_file_name, 'w')
    for itr in range(num_itr):
        for cmd in total_cmd[itr]:
            trace_file.write(cmd + "\n")

    trace_file.close()


def main():
    global num_experts, token_experts, shared_experts, hidden_size, moe_intermediate_size, shared_moe_intermediate_size, batch_size, n_channel

    parser = argparse.ArgumentParser(description="Output path and operation infos",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-ne", "--num_experts", type=int, default=60,
                        help="Number of routed experts, default = 64")
    parser.add_argument("-te", "--token_experts", type=int, default=4,
                        help="Number of activated experts per token, default = 6")
    parser.add_argument("-se", "--shared_experts", type=int, default=1,
                        help="Number of shared experts per token, default = 2")
    parser.add_argument("-hs", "--hidden_size", type=int, default=2048,
                        help="Hidden size, default= 2048")
    parser.add_argument("-mis", "--moe_intermediate_size", type=int, default=1408,
                        help="Moe layer intermediate size, default = 1408")
    parser.add_argument("-smis", "--shared_moe_intermediate_size", type=int, default=1408,
                        help="Shared moe layer intermediate size, default = 1408")
    parser.add_argument("-bs", "--batch_size", type=int, default=1,
                        help="Batch size, default = 1")
    parser.add_argument("-db", "--dbyte", type=int, default=2,
                        help="data type (B), default = 2")
    parser.add_argument("-tn", "--token_num", type=int, default=333,
                        help="Number of token in a expert, default = 1")
    parser.add_argument("-o", "--output", type=str, default="courier_pim.trace",
                        help="output path")
    parser.add_argument("-ch", "--num_channels", type=int, default=4,
                        help="Number of channels in NMP, default = 4")

    args = parser.parse_args()

    num_experts = args.num_experts
    token_experts = args.token_experts
    shared_experts = args.shared_experts
    hidden_size = args.hidden_size
    moe_intermediate_size = args.moe_intermediate_size
    shared_moe_intermediate_size = args.shared_moe_intermediate_size
    batch_size = args.batch_size
    token_num = args.token_num
    data_size = args.dbyte
    n_channel = args.num_channels

    print("------   Make a trace of naive courier mapping  ------")

    args_dict = vars(args)
    print("All Arguments:")
    for key, value in args_dict.items():
        print(f"     {key}: {value}")
    print("---------------------------------------------------")
    run_attention(token_num, args.output)


if __name__ == "__main__":
    main()
