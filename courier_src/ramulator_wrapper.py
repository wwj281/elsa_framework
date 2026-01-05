import pandas as pd
import subprocess
import math
import os
from courier_src.config import *
from courier_src.model import *
from courier_src.type import *


class Ramulator:

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 pim_type: PIMType,
                 mapping_strategy: MappingStrategyType = MappingStrategyType.NAIVE, 
                 output_log='',
                 fast_mode=False,
                 num_hbm=5):
        self.df = pd.DataFrame()
        self.ramulator_dir = ramulator_dir
        self.pim_type = pim_type
        self.mapping_strategy = mapping_strategy
        self.output_log = output_log
        if os.path.exists(output_log):
            self.df = pd.read_csv(output_log)
        if self.pim_type == PIMType.DDR4:
            self.tCK = 0.625  # ns
        else:
            self.tCK = 0.769
        self.num_hbm = num_hbm
        self.nhead = modelinfos['num_heads']
        self.dhead = modelinfos['dhead']
        self.moe = modelinfos['moe']
        if self.moe:
            self.num_experts = modelinfos['num_experts']
            self.token_experts = modelinfos['activated_experts']
            self.shared_experts = modelinfos['shared_experts']
            self.hidden_size = modelinfos['hdim']
            self.moe_intermediate_size = math.ceil(modelinfos['hdim'] * modelinfos['ff_scale'])
            self.shared_moe_intermediate_size = math.ceil(modelinfos['hdim'] * modelinfos['ff_scale'])
        self.fast_mode = fast_mode

    def make_yaml_file(self, yaml_file, file_name, power_constraint):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")
        line = ""
        line += "Frontend:\n"
        line += "  impl: PIMLoadStoreTrace\n"
        line += "  path: {}\n".format(trace_path)
        line += "  clock_ratio: 1\n"
        line += "\n"
        line += "  Translation:\n"
        line += "    impl: NoTranslation\n"
        line += "    max_addr: 2147483648\n"
        line += "              \n"
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: PIMDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: DDR4-PIM\n"
        line += "    org:\n"
        line += "      preset: DDR4_512Gb_x16\n"
        line += "      channel: 8\n"
        line += "    timing:\n"
        if power_constraint:
            line += "      preset: DDR4_3200W\n"
        else:
            line += "      preset: DDR4_3200AA\n"
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: DDR-PIM\n"
        line += "    Scheduler:\n"
        line += "      impl: PIM\n"
        line += "    RefreshManager:\n"
        line += "      impl: AllBank\n"
        line += "      #impl: No\n"
        line += "    plugins:\n"
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: ChRaBaRoCo\n"
        with open(yaml_file, 'w') as f:
            f.write(line)

    def update_log_file(self, log):
        if self.df.empty:
            if os.path.exists(self.output_log):
                df = pd.read_csv(self.output_log)
            else:
                if self.moe:
                    columns = [
                        't', 'h_size', 'i_size', 'dbyte', 'pim_type',
                        'power_constraint', 'cycle', 'mac', 'mvgb', 'wrgb',
                        'acc', 'af', 'ewmul'
                    ]
                else:
                    columns = [
                        'L', 'nhead', 'dhead', 'dbyte', 'pim_type',
                        'power_constraint', 'cycle', 'mac', 'softmax', 'mvgb',
                        'mvsb', 'wrgb'
                    ]
                df = pd.DataFrame(columns=columns)
        else:
            df = self.df
        if len(df.columns) > 12:
            import pdb
            pdb.set_trace()
        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log
        df = pd.concat([df, new_df]).drop_duplicates()
        self.df = df
        self.df.to_csv(self.output_log, index=False)

    # def run_ramulator(self):
    def run_ramulator(self, pim_type: PIMType, l, num_ops_per_hbm, dbyte,
                      yaml_file, file_name, batch_size):
        pim_type_name = pim_type.name.lower(
        ) if not pim_type == PIMType.BA else "bank"
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')

        # trace_exc = os.path.join(
        #     self.ramulator_dir,
        #     "trace_gen/gen_trace_attacc_{}.py".format(pim_type_name))
        # trace_args = "--dhead {} --nhead {} --seqlen {} --dbyte {} --output {}".format(
        #     self.dhead, num_ops_per_hbm, l, dbyte, trace_file)
        if self.mapping_strategy == MappingStrategyType.NAIVE:
            trace_exc = os.path.join(self.ramulator_dir, "trace_gen/gen_trace_naive.py")
            trace_args = "--num_experts {} --token_experts {} --shared_experts {} --hidden_size {} --moe_intermediate_size {} --shared_moe_intermediate_size {} --batch_size {} --dbyte {} --output {}".format(
                self.num_experts, self.token_experts, self.shared_experts, self.hidden_size, self.moe_intermediate_size,
                self.moe_intermediate_size, batch_size, dbyte, trace_file)
        else:
            trace_exc = os.path.join(self.ramulator_dir, "trace_gen/gen_trace_h2.py")
            trace_args = "--num_experts {} --token_experts {} --shared_experts {} --hidden_size {} --moe_intermediate_size {} --shared_moe_intermediate_size {} --batch_size {} --dbyte {} --output {} --token_num {}".format(
                self.num_experts, self.token_experts, self.shared_experts, self.hidden_size, self.moe_intermediate_size,
                self.moe_intermediate_size, batch_size, dbyte, trace_file, l)

        gen_trace_cmd = f"python {trace_exc} {trace_args}"

        # generate trace
        try:
            os.system(gen_trace_cmd)
        except Exception as e:
            print(f"Error: {e}")
        # run ramulator
        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
        run_ramulator_cmd = f"{ramulator_file} -f {yaml_file}"
        try:
            result = subprocess.run(run_ramulator_cmd,
                                    stdout=subprocess.PIPE,
                                    text=True,
                                    shell=True)
            output_lines = result.stdout.strip().split('\n')
            output_list = [line.strip() for line in output_lines]
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            assert 0

        # # remove trace
        # rm_trace_cmd = f"rm {trace_file}"
        # try:
        #     os.system(rm_trace_cmd)
        # except Exception as e:
        #     print(f"Error: {e}")

        # parsing output
        if self.moe:
            n_cmds = {"mac": 0, "mvgb": 0, "wrgb": 0, "acc": 0, "af": 0, "ewmul": 0}
            cycle = 0
            for line in output_list:
                if "mac" in line:
                    n_cmds["mac"] += int(line.split()[-1])
                elif "move_to_gemv_buffer" in line:
                    n_cmds["mvgb"] += int(line.split()[-1])
                elif "write_to_gemv_buffer" in line:
                    n_cmds["wrgb"] += int(line.split()[-1])
                elif "accumulate" in line:
                    n_cmds["acc"] += int(line.split()[-1])
                elif "activate_function" in line:
                    n_cmds["af"] += int(line.split()[-1])
                elif "multiply_element_wise" in line:
                    n_cmds["ewmul"] += int(line.split()[-1])
                elif "memory_system_cycles" in line:
                    cycle += int(line.split()[-1])

            out = [
                cycle, n_cmds["mac"], n_cmds["mvgb"], n_cmds["wrgb"], n_cmds["acc"],
                n_cmds["af"], n_cmds["ewmul"]
            ]
        else:
            n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
            cycle = 0
            for line in output_list:
                if "mac" in line:
                    n_cmds["mac"] += int(line.split()[-1])
                elif "softmax_requests" in line:
                    n_cmds["sfm"] += int(line.split()[-1])
                elif "move_to_gemv_buffer" in line:
                    n_cmds["mvgb"] += int(line.split()[-1])
                elif "move_to_softmax_buffer" in line:
                    n_cmds["mvsb"] += int(line.split()[-1])
                elif "write_to_gemv_buffer" in line:
                    n_cmds["wrgb"] += int(line.split()[-1])
                elif "memory_system_cycles" in line:
                    cycle += int(line.split()[-1])

            out = [
                cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"], n_cmds["mvsb"],
                n_cmds["wrgb"]
            ]
        return out

    def run(self, pim_type: PIMType, layer: Layer, power_constraint=True, batch_size=1):
        if os.path.exists(self.ramulator_dir):
            if layer.type == LayerType.FC:
                l = layer.m
            else:
                l = layer.n
            dhead = self.dhead
            dbyte = layer.dbyte
            num_ops_per_attacc = layer.numOp
            num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
                num_ops_per_hbm = minimum_heads
            h_size = self.hidden_size
            i_size = self.moe_intermediate_size
            if self.moe:
                file_name = "courier_t{}_h{}_i{}_dbyte{}_pc{}".format(
                    l, h_size, i_size, layer.dbyte, int(power_constraint))
            else:
                file_name = "attacc_l{}_nattn{}_dhead{}_dbyte{}_pc{}".format(
                    l, num_ops_per_hbm, dhead, layer.dbyte, int(power_constraint))
            yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
            self.make_yaml_file(yaml_file, file_name, power_constraint)

            result = self.run_ramulator(pim_type, l, num_ops_per_hbm,
                                        layer.dbyte, yaml_file, file_name, batch_size)

            # # remove trace
            # rm_yaml_cmd = f"rm {yaml_file}"
            # try:
            #     os.system(rm_yaml_cmd)
            # except Exception as e:
            #     print(f"Error: {e}")

            # post processing
            # 32: read granularity
            if pim_type == PIMType.DDR4:
                cycle, mac, mvgb, wrgb, acc, af, ewmul = result
                si_io = wrgb * 128  # 256 bit
                tsv_io = (wrgb + mvgb) * 128
                giomux_io = (wrgb + mvgb) * 128
                bgmux_io = (wrgb + mvgb) * 128
                mem_acc = mac * 128
            elif pim_type in [PIMType.BA, PIMType.BG, PIMType.BUFFER]:
                cycle, mac, sfm, mvgb, mvsb, wrgb = result
                si_io = wrgb * 32  # 256 bit
                tsv_io = (wrgb + mvsb + mvgb) * 32
                giomux_io = (wrgb + mvsb + mvgb) * 32
                bgmux_io = (wrgb + mvsb + mvgb) * 32
                mem_acc = mac * 32
            else:
                assert "Incorrect pim type!"
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            elif pim_type == PIMType.DDR4:
                mem_acc *= 2 * 2 * 8 * 4
            else:
                mem_acc *= 1

            ## update log file
            if self.moe:
                log = [
                          l, h_size, i_size, dbyte, pim_type.name,
                          power_constraint
                      ] + result
            else:
                log = [
                          l, num_ops_per_hbm, dhead, dbyte, pim_type.name,
                          power_constraint
                      ] + result
            self.update_log_file(log)

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            print('cycle:', cycle, 'exec_time:', exec_time)
            return exec_time, traffic

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True, batch_size=1):
        if self.df.empty:
            print('self.run 1')
            self.run(pim_type, layer, power_constraint, batch_size)

        num_ops_per_attacc = layer.numOp
        num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
        num_ops_group = 1
        if self.fast_mode:
            minimum_heads = 64
            num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
            num_ops_per_hbm = minimum_heads

        dbyte = layer.dbyte
        if self.moe:
            l = layer.m
            h_size = self.hidden_size
            i_size = self.moe_intermediate_size
            row = self.df[(self.df['t'] == l) & (self.df['h_size'] == h_size) & \
                          (self.df['i_size'] == i_size) & (self.df['dbyte'] == dbyte) & \
                          (self.df['power_constraint'] == power_constraint) & \
                          (self.df['pim_type'] == pim_type.name)]
        else:
            l = layer.n
            dhead = layer.k
            row = self.df[(self.df['L'] == l) & (self.df['nhead'] == num_ops_per_hbm) & \
                          (self.df['dbyte'] == dbyte) & (self.df['dhead'] == dhead) & \
                          (self.df['power_constraint'] == power_constraint) & \
                          (self.df['pim_type'] == pim_type.name)]
        if row.empty:
            print('self.run 2')
            return self.run(pim_type, layer, power_constraint, batch_size)

        else:
            cycle = int(row.iloc[0]['cycle'])
            mac = int(row.iloc[0]['mac'])
            mvgb = int(row.iloc[0]['mvgb'])
            wrgb = int(row.iloc[0]['wrgb'])
            if pim_type == PIMType.DDR4:
                softmax = 0
                si_io = wrgb * 128  # 1024 bit
                tsv_io = (wrgb + mvgb) * 128
                giomux_io = (wrgb + mvgb) * 128
                bgmux_io = (wrgb + mvgb) * 128
                mem_acc = mac * 128
            else:
                softmax = int(row.iloc[0]['softmax'])
                mvsb = int(row.iloc[0]['mvsb'])
                si_io = wrgb * 32  # 256 bit
                tsv_io = (wrgb + mvsb + mvgb) * 32
                giomux_io = (wrgb + mvsb + mvgb) * 32
                bgmux_io = (wrgb + mvsb + mvgb) * 32
                mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            elif pim_type == PIMType.DDR4:
                mem_acc *= 2 * 2 * 8 * 4
            else:
                mem_acc *= 2

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            exec_time *= num_ops_group
            return exec_time, traffic
