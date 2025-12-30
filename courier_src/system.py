from .type import *
from .model import *
from .devices import *
from .config import *
import os
import json

RAMPATH = "./ramulator2"
RAMLOG = "./ramulator.out"

OPB_PRINT = False


class System:

    def __init__(self,
                 gpu_config,
                 modelinfos=None,
                 hetero_name: DeviceType = DeviceType.NONE,
                 hetero_config=None,
                 expert_token_fusion_stats_path=None,
                 expert_gate_sum_stats_path=None,
                 expert_location_path=None):
        # 新增：读取专家token融合统计和gate weight统计，文件路径由外部参数传入
        self.expert_token_fusion_stats = None
        self.expert_gate_sum_stats = None
        self.expert_location = None
        if expert_token_fusion_stats_path:
            try:
                with open(expert_token_fusion_stats_path, 'r', encoding='utf-8') as f:
                    self.expert_token_fusion_stats = json.load(f)
            except Exception as e:
                print(f"[System] Failed to load {expert_token_fusion_stats_path}: {e}")
        if expert_gate_sum_stats_path:
            try:
                with open(expert_gate_sum_stats_path, 'r', encoding='utf-8') as f:
                    self.expert_gate_sum_stats = json.load(f)
            except Exception as e:
                print(f"[System] Failed to load {expert_gate_sum_stats_path}: {e}")
        if expert_location_path:
            try:
                with open(expert_location_path, 'r', encoding='utf-8') as f:
                    self.expert_location = json.load(f)
            except Exception as e:
                print(f"[System] Failed to load {expert_location_path}: {e}")
        scaling_factor = SCALING_FACTOR
        self.hetero_name = hetero_name
        self.GPU = xPU(DeviceType.GPU, gpu_config, scaling_factor)
        self.AttDevice = self.GPU
        if self.hetero_name == DeviceType.PIM:
            self.AttDevice = PIM(hetero_config, scaling_factor)

        elif self.hetero_name == DeviceType.CPU:
            self.AttDevice = xPU(DeviceType.CPU, hetero_config, scaling_factor)

        self.devices = {'GPU': self.GPU, 'Acc': self.AttDevice}

        self.model_set = 0
        if modelinfos is not None:
            self.model = Transformer(modelinfos,
                                     tensor_parallel=self.GPU.num_xpu)
            self.model_set = 1

        self.scaling_factor = scaling_factor

    def set_model(self, modelinfos):
        self.model = Transformer(modelinfos, tensor_parallel=self.GPU.num_xpu)
        self.model_set = 1

    def set_accelerator(self, modelinfos, name: DeviceType, config):
        self.hetero_name = name
        if self.hetero_name == DeviceType.PIM:
            ramulator = Ramulator(modelinfos, "ramulator2", "ramulator.out")
            self.devices['Acc'] = PIM(config,
                                      self.scaling_factor,
                                      ramulator)

        elif self.hetero_name == DeviceType.CPU:
            self.devices['Acc'] = xPU(DeviceType.CPU, config,
                                      self.scaling_factor)

    # Set all device to GPU
    def set_xpu(self, config):
        self.hetero_name = DeviceType.NONE
        self.GPU = xPU(DeviceType.GPU, config, self.scaling_factor)
        self.devices['GPU'] = self.GPU
        self.devices['Acc'] = self.GPU
        self.model.tp = self.GPU.num_xpu

    def simulate(self,
                 batch_size,
                 lin,
                 lout,
                 perfs=None,
                 pipe=False,
                 parallel_ff=False,
                 power_constraint=False,
                 num_reqs=0,
                 attn_on_hetero=False,
                 act_on_hetero=False,
                 moe_on_hetero=True,
                 token_fusion_expert_ratio=None):

        def add_infos(name, infos, time, energy, bound):
            new_name = name
            if new_name in infos.keys():
                infos[new_name]["time"] += time
                infos[new_name]["energy"] = [
                    eng + energy[i]
                    for i, eng in enumerate(infos[new_name]["energy"])
                ]
            else:
                infos[new_name] = {
                    "time": time,
                    "energy": energy,
                    "bound": bound
                }

        def acc_time(type, exec_times, exec_time):
            if type in exec_times.keys():
                exec_times[type] += exec_time
            else:
                exec_times[type] = exec_time

        def acc_energy(type, energies, energy):
            if type in energies.keys():
                energy_ = energies[type]
                energies[type] = [
                    energy_[i] + energy[i] for i in range(len(energy_))
                ]
            else:
                energies[type] = energy

        def _opb_print(layer, stage_name):
            if OPB_PRINT and layer.off_traffic != 0:
                opb = layer.get_flops() / layer.off_traffic
                tflops = layer.get_flops(
                ) / exec_time / 1000 / 1000 / 1000 / 1000
                print("{},{},{},{},{},{}".format(stage_name, batch_size, lin,
                                                 layer.name, opb, tflops))

        def _pipeline(layers, level=False):
            qkv_time, prj_time, score_time, context_time, x2g_time, softmax_time = 0, 0, 0, 0, 0, 0
            for layer in layers:
                if layer.name in ["qkv"]:
                    qkv_time += layer.exec_time
                elif layer.name in ["proj"]:
                    prj_time += layer.exec_time
                elif layer.name in ["comm_x2g"]:
                    x2g_time += layer.exec_time
                elif layer.name in ["score"]:
                    score_time += layer.exec_time
                elif layer.name in ["context"]:
                    context_time += layer.exec_time
                elif layer.name in ["softmax"]:
                    softmax_time += layer.exec_time

            minimum_ratio = 1 / (self.model.num_heads / self.GPU.num_xpu)
            if level == False:
                # softmax_time = 0
                attn_time = score_time + context_time + softmax_time
                if attn_time > x2g_time:
                    x2g_time *= minimum_ratio
                else:
                    x2g_time -= attn_time * (1 - minimum_ratio)

            else:
                # softmax_time = 0
                fc_time = qkv_time + prj_time
                attn_time = score_time + context_time + softmax_time
                if attn_time > fc_time:
                    qkv_time *= minimum_ratio
                    prj_time *= minimum_ratio

                    if attn_time > x2g_time:
                        x2g_time *= minimum_ratio
                    else:
                        x2g_time -= attn_time * (1 - minimum_ratio)
                else:
                    if fc_time > x2g_time:
                        x2g_time *= minimum_ratio
                        qkv_time -= attn_time * (1 - minimum_ratio) * (3 / 4)
                        prj_time -= attn_time * (1 - minimum_ratio) * (1 / 4)
                    else:
                        x2g_time -= attn_time * (1 - minimum_ratio)
                        qkv_time *= minimum_ratio
                        prj_time *= minimum_ratio
            softmax_time = 0

            for layer in layers:
                if layer.name in ["qkv"]:
                    layer.exec_time = qkv_time
                elif layer.name in ["proj"]:
                    layer.exec_time = prj_time
                elif layer.name in ["comm_x2g"]:
                    # for 2 comm_x2g layers
                    layer.exec_time = x2g_time / 2
                elif layer.name in ["softmax"]:
                    layer.exec_time = softmax_time

        def _ff_parallel(layers):
            bw_scale = self.devices['Acc'].peak_memory_bandwidth / self.devices[
                'GPU'].peak_memory_bandwidth
            for layer in layers:
                if "ff" in layer.name:
                    if layer.bound == "compute":
                        attn_flops = self.devices[
                                         'GPU'].peak_memory_bandwidth / layer.dbyte * 2 * bw_scale
                        ratio = self.devices['GPU'].peak_flops / (
                                self.devices['GPU'].peak_flops + attn_flops)
                        layer.exec_time *= ratio

                    elif layer.bound == "memory":
                        attn_eff_bw = self.devices[
                                          'GPU'].peak_memory_bandwidth * bw_scale / bs
                        ratio = self.devices['GPU'].peak_memory_bandwidth / (
                                self.devices['GPU'].peak_memory_bandwidth +
                                attn_eff_bw)
                        layer.exec_time *= ratio

        assert self.model_set, "Need to set_model"
        self.model.build(batch_size, lin, lout, attn_on_hetero=attn_on_hetero, act_on_hetero=act_on_hetero, moe_on_hetero=moe_on_hetero)
        second_batch_size = num_reqs % batch_size
        num_batches = 1
        target_bs = [batch_size]
        if num_reqs > 0:
            num_batches = int(num_reqs / batch_size)
            if second_batch_size > 0:
                target_bs = [batch_size, second_batch_size]

        s_flops = 0
        g_flops = 0

        gen_energies = {}

        unit_energy = {
            'g_all': 0,
            'g_offmem': 0,
            'g_l2': 0,
            'g_l1': 0,
            'g_reg': 0,
            'g_alu': 0,
            'g_comm': 0
        }

        perf_all = []
        energy_all = []
        ramulator_call_count = 0
        expert_schedule = self.expert_schedule_simulation()

        for itr, bs in enumerate(target_bs):
            time = 0
            wrt_io_busy = 0
            s_decoder = self.model.sum_decoder
            g_decoder = self.model.gen_decoder

            ## Summarization stage
            for idx, layer in enumerate(s_decoder):
                # 动态调度X2G和ff层
                if layer.type == LayerType.X2G:
                    # 统计需要搬运到GPU的专家数
                    move_to_gpu_ids = expert_schedule.get('move_to_gpu_ids', [])
                    if move_to_gpu_ids:
                        move_expert_num = len(move_to_gpu_ids)  
                        # 构造新layer，继承原layer属性
                        new_layer = copy.deepcopy(layer)
                        new_layer.m = new_layer.m * move_expert_num  # 修改m为需要搬运的专家数
                        # 用GPU计算X2G搬运时间
                        exec_time, energy = self.devices['GPU'].get_time_and_energy(new_layer)
                        exec_time += max(wrt_io_busy - time, 0)
                        wrt_io_busy = time + exec_time
                        layer.exec_time = exec_time
                        layer.energy = energy
                    else:
                        # 没有专家需要搬运，X2G层时间为0
                        layer.exec_time = 0
                        layer.energy = [0, 0, 0, 0, 0, 0]
                elif layer.name in ['ff1', 'ff2', 'ff3', 'silu'] and self.hetero_name in [DeviceType.CPU, DeviceType.PIM]:
                    if layer.name == 'ff1':
                        # ff层对每个专家分别构造layer并计算延迟
                        gpu_expert_ids = expert_schedule.get('gpu_expert_ids', [])
                        acc_expert_ids = expert_schedule.get('acc_expert_ids', [])
                        expert_actual_tokens = expert_schedule.get('expert_actual_tokens', {})
                        gpu_total_time = 0.0
                        acc_total_time = 0.0
                        gpu_total_energy = [0, 0, 0, 0, 0, 0]
                        acc_total_energy = [0, 0, 0, 0, 0, 0]
                        # 分别计算GPU专家
                        for eid in gpu_expert_ids:
                            tokens = expert_actual_tokens.get(eid, 0)
                            act_layer = copy.deepcopy(s_decoder[idx + 2])    # 激活函数层
                            if tokens > 0:
                                expert_layer = copy.deepcopy(layer)
                                expert_layer.m = tokens
                                act_layer.m = tokens
                                t, e = self.devices['GPU'].get_time_and_energy(expert_layer)
                                act_t, act_e = self.devices['GPU'].get_time_and_energy(act_layer)
                                e = [e[i] * 3 + act_e[i] for i in range(len(e))]
                                gpu_total_time += t * 3 + act_t # ff1, ff2, ff3
                                gpu_total_energy = [a + b for a, b in zip(gpu_total_energy, e)]
                        # 分别计算加速器专家
                        for eid in acc_expert_ids:
                            tokens = expert_actual_tokens.get(eid, 0)
                            if tokens > 0:
                                expert_layer = copy.deepcopy(layer)
                                expert_layer.m = tokens
                                t, e = self.devices['Acc'].get_time_and_energy(expert_layer)
                                acc_total_time += t
                                acc_total_energy = [a + b for a, b in zip(acc_total_energy, e)]
                        # 取决于搬运延迟
                        x2g_time = 0
                        for l in s_decoder:
                            if l.type == LayerType.X2G:
                                x2g_time = l.exec_time
                                break
                        # 汇总延迟
                        if acc_total_time > gpu_total_time + x2g_time:
                            layer.exec_time = acc_total_time
                            layer.energy = acc_total_energy
                        else:
                            layer.exec_time = gpu_total_time
                            layer.energy = gpu_total_energy
                else:
                    # 其他层保持原逻辑
                    exec_time, energy = self.devices['GPU'].get_time_and_energy(layer)
                    layer.exec_time = exec_time
                    layer.energy = energy

                s_flops += layer.get_flops() * self.devices['GPU'].num_xpu
                time += layer.exec_time
                _opb_print(layer, 'sum')
                print('Prefill layer:', layer.type, 'exec_time:', layer.exec_time)

            ## Generation stage
            for gen_stage, decoder_block in enumerate(g_decoder):
                ramulator_call_count = 0
                for l_idx, layer in enumerate(decoder_block):
                    if self.hetero_name in [DeviceType.CPU, DeviceType.PIM]:
                        if layer.name not in ['ff1', 'ff2', 'ff3'] and layer.type not in [LayerType.ACT]:
                            exec_time, energy = self.devices['GPU'].get_time_and_energy(layer)
                        else:
                            if self.hetero_name in [DeviceType.CPU]:
                                exec_time, energy = self.devices[
                                    'Acc'].get_time_and_energy(layer, batch_size)
                            else:
                                if layer.name == 'ff1' and ramulator_call_count == 0:
                                    exec_time, energy = self.devices[
                                    'Acc'].get_time_and_energy(layer, batch_size)
                                    ramulator_call_count += 1
                                else:
                                    exec_time, energy = 0, [0, 0, 0, 0, 0, 0]
                    else:
                        exec_time, energy = self.devices['GPU'].get_time_and_energy(layer)
                    layer.exec_time = exec_time
                    layer.energy = energy
                    g_flops += layer.get_flops() * self.devices['GPU'].num_xpu
                    time += exec_time
                    print('Decode layer:', layer.type, 'exec_time:', layer.exec_time)
                    if gen_stage == 0:
                        _opb_print(layer, 'gen')
                    # energy
                    if layer.type in gen_energies:
                        gen_energies[layer.type]['mem'] += layer.energy[0]
                        gen_energies[layer.type]['comp'] += sum(
                            layer.energy[1:5])
                        gen_energies[layer.type]['comm'] += layer.energy[5]
                    else:
                        gen_energies[layer.type] = {}
                        gen_energies[layer.type]['mem'] = layer.energy[0]
                        gen_energies[layer.type]['comp'] = sum(
                            layer.energy[1:5])
                        gen_energies[layer.type]['comm'] = layer.energy[5]

                    unit_energy['g_all'] += sum(layer.energy)
                    unit_energy['g_offmem'] += layer.energy[0]
                    unit_energy['g_l2'] += layer.energy[1]
                    unit_energy['g_l1'] += layer.energy[2]
                    unit_energy['g_reg'] += layer.energy[3]
                    unit_energy['g_alu'] += layer.energy[4]
                    unit_energy['g_comm'] += layer.energy[5]

                # pipeline
                if self.hetero_name == DeviceType.PIM:
                    _pipeline(decoder_block, pipe)
                    if parallel_ff:
                        _ff_parallel(decoder_block)

            s_perf = {
                'all': 0,
                'matmul': 0,
                'fc': 0,
                'comm': 0,
                'softmax': 0,
                'act': 0,
                'norm': 0
            }
            for layer in s_decoder:
                exec_time = layer.exec_time
                if layer.type == LayerType.FC:
                    s_perf['all'] += exec_time
                    s_perf['fc'] += exec_time
                elif layer.type == LayerType.MATMUL:
                    s_perf['all'] += exec_time
                    s_perf['matmul'] += exec_time
                elif layer.type in [LayerType.G2G, LayerType.X2G]:
                    s_perf['all'] += exec_time
                    s_perf['comm'] += exec_time
                elif layer.type == LayerType.SOFTMAX:
                    s_perf['all'] += exec_time
                    s_perf['softmax'] += exec_time
                elif layer.type == LayerType.ACT:
                    s_perf['all'] += exec_time
                    s_perf['act'] += exec_time
                elif layer.type == LayerType.NORM:
                    s_perf['all'] += exec_time
                    s_perf['norm'] += exec_time

            g_perf = {
                'all': 0,
                'matmul': 0,
                'fc': 0,
                'comm': 0,
                'etc': 0,
                'qkv': 0,
                'prj': 0,
                'ff': 0,
                'g2g': 0,
                'x2g': 0,
                'softmax': 0,
                'act': 0,
                'norm': 0
            }

            for gen_stage, decoder_block in enumerate(g_decoder):
                for l_idx, layer in enumerate(decoder_block):
                    exec_time = layer.exec_time
                    g_perf['all'] += exec_time
                    if layer.type == LayerType.FC:
                        g_perf['fc'] += exec_time
                        if 'ff' in layer.name:
                            g_perf['ff'] += exec_time
                        elif 'qkv' in layer.name:
                            g_perf['qkv'] += exec_time
                        elif 'proj' in layer.name:
                            g_perf['prj'] += exec_time
                    elif layer.type == LayerType.MATMUL:
                        g_perf['matmul'] += exec_time
                    elif layer.type in [LayerType.G2G, LayerType.X2G]:
                        g_perf['comm'] += exec_time
                        if 'x2g' in layer.name:
                            g_perf['x2g'] += exec_time
                        elif 'g2g' in layer.name:
                            g_perf['g2g'] += exec_time
                    elif layer.type in [LayerType.ACT, LayerType.NORM]:
                        g_perf['etc'] += exec_time
                        if layer.type == LayerType.ACT:
                            g_perf['act'] += exec_time
                        elif layer.type == LayerType.NORM:
                            g_perf['norm'] += exec_time
                    elif layer.type == LayerType.SOFTMAX:
                        g_perf['softmax'] += exec_time

            # 计算整体延迟而不是输出单个Token的延迟
            # g_perf = {k: v / (lout - 1) for k, v in g_perf.items()}

            energies = [
                unit_energy['g_all'], unit_energy['g_offmem'],
                unit_energy['g_l2'], unit_energy['g_l1'], unit_energy['g_reg'],
                unit_energy['g_alu'], gen_energies[LayerType.FC]['mem'],
                gen_energies[LayerType.FC]['comp'],
                gen_energies[LayerType.MATMUL]['mem'] +
                gen_energies[LayerType.SOFTMAX]['mem'],
                gen_energies[LayerType.MATMUL]['comp'] +
                gen_energies[LayerType.SOFTMAX]['comp'],
                gen_energies[LayerType.ACT]['mem'] +
                gen_energies[LayerType.NORM]['mem'],
                gen_energies[LayerType.ACT]['comp'] +
                gen_energies[LayerType.NORM]['comp']
            ]
            comm_energy = sum([v['comm'] for k, v in gen_energies.items()])
            energies.append(comm_energy)

            # 计算整体能耗而不是输出单个Token的能耗
            # energies = [i / (lout - 1) for i in energies]

            perf = list(s_perf.values()) + list(g_perf.values())

            cap_usage = sum(self.get_required_mem_capacity(bs, lin, lout))

            ## Scaling to all decoder
            ## Perf: ms, energy: nJ
            perf = [t * self.model.ndec * 1000 for t in perf]
            energies = [t * self.model.ndec / 1000 for t in energies]

            if itr == 0:
                if len(perf_all) > 0:
                    perf_all = [
                        v + perf[i] * num_batches
                        for i, v in enumerate(perf_all)
                    ]
                    energy_all = [
                        v + energy[i] * num_batches
                        for i, v in enumerate(energy_all)
                    ]
                else:
                    perf_all = copy.deepcopy(perf)
                    energy_all = copy.deepcopy(energies)
            else:
                perf_all = [v + perf[i] for i, v in enumerate(perf_all)]
                energy_all = [v + energy[i] for i, v in enumerate(energy_all)]

        s_flops = s_flops * self.model.ndec / (lout - 1)
        g_flops = g_flops * self.model.ndec / (lout - 1)

        ## Concat tag
        cap = self.devices['GPU'].aggregate_memory_capacity
        if self.hetero_name in [DeviceType.CPU, DeviceType.PIM]:
            cap += self.devices['Acc'].aggregate_memory_capacity
        cap = int(cap / (1024 * 1024 * 1024))
        bw_scale = self.devices['Acc'].peak_memory_bandwidth / self.devices[
            'GPU'].peak_memory_bandwidth

        opb = self.devices['GPU'].peak_flops / self.devices[
            'GPU'].peak_memory_bandwidth
        if self.model.dtype in ['W8A8']:
            opb *= 2

        tag = [
            self.model.name, self.model.dtype.name,
            self.devices['GPU'].name.name, cap, bw_scale, opb
        ]
        config = [
            self.hetero_name.name, self.devices['GPU'].num_xpu, pipe,
            parallel_ff, power_constraint, 0, lin, lout, batch_size,
            cap_usage, s_flops, g_flops
        ]
        if self.hetero_name == DeviceType.PIM:
            config[0] = self.devices['Acc'].pim_type.name

        output = [tag, config, perf_all, energy_all]
        print(
            "    Batch: {}, Throughput: {:.2f} tokens/s Latency: {:.2f}ms, pipe/ff_parallel: {}/{}, powerlimit: {}"
                .format(batch_size, batch_size / ((perf_all[len(s_perf)]) / 1000),
                        perf_all[len(s_perf)], pipe, parallel_ff, power_constraint))

        if perfs is not None:
            perfs.append(output)
        else:
            perfs = [output]
    


    def get_expert_token_threshold(self):
        """
        根据token数、权重矩阵大小、FLOPS和带宽等参数，判断专家何时应load到GPU。
        假设：
        - 专家权重矩阵大小为 (k, n) * 3（3个权重矩阵）
        - 专家原本存储在加速器上
        - 加速器执行时间 = 矩阵乘法FLOPS/加速器算力 + 权重大小/加速器带宽
        - GPU执行时间 = 矩阵乘法FLOPS/GPU算力 + 权重大小/GPU带宽 + 权重大小/加速器到GPU带宽
        返回：token阈值（int），大于该值建议load到GPU
        """
        # 获取硬件参数
        gpu_flops = self.devices['GPU'].peak_flops  # FLOPS
        acc_flops = self.devices['Acc'].peak_flops  # FLOPS
        gpu_bw = self.devices['GPU'].peak_memory_bandwidth
        acc_bw = self.devices['Acc'].peak_memory_bandwidth
        acc_to_gpu_bandwidth = self.devices['GPU'].max_interface_bandwidth
        k = self.model.hdim  # 权重矩阵维度k
        n = self.model.hdim * self.model.ff_scale  # 权重矩阵维度n

        dtype_size = self.model.dtype # 可根据实际类型调整
        weight_size = k * n * 3 * dtype_size

        # 单token矩阵乘法FLOPS（假设为2*k*n*3）
        flop_per_token = 2 * k * n * 3

        # 设token数为T
        # acc_time = flop_per_token*T/acc_flops + weight_size/acc_bw
        # gpu_time = flop_per_token*T/gpu_flops + weight_size/gpu_bw + weight_size/acc_to_gpu_bandwidth
        # 求T使得gpu_time < acc_time
        # flop_per_token*T/gpu_flops + weight_size/gpu_bw + weight_size/acc_to_gpu_bandwidth < flop_per_token*T/acc_flops + weight_size/acc_bw
        # flop_per_token*T*(1/gpu_flops - 1/acc_flops) < weight_size/acc_bw - weight_size/gpu_bw - weight_size/acc_to_gpu_bandwidth
        # T < [weight_size/acc_bw - weight_size/gpu_bw - weight_size/acc_to_gpu_bandwidth] / [flop_per_token*(1/gpu_flops - 1/acc_flops)]
        denom = flop_per_token * (1/gpu_flops - 1/acc_flops)
        num = weight_size/acc_bw - weight_size/gpu_bw - weight_size/acc_to_gpu_bandwidth
        if denom <= 0:
            return None
        threshold = num / denom
        if threshold <= 0:
            return None
        return int(threshold)
    

    def expert_schedule_simulation(self, layer_idx=0):
        """
        专家调度模拟：
        1. 获取token阈值，遍历第layer_idx层所有专家，判断其计算位置（GPU/加速器），并记录相关id。
        2. 根据分配结果模拟GPU和加速器的执行时间。
        返回：
            gpu_expert_ids, acc_expert_ids, move_to_gpu_ids, fusion_acc_expert_ids, gpu_total_time, acc_total_time
        """
        # 获取token阈值
        token_threshold = self.get_expert_token_threshold()
        if token_threshold is None:
            raise RuntimeError("无法获取专家load到GPU的token阈值")

        # 获取第layer_idx层专家token融合统计和原始位置
        layer_key = f"model.layers.{layer_idx}.mlp"
        fusion_stats = self.expert_token_fusion_stats.get(layer_key, {}) if self.expert_token_fusion_stats else {}
        expert_locs = self.expert_location.get(layer_key, {}) if self.expert_location else {}

        gpu_expert_ids = []
        acc_expert_ids = []
        move_to_gpu_ids = []
        fusion_acc_expert_ids = []

        gpu_total_time = 0.0
        acc_total_time = 0.0

        # 获取硬件参数
        gpu_flops = self.devices['GPU'].peak_flops  # FLOPS
        acc_flops = self.devices['Acc'].peak_flops  # FLOPS
        gpu_bw = self.devices['GPU'].peak_memory_bandwidth
        acc_bw = self.devices['Acc'].peak_memory_bandwidth
        acc_to_gpu_bw = self.devices['GPU'].max_interface_bandwidth
        k = self.model.hdim
        n = self.model.hdim * self.model.ff_scale
        dtype_size = self.model.dtype
        weight_size = k * n * 3 * dtype_size

        # 记录每个专家实际处理的token数
        expert_actual_tokens = {}

        for expert_id, stat in fusion_stats.items():
            loc = expert_locs.get(expert_id, None)
            orig_token = stat.get('total_tokens', 0)
            fused_token = stat.get('tokens_after_merge', 0)
            use_fusion = fused_token > 0 and orig_token >= token_threshold

            # 判断专家是否在GPU上
            if loc == 'GPU':
                gpu_expert_ids.append(expert_id)
                # GPU执行时间
                flop = orig_token * 2 * k * n * 3
                time = flop / gpu_flops + weight_size / gpu_bw
                # 不需要搬运权重
                gpu_total_time += time
                expert_actual_tokens[expert_id] = orig_token
            else:
                # 判断是否在加速器上执行
                if (orig_token <= token_threshold) or (use_fusion and fused_token <= token_threshold):
                    acc_expert_ids.append(expert_id)
                    # 加速器执行时间
                    flop = (fused_token if use_fusion else orig_token) * 2 * k * n * 3
                    time = flop / acc_flops + weight_size / acc_bw
                    acc_total_time += time
                    if use_fusion:
                        fusion_acc_expert_ids.append(expert_id)
                    expert_actual_tokens[expert_id] = fused_token if use_fusion else orig_token
                else:
                    gpu_expert_ids.append(expert_id)
                    move_to_gpu_ids.append(expert_id)
                    # GPU执行时间（需要搬运权重）
                    flop = orig_token * 2 * k * n * 3
                    time = flop / gpu_flops + weight_size / gpu_bw + weight_size / acc_to_gpu_bw
                    gpu_total_time += time
                    expert_actual_tokens[expert_id] = orig_token

        # -------- 微调分配以最小化总延迟 --------
        def calc_time(gpu_ids, acc_ids):
            gpu_time = 0.0
            acc_time = 0.0
            for expert_id in gpu_ids:
                stat = fusion_stats.get(expert_id, {})
                orig_token = stat.get('total_tokens', 0)
                flop = orig_token * 2 * k * n * 3
                loc = expert_locs.get(expert_id, None)
                move_weight = (loc != 'GPU')
                t = flop / gpu_flops + weight_size / gpu_bw
                if move_weight:
                    t += weight_size / acc_to_gpu_bw
                gpu_time += t
            for expert_id in acc_ids:
                stat = fusion_stats.get(expert_id, {})
                orig_token = stat.get('total_tokens', 0)
                fused_token = stat.get('tokens_after_merge', 0)
                use_fusion = fused_token > 0 and fused_token != orig_token
                flop = (fused_token if use_fusion else orig_token) * 2 * k * n * 3
                t = flop / acc_flops + weight_size / acc_bw
                acc_time += t
            return gpu_time, acc_time

        # 记录初始分配
        best_gpu_ids = gpu_expert_ids.copy()
        best_acc_ids = acc_expert_ids.copy()
        best_total_latency = max(gpu_total_time, acc_total_time)
        improved = True
        while improved:
            improved = False
            gpu_time, acc_time = calc_time(gpu_expert_ids, acc_expert_ids)
            total_latency = max(gpu_time, acc_time)
            # 尝试从GPU移动到加速器
            if gpu_time > acc_time and len(gpu_expert_ids) > 0:
                # 找到GPU端融合后token数最少的专家
                min_fusion_id = None
                min_fusion_token = float('inf')
                for eid in gpu_expert_ids:
                    stat = fusion_stats.get(eid, {})
                    fused_token = stat.get('tokens_after_merge', stat.get('total_tokens', 0))
                    if fused_token < min_fusion_token:
                        min_fusion_token = fused_token
                        min_fusion_id = eid
                if min_fusion_id is not None:
                    # 尝试移动
                    gpu_expert_ids.remove(min_fusion_id)
                    acc_expert_ids.append(min_fusion_id)
                    new_gpu_time, new_acc_time = calc_time(gpu_expert_ids, acc_expert_ids)
                    new_total_latency = max(new_gpu_time, new_acc_time)
                    if new_total_latency < total_latency:
                        best_gpu_ids = gpu_expert_ids.copy()
                        best_acc_ids = acc_expert_ids.copy()
                        fusion_acc_expert_ids.append(min_fusion_id)
                        expert_actual_tokens[min_fusion_id] = fusion_stats.get(min_fusion_id, {}).get('tokens_after_merge', expert_actual_tokens.get(min_fusion_id, 0))
                        best_total_latency = new_total_latency
                        improved = True
                    else:
                        # 回退
                        acc_expert_ids.remove(min_fusion_id)
                        gpu_expert_ids.append(min_fusion_id)
            # 尝试从加速器移动到GPU
            elif acc_time > gpu_time and len(acc_expert_ids) > 0:
                max_fusion_id = None
                max_fusion_token = -1
                for eid in acc_expert_ids:
                    stat = fusion_stats.get(eid, {})
                    fused_token = stat.get('tokens_after_merge', stat.get('total_tokens', 0))
                    if fused_token > max_fusion_token:
                        max_fusion_token = fused_token
                        max_fusion_id = eid
                if max_fusion_id is not None:
                    # 尝试移动
                    acc_expert_ids.remove(max_fusion_id)
                    gpu_expert_ids.append(max_fusion_id)
                    new_gpu_time, new_acc_time = calc_time(gpu_expert_ids, acc_expert_ids)
                    new_total_latency = max(new_gpu_time, new_acc_time)
                    if new_total_latency < total_latency:
                        best_gpu_ids = gpu_expert_ids.copy()
                        best_acc_ids = acc_expert_ids.copy()
                        if max_fusion_id in fusion_acc_expert_ids:
                            fusion_acc_expert_ids.remove(max_fusion_id)
                            expert_actual_tokens[max_fusion_id] = fusion_stats.get(max_fusion_id, {}).get('total_tokens', expert_actual_tokens.get(max_fusion_id, 0))   
                        best_total_latency = new_total_latency
                        improved = True
                    else:
                        # 回退
                        gpu_expert_ids.remove(max_fusion_id)
                        acc_expert_ids.append(max_fusion_id)

        # 重新统计最终分配下的专家id和时间
        final_gpu_time, final_acc_time = calc_time(best_gpu_ids, best_acc_ids)
        move_to_gpu_ids = [eid for eid in best_gpu_ids if expert_locs.get(eid, None) != 'GPU']

        return {
            'gpu_expert_ids': best_gpu_ids,
            'acc_expert_ids': best_acc_ids,
            'move_to_gpu_ids': move_to_gpu_ids,
            'fusion_acc_expert_ids': fusion_acc_expert_ids,
            'gpu_total_time': final_gpu_time,
            'acc_total_time': final_acc_time,
            'total_latency': max(final_gpu_time, final_acc_time),
            'expert_actual_tokens': expert_actual_tokens
        }


    def get_required_mem_capacity(self, batch_size, lin, lout):
        ndec = self.model.ndec
        hdim = self.model.hdim
        nhead = self.model.num_heads
        ff_scale = self.model.ff_scale
        w_byte = 2 if self.model.dtype in [DataType.W16A16, DataType.W16A8
                                           ] else 1
        a_byte = 2 if self.model.dtype in [DataType.W16A16, DataType.W8A16
                                           ] else 1
        l = lin + lout - 1

        if 'LLAMA' in self.model.name or self.model.moe:
            weight_memory = ndec * hdim * (2 * hdim + 2 * (hdim) +
                                           3 * ff_scale * hdim) * w_byte
        else:
            weight_memory = ndec * hdim * (2 * hdim + 2 * (hdim) +
                                           2 * ff_scale * hdim) * w_byte

        temp_memory = max((hdim + l * nhead) * a_byte, hdim * 2 * a_byte,
                          l * nhead * 2 * a_byte,
                          (ff_scale * hdim + hdim) * a_byte) + l * nhead
        kv_memory = ndec * 2 * l * (hdim) * a_byte

        return weight_memory, kv_memory * batch_size, temp_memory * batch_size
