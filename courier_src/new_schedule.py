def expert_schedule_simulation(self, layer_idx=0):
        """
        专家调度模拟（优化版）：
        预计算每个专家在GPU/加速器上的执行时间，使用增量更新进行负载均衡优化。
        返回：gpu_expert_ids, acc_expert_ids, move_to_gpu_ids, fusion_acc_expert_ids, gpu_total_time, acc_total_time
        """
        token_threshold = self.get_expert_token_threshold()
        if token_threshold is None:
            raise RuntimeError("无法获取专家load到GPU的token阈值")

        # 获取配置数据
        layer_key = f"model.layers.{layer_idx}.mlp"
        fusion_stats = self.expert_token_fusion_stats.get(layer_key, {}) if self.expert_token_fusion_stats else {}
        expert_locs = set(self.expert_location.get(layer_key, []) if self.expert_location else [])

        # 预计算硬件参数
        gpu = self.devices['GPU']
        acc = self.devices['Acc']
        gpu_flops = gpu.peak_flops * gpu.num_xpu
        acc_flops = acc.peak_flops * acc.num_attacc
        gpu_bw = gpu.peak_memory_bandwidth * gpu.num_xpu
        acc_bw = acc.peak_memory_bandwidth * acc.num_attacc
        acc_to_gpu_bw = gpu.max_interface_bandwidth
        
        k, n = self.model.hdim, self.model.hdim * self.model.ff_scale
        dtype_size = 2 if self.model.dtype in [DataType.W16A16, DataType.W16A8] else 1
        weight_size = k * n * 3 * dtype_size
        flop_factor = 2 * k * n * 3
        
        # 预计算常量时间项
        weight_load_gpu = weight_size / gpu_bw
        weight_load_acc = weight_size / acc_bw
        weight_transfer = weight_size / acc_to_gpu_bw

        # 预计算每个专家的时间信息
        expert_info = {}  # {eid: {orig_tokens, fused_tokens, gpu_time, acc_time, on_gpu, use_fusion}}
        
        for key, stat in fusion_stats.items():
            eid = int(key[7:])  # 解析 "expert_X"
            orig = stat.get('total_tokens', 0)
            fused = stat.get('tokens_after_merge', 0)
            on_gpu = eid in expert_locs
            use_fusion = fused > 0 and orig >= token_threshold
            
            # GPU执行时间（始终用原始token数）
            gpu_time = (orig * flop_factor) / gpu_flops + weight_load_gpu
            if not on_gpu:
                gpu_time += weight_transfer
            
            # 加速器执行时间（可用融合token数）
            acc_tokens = fused if use_fusion else orig
            acc_time = (acc_tokens * flop_factor) / acc_flops + weight_load_acc
            
            expert_info[eid] = {
                'orig': orig, 'fused': fused, 'acc_tokens': acc_tokens,
                'gpu_time': gpu_time, 'acc_time': acc_time,
                'on_gpu': on_gpu, 'use_fusion': use_fusion
            }

        # 初始分配：根据阈值决定专家位置
        gpu_set, acc_set = set(), set()
        gpu_time_total, acc_time_total = 0.0, 0.0
        
        for eid, info in expert_info.items():
            if info['on_gpu']:
                # 已在GPU上的专家保持在GPU
                gpu_set.add(eid)
                gpu_time_total += info['gpu_time']
            elif info['orig'] <= token_threshold or (info['use_fusion'] and info['acc_tokens'] <= token_threshold):
                # token少的专家放加速器
                acc_set.add(eid)
                acc_time_total += info['acc_time']
            else:
                # token多的专家搬到GPU
                gpu_set.add(eid)
                gpu_time_total += info['gpu_time']

        # 贪心优化：通过移动专家平衡GPU和加速器负载
        # 可移动的专家：不在GPU原始位置上的专家
        movable = {eid for eid in expert_info if not expert_info[eid]['on_gpu']}
        
        improved = True
        while improved:
            improved = False
            
            if gpu_time_total > acc_time_total:
                # GPU负载重，尝试将专家移到加速器
                # 选择移动后收益最大的专家（GPU时间减少 - 加速器时间增加 最大）
                best_eid, best_gain = None, 0
                for eid in gpu_set & movable:
                    info = expert_info[eid]
                    gain = info['gpu_time'] - info['acc_time']
                    if gain > best_gain:
                        best_gain, best_eid = gain, eid
                
                if best_eid is not None:
                    new_gpu = gpu_time_total - expert_info[best_eid]['gpu_time']
                    new_acc = acc_time_total + expert_info[best_eid]['acc_time']
                    if max(new_gpu, new_acc) < max(gpu_time_total, acc_time_total):
                        gpu_set.remove(best_eid)
                        acc_set.add(best_eid)
                        gpu_time_total, acc_time_total = new_gpu, new_acc
                        improved = True
            
            elif acc_time_total > gpu_time_total:
                # 加速器负载重，尝试将专家移到GPU
                best_eid, best_gain = None, 0
                for eid in acc_set & movable:
                    info = expert_info[eid]
                    gain = info['acc_time'] - info['gpu_time']
                    if gain > best_gain:
                        best_gain, best_eid = gain, eid
                
                if best_eid is not None:
                    new_gpu = gpu_time_total + expert_info[best_eid]['gpu_time']
                    new_acc = acc_time_total - expert_info[best_eid]['acc_time']
                    if max(new_gpu, new_acc) < max(gpu_time_total, acc_time_total):
                        acc_set.remove(best_eid)
                        gpu_set.add(best_eid)
                        gpu_time_total, acc_time_total = new_gpu, new_acc
                        improved = True

        # 构建返回结果
        gpu_expert_ids = sorted(gpu_set)
        acc_expert_ids = sorted(acc_set)
        move_to_gpu_ids = [eid for eid in gpu_expert_ids if eid not in expert_locs]
        fusion_acc_expert_ids = [eid for eid in acc_expert_ids if expert_info[eid]['use_fusion']]
        expert_actual_tokens = {
            eid: info['acc_tokens'] if eid in acc_set else info['orig']
            for eid, info in expert_info.items()
        }

        return {
            'gpu_expert_ids': gpu_expert_ids,
            'acc_expert_ids': acc_expert_ids,
            'move_to_gpu_ids': move_to_gpu_ids,
            'fusion_acc_expert_ids': fusion_acc_expert_ids,
            'gpu_total_time': gpu_time_total,
            'acc_total_time': acc_time_total,
            'total_latency': max(gpu_time_total, acc_time_total),
            'expert_actual_tokens': expert_actual_tokens
        }


    def expert_schedule_simulation_naive(self, layer_idx=0):
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
        expert_locs = self.expert_location.get(layer_key, []) if self.expert_location else []

        gpu_expert_ids = []
        acc_expert_ids = []
        move_to_gpu_ids = []
        fusion_acc_expert_ids = []

        gpu_total_time = 0.0
        acc_total_time = 0.0

        # 获取硬件参数
        gpu_flops = self.devices['GPU'].peak_flops * self.devices['GPU'].num_xpu  # FLOPS
        acc_flops = self.devices['Acc'].peak_flops * self.devices['Acc'].num_attacc  # FLOPS
        gpu_bw = self.devices['GPU'].peak_memory_bandwidth * self.devices['GPU'].num_xpu
        acc_bw = self.devices['Acc'].peak_memory_bandwidth * self.devices['Acc'].num_attacc
        acc_to_gpu_bw = self.devices['GPU'].max_interface_bandwidth
        k = self.model.hdim
        n = self.model.hdim * self.model.ff_scale
        dtype_size = 2 if self.model.dtype in [DataType.W16A16, DataType.W16A8] else 1
        weight_size = k * n * 3 * dtype_size

        # 记录每个专家实际处理的token数
        expert_actual_tokens = {}

        for expert_id, stat in fusion_stats.items():
            expert_id = int(expert_id[7:])
            orig_token = stat.get('total_tokens', 0)
            fused_token = stat.get('tokens_after_merge', 0)
            use_fusion = fused_token > 0 and orig_token >= token_threshold

            # 判断专家是否在GPU上
            if expert_id in expert_locs:
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
                stat = fusion_stats.get('expert_' + str(expert_id), {})
                orig_token = stat.get('total_tokens', 0)
                flop = orig_token * 2 * k * n * 3
                move_weight = (expert_id not in expert_locs)
                t = flop / gpu_flops + weight_size / gpu_bw
                if move_weight:
                    t += weight_size / acc_to_gpu_bw
                gpu_time += t
            for expert_id in acc_ids:
                stat = fusion_stats.get('expert_' + str(expert_id), {})
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
                    stat = fusion_stats.get('expert_' + str(eid), {})
                    fused_token = stat.get('tokens_after_merge', stat.get('total_tokens', 0))
                    if fused_token < min_fusion_token and eid not in expert_locs:
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
                        expert_actual_tokens[min_fusion_id] = min_fusion_token
                        best_total_latency = new_total_latency
                        improved = True
                    else:
                        # 回退
                        acc_expert_ids.remove(min_fusion_id)
                        gpu_expert_ids.append(min_fusion_id)
            # 尝试从加速器移动到GPU
            elif acc_time > gpu_time and len(acc_expert_ids) > 0:
                max_actual_id = None
                max_actual_token = -1
                for eid in acc_expert_ids:
                    actual_token = expert_actual_tokens[eid]
                    if actual_token > max_actual_token:
                        max_actual_token = actual_token
                        max_actual_id = eid
                if max_actual_id is not None:
                    # 尝试移动
                    acc_expert_ids.remove(max_actual_id)
                    gpu_expert_ids.append(max_actual_id)
                    new_gpu_time, new_acc_time = calc_time(gpu_expert_ids, acc_expert_ids)
                    new_total_latency = max(new_gpu_time, new_acc_time)
                    if new_total_latency < total_latency:
                        best_gpu_ids = gpu_expert_ids.copy()
                        best_acc_ids = acc_expert_ids.copy()
                        if max_actual_id in fusion_acc_expert_ids:
                            fusion_acc_expert_ids.remove(max_actual_id)
                            expert_actual_tokens[max_actual_id] = fusion_stats.get('expert_' + str(max_actual_id), {}).get('total_tokens')
                        best_total_latency = new_total_latency
                        improved = True
                    else:
                        # 回退
                        gpu_expert_ids.remove(max_actual_id)
                        acc_expert_ids.append(max_actual_id)

        # 重新统计最终分配下的专家id和时间
        final_gpu_time, final_acc_time = calc_time(best_gpu_ids, best_acc_ids)
        move_to_gpu_ids = [eid for eid in best_gpu_ids if eid not in expert_locs]

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