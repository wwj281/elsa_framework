import json
import math

def extract_top_25_percent_experts(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    result = {}

    for layer_name, experts in data.items():
        # 按 gate_weight_sum 从大到小排序
        experts_sorted = sorted(
            experts,
            key=lambda x: x["gate_weight_sum"],
            reverse=True
        )

        total_experts = len(experts_sorted)
        top_k = math.ceil(total_experts * 0.25)

        top_expert_ids = [
            e["expert_id"] for e in experts_sorted[:top_k]
        ]

        result[layer_name] = {
            "total_experts": total_experts,
            "top_25_percent_count": top_k,
            "expert_ids": top_expert_ids,
        }

    return result


json_path = "gate_weight_data/input1024_batch4/qwen_3_30b/expert_gate_sum_t0.22_r1.00.json"
top_experts = extract_top_25_percent_experts(json_path)

for layer, info in top_experts.items():
    print(layer)
    print("Top 25% expert ids:", info["expert_ids"])
    break
