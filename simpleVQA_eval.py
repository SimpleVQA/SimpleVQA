import random 
import re
import pandas

import json
import copy

model_keys = ["g4o_response"]
mapper = {"正确": "is_correct", "错误": "is_incorrect", "未尝试": "is_not_attempted"}

def divide_dict_values(d, divisor):
    for key, value in d.items():
        if isinstance(value, dict):
            divide_dict_values(value, divisor)
        else:
            d[key] = value / divisor
    return d


def SimpleVQAEval(in_file_path, out_file_path):
    amount = 0

    with open(in_file_path, 'r', encoding='utf-8') as file:
        lines = "".join(file.readlines())
        data = json.loads(lines)
        amount = len(data)
        result_metrics = {}
        
        for idx, line in enumerate(data):
            judge_res = line["judge_res"].copy()
            # print(idx, line["judge_res"])
            for key, value in judge_res.items(): ## key表示模型的response
                # if value["conclusion"] not in ["正确", "错误", "未尝试"]: ## 拒答==未尝试
                #     value["conclusion"] = value["conclusion"].replace("**", "")
                if isinstance(value, dict):
                    judge_res[key] = value["conclusion"].replace("**", "")
                elif judge_res[key] not in ["正确", "错误", "未尝试"]:
                    judge_res[key] = judge_res[key].replace("**", "")
                    if len(judge_res[key]) > 3:
                        judge_res[key] = "未尝试"
                    
                if key not in result_metrics:
                    result_metrics[key] = {
                        "is_correct": 0,
                        "is_incorrect": 0,
                        "is_not_attempted": 0
                    }
                # result_metrics[key][mapper[value["conclusion"]]] += 1 ## 按模型的回答情况计数 
                result_metrics[key][mapper[judge_res[key]]] += 1
                if key not in model_keys:
                    model_keys.append(key)
    print("ALL data count: ", amount)
    res_json = []
    # Aggregate metrics
    for model_key in model_keys:
        aggregate_metrics = {
            "LVLM_name": model_key,
            "is_correct": round(result_metrics[model_key]["is_correct"] / amount, 4),
            "is_incorrect": round(result_metrics[model_key]["is_incorrect"] / amount, 4),
            "is_not_attempted": round(result_metrics[model_key]["is_not_attempted"] / amount, 4),
        }
        aggregate_metrics["is_given_attempted"] = round(aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"], 4)
        # Calculate accuracy_given_attempted
        aggregate_metrics["accuracy_given_attempted"] = (
            aggregate_metrics["is_correct"]
            / aggregate_metrics["is_given_attempted"]
            if aggregate_metrics["is_given_attempted"] > 0
            else 0
        )
        aggregate_metrics["f1"] = (
                2 * aggregate_metrics["accuracy_given_attempted"] * aggregate_metrics["is_correct"]
                / (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"])
                if (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"]) > 0
                else 0
            )
        print(model_key + ": AGGREGATE METRICS") 
        print(aggregate_metrics)
        print("##################")
        
        res_json.append(aggregate_metrics)
        
        output_d = {
            "accuracy_given_attempted": aggregate_metrics["accuracy_given_attempted"],
            "f1": aggregate_metrics["f1"]
        }
        
        print(f"Accuracy Given Attempted: {output_d['accuracy_given_attempted']:.3f}")
        print(f"F1 Score: {output_d['f1']:.3f}")
        
    with open(out_file_path, "a", encoding="utf-8") as fout:
        json.dump(res_json, fout, ensure_ascii=False, indent=4)
    # print(divide_dict_values(copy.deepcopy(aggregate_metrics), amount))

if __name__ == "__main__":
    file_path = "simpleVQA/SimpleVQA_CN_qwen72B_janus7B_qw2572B_recheck_clip2_judge1.json"
    out_path_json = "simpleVQA/SimpleVQA_CN_qwen72B_janus7B_qw2572B_recheck_clip2_judge_metrics.json"
    
    SimpleVQAEval(file_path, out_path_json)
