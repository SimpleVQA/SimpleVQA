import json
import copy

file_path = "simpleVQA/judge_results_mme_for_simplevqa_qwen_g4opp_doubao_eb_response.json"
out_path_json = "difficulty_simplevqa_mme_judge_by_doubao.json"
model_keys = ["g4o_response", "qwen_response", "eb_response", "doubao_response"]

all_socre = {
    "g4o_response":{
        "right":0,
        "wrong":0,
        "noanswer":0
    },
    "qwen_response":{
        "right":0,
        "wrong":0,
        "noanswer":0
    },
    "eb_response":{
        "right":0,
        "wrong":0,
        "noanswer":0
    },
    "doubao_response":{
        "right":0,
        "wrong":0,
        "noanswer":0
    }
}

pure_socre = {
    "g4o_response":{
        "right":0,
        "wrong":0,
        "noanswer":0
    },
    "qwen_response":{
        "right":0,
        "wrong":0,
        "noanswer":0
    },
    "eb_response":{
        "right":0,
        "wrong":0,
        "noanswer":0
    },
    "doubao_response":{
        "right":0,
        "wrong":0,
        "noanswer":0
    }
}

mapper = {"正确":"right", "错误":"wrong", "拒答":"noanswer"}

def divide_dict_values(d, divisor):
    for key, value in d.items():
        if isinstance(value, dict):
            divide_dict_values(value, divisor)
        else:
            d[key] = value / divisor
    return d

all_right_questions = []
difficulty_cases = []
amount = 0

with open(file_path, 'r', encoding='utf-8') as file:
    lines = "".join(file.readlines())
    data = json.loads(lines)
    amount = len(data)
    for idx, line in enumerate(data):
        flag = 0
        try:
            right_num = [1 for model in model_keys if line["judge_res"][model]["conclusion"] == "正确"]
        except:
            print(line['data_id'])
        if sum(right_num) == len(model_keys):
            all_right_questions.append((line["question"], line["data_id"]))
            flag = 1
        else:
            # difficult_case = {
            #     "data_id": line['data_id'],
            #     "question": line['question'],
            #     "answer": line['answer'],
            #     "category": line['category'],
            #     "image": line['image'],
            #     "l2-category": line['l2-category'],
            #     "qwen_response": line['qwen_response'],
            #     "g4o_response": line['g4o_response'],
            #     "doubao_response": line['doubao_response'],
            #     "eb_response": line['eb_response'],
            # }
            difficult_case = {
                "data_id": line['data_id'],
                "question": line['question'],
                "answer": line['answer'],
                "cate": line['cate'],
                "image": line['image'],
                "qwen_response": line['qwen_response'],
                "g4o_response": line['g4o_response'],
                "doubao_response": line['doubao_response'],
                "eb_response": line['eb_response'],
            }
            difficulty_cases.append(difficult_case)

        for key, value in line["judge_res"].items():
            if value["conclusion"] not in ["正确", "错误", "拒答"]:
                value["conclusion"] = value["conclusion"].replace("**", "")
            all_socre[key][mapper[value["conclusion"]]] += 1
            if flag == 0:
                pure_socre[key][mapper[value["conclusion"]]] += 1

print(all_socre)
print(pure_socre)

# print(all_right_questions)

print(divide_dict_values(copy.deepcopy(all_socre), amount))
print(divide_dict_values(copy.deepcopy(pure_socre), amount-len(all_right_questions)))

with open(out_path_json, "w", encoding="utf-8") as fout:
    json.dump(difficulty_cases, fout, ensure_ascii=False, indent=4)