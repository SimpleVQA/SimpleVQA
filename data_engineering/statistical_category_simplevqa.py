import os
from tqdm import tqdm
import json
import time
import traceback
import random

input_json_list = [
    'simpleVQA/SimpleVQA_CN_qwen72B_janus7B_qw2572B_recheck_clip2.json',
    'simpleVQA/SimpleVQA_EN_add_atomicQApp_classified.json'
]
out_path_json = 'simpleVQA/SimpleVQA_category_simplevqa.json'

task_category = {}
subject_category = {}
entity_class = {}

fout = open(out_path_json, 'w', encoding='utf-8')
for in_path_json in input_json_list:
    with open(in_path_json, 'r', encoding='utf-8') as f:
        lines = json.load(f) # 解析json文件
        for line in tqdm(lines):
            if line['vqa_category'] == {}: continue
            if line['vqa_category']['task_category'] not in task_category:
                task_category[line['vqa_category']['task_category']] = 1
            else:
                task_category[line['vqa_category']['task_category']] += 1

            if line['vqa_category']['subject_category'] not in subject_category:
                subject_category[line['vqa_category']['subject_category']] = 1
            else:
                subject_category[line['vqa_category']['subject_category']] += 1

            if line['vqa_category']['entity_class'] not in entity_class:
                entity_class[line['vqa_category']['entity_class']] = 1
            else:
                entity_class[line['vqa_category']['entity_class']] += 1

res_json = [task_category, subject_category, entity_class]            
json.dump(res_json, fout, ensure_ascii=False, indent=4)
print(res_json)

fout.close()