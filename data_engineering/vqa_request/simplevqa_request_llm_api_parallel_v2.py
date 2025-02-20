"""
多进程跑LLM的response
"""
import os
import random
import requests
import json
import time
import base64
import datetime
import traceback
import collections
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import nltk
import pandas as pd
from tqdm import tqdm

from gpt import gpt_35_turbo_call


PROMPT_TEMPLATE_CN = """请阅读以下问题：
{question}
请基于此问题提供你的最佳答案，要求只能回答一个命名实体；并用0到100的分数表示你对该答案的信心（置信度）。请以如下的JSON格式给出回复：
{{
    "answer": "你的答案",
    "confidence_score": "你的置信度",
}}
"""
PROMPT_TEMPLATE_EN = """Please read the following questions:
{question}
Please provide your best answer based on this question, requesting that only one named entity be answered, and use a score of 0 to 100 to indicate your confidence in the answer (confidence). Please reply in the following JSON format:
{{
    "answer": "your answer",
    "confidence_score": "your_confidence",
}}
"""
RETRY_TIMES = 16 # 如果请求失败了，尝试的次数
THREAD_NUM = 2
PROCESS_NUM = 4
ANSWER_NUM = 1 # 需要几个答案


def image_to_base64(image_path):
    # print(image_path)
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string


def get_response(image_path, prompt):
    res = ""
    for i in range(RETRY_TIMES):
        try:
            base64_image = image_to_base64(image_path)
            # 需要注意 content 里面是先输入图片，再输出text。也就是保持输入的顺序
            block = {
                "model":
                    "gpt-4o",
                "messages": [{
                    "role":
                        "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }]
                }]
            }
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {sk}'}
            # print("block = ",block)
            response = requests.post(url, headers=headers, json=block).json()
            # print("repsonse ==== 11111 = ",response)
            res = response["choices"][0]["message"]["content"]
            break # 如果访问成功了，就不用再访问了！！！这个地方很重要！
        except Exception:
            res = "访问异常，需重试"
            print("访问异常，重试中...")
            # print("ERROR image_path:{} \n error:{} input:{}".format(image_path, traceback.format_exc(), prompt))
    if res == "":
        print("无返回结果，请注意！！！")
    return res


def get_vqa_response(image_path, query):
    response = []
    for i in range(ANSWER_NUM):
        res = get_response(image_path, query)
        response.append(res)
    return response


def get_case_vqa(query, image_path):
    tcase = {}
    try:
        response = get_vqa_response(image_path, query)
        tcase["query"] = query
        tcase["response"] = response
        tcase["image_path"] = image_path
    except:
        print("ERROR  line:{}".format( traceback.format_exc()))
    return tcase


def openai_call(model_name, prompt):
    response = ''
    if model_name == 'gpt35-turbo':
        response = gpt_35_turbo_call(prompt=prompt)
    else:
        raise NotImplementedError()
    return response


def api_call(model_name, prompt):
    res = ''
    if model_name in ['gpt4o-0806', 'gpt35-turbo']:
        res =  openai_call(model_name, prompt)
    else:
        raise NotImplementedError()
    return res


def main(in_path_json, out_path_json, model_name, rollback=False):
    '''
    多进程请求api模型
    '''
    out_json = []
    post_prompt = []
    result_list = []

    with open(in_path_json, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = json.loads(line)
            if line["data_id"] < 1012: ## 中文
                prompt = PROMPT_TEMPLATE_CN.format(**{'question': line['question']})
            else:  ## 英文
                prompt = PROMPT_TEMPLATE_EN.format(**{'question': line['question']})
            # info = (line['data_id'], prompt)
            if rollback:
                if "答案解析失败" == line["{}_response".format(model_name)]:
                    post_prompt.append([line, prompt])
                else:
                    fout.write(json.dumps(line, ensure_ascii=False) + '\n')
            else:    
                post_prompt.append([line, prompt])
            # post_prompt.append(info)
            # if idx > 10:
            #     break

    
    with ProcessPoolExecutor(max_workers=PROCESS_NUM) as executor:
        future_to_task = {executor.submit(api_call, model_name, item[1]) : item[0] for item in post_prompt}

        for future in as_completed(future_to_task):
            data_id = future_to_task[future]  # 获取对应的任务值
            try:
                result = future.result()
                res = {'data_id': data_id, 'response': result}
                result_list.append(res)
                print(f"Task {data_id} completed with result {result}")
            except Exception as e:
                print(f"Task {data_id} generated an exception: {e}")
   

    with open(out_path_json, 'w', encoding='utf-8') as fout:
        for res_dict in result_list:
            data_id = res_dict['data_id'] # 原来的json line
            res = res_dict['response']
            try:
                res = res['predict']
                res = res.replace("```json","").replace("```python","").replace("```","").strip()
                res = eval(res)
                # res['data_id'] = data_id
                data_id["{}_response".format(model_name)] = res['answer']
                data_id["{}_confidence_score".format(model_name)] = res['confidence_score']
            except Exception as e:
                print(f'error parse {res_dict} with error {e}')
                data_id["{}_response".format(model_name)] = "答案解析失败"
                data_id["{}_confidence_score".format(model_name)] = -1
                # res = {"answer": "答案解析失败", "confidence_score": -1, "response": res_dict}

            fout.write(json.dumps(data_id, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # 这里填所有要跑的模型
    model_list = ['gpt4o-0806', 'gpt35-turbo']

    # 这里填输入输出
    in_path_json = "/Users/ccc/github/simpleVQA/SimpleVQA_final_atomicQA_add_2025_without_reclassified.jsonl"
    model_name = "gpt35-turbo"
    out_path_json = f"/Users/ccc/github/simpleVQA/tmp_output/{model_name}_res.jsonl"
    with_confidence = True
    rollback = False
    assert model_name in model_list, f'error model name {model_name}, out of model list {model_list}'

    # 这里开始跑
    main(in_path_json, out_path_json, model_name, rollback)
