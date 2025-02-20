import os
import nltk
from utils import get_query2image_path, image_to_base64, request_by_query_and_image_path
from editdistance import eval  
from concurrent.futures import ThreadPoolExecutor
import random
from tqdm import tqdm
import requests
import json
import time
import pandas as pd
import base64
import datetime
import traceback
import collections

api_url_list = ["",
                "",
                ""]
RETRY_TIMES = 16 # 如果请求失败了，尝试的次数
THREAD_NUM = 8
ANSWER_NUM = 1 # 需要几个答案

def image_to_base64(image_path):
    # print(image_path)
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

def post_multi_times(api_url, image_path, query, temperature = 0.8, top_p = 0.8, times = 1):
    '''
    func: 发送多次请求，从而比较哪个结果是最好的
    '''
    answers = []
    val = []
    for j in range(times):
        # print(f"query = {query}...")
        answer = request_by_query_and_image_path(api_url, image_path, query)
        answer = answer['result']['response']['utterance']
        # answer = answer.replace("\n","")
        answers.append(answer)
    return answers


def main(in_path_json, out_path_json):
    '''
    Func:
        请求g4o获取输出
    '''
    pool = ThreadPoolExecutor(THREAD_NUM)
    futures = []

    cnt = 0
    out_json = []
    fout = open(out_path_json, 'a', encoding='utf-8')
    with open(in_path_json, 'r', encoding='utf-8') as f:
        lines = json.load(f) # 解析json文件
        for line in tqdm(lines):
            # if line["g4o_response"] != "访问异常，需重试":
            #     out_json.append(line)
            #     continue
            # print(line['question'])
            image_path = os.path.join("/mnt/personal-code/simpleVQA/images/CCBench", line['image'])
            if image_path == "":
                continue

            res_json = line # {"query": line["query"], "image_path": line['image_path']}
            prompt = f"""请理解图片内容，然后回答下面的问题，要求仅回答一个命名实体即可：
            ## 问题: 
            {line["question"]}
            """
            # print(query)
            # data = {"query": prompt, "image_path":image_path, "use_g4o":True}
            # response = requests.post(api_url_list[0], json=data).json()
            # response = pool.submit(get_case_refine, prompt, image_path)
            response = pool.submit(post_multi_times, api_url_list[0], image_path, prompt, 1)
            cnt += 1
            # futures.append([prompt, image_path, pool.submit(get_case_refine, prompt, image_path)])
            print(response.result())
            try:
                res = response.result()[0]
                # res = response['response'][0].replace("```json", "").replace("```", "")
            except Exception as e:
                res = "答案解析失败"

            # res = response['response'][0]
            # print(res)
            res_json["eb_response"] = res
            out_json.append(res_json)
            print("--"*10)

    # 将 Python 对象保存到 JSON 文件
    json.dump(out_json, fout, ensure_ascii=False, indent=4)
    # fout.write("\n")
    fout.close()
    return

if __name__ == "__main__":

    ####################  需要校验下面这几个参数  ###################
    in_path_json = "/mnt/personal-code/simpleVQA/CCBench_for_simplevqa_qwen_g4opppppp_response.json"
    out_path_json = "/mnt/personal-code/simpleVQA/CCBench_for_simplevqa_qwen_g4opppppp_eb_response.json"


    main(in_path_json, out_path_json)
