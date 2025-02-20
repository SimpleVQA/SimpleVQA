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

sk = ""
url = ""
RETRY_TIMES = 8 # 如果请求失败了，尝试的次数
THREAD_NUM = 4
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
                    "qwen-vl-max", # "doubao-vision-pro-32k", # "gpt-4o", # 
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
            responose = "error"
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

def get_refine_response(image_path, query):
    response = []
    for i in range(ANSWER_NUM):
        res = get_response(image_path, query)
        response.append(res)
    return response


def get_case_refine(query, image_path):
    tcase = {}
    try:
        response = get_refine_response(image_path, query)
        tcase["query"] = query
        tcase["response"] = response
        tcase["image_path"] = image_path
    except:
        print("ERROR  line:{}".format( traceback.format_exc()))
    return tcase

def main(in_path_json, out_path_json):
    '''
    Func:
        请求g4o获取输出
    '''
    pool = ThreadPoolExecutor(THREAD_NUM)
    futures = []

    out_json = []
    fout = open(out_path_json, 'a', encoding='utf-8')
    with open(in_path_json, 'r', encoding='utf-8') as f:
        lines = json.load(f) # 解析json文件
        for line in tqdm(lines):
            # print(line['question'])
            # if line["doubao_response"] != "访问异常，需重试":
            #     out_json.append(line)
            #     continue
            image_path = os.path.join("/mnt/", line['image'])
            if image_path == "":
                continue
            # if len(line[line['answer']])>25:
            #     continue
            res_json = line # {"query": line["query"], "image_path": line['image_path']}
            prompt = f"""If you are a computer vision expert, please understand the picture and answer the question in English. The answer must be a short named entity:
            ## Now let's ask the question:
            {line["question"]}
            """
            # print(query)
            # data = {"query": prompt, "image_path":image_path, "use_g4o":True}
            # response = requests.post(api_url_list[0], json=data).json()
            response = pool.submit(get_case_refine, prompt, image_path)
            # futures.append([prompt, image_path, pool.submit(get_case_refine, prompt, image_path)])
            print(response.result())
            try:
                res = response.result()['response'][0]
                # res = response['response'][0].replace("```json", "").replace("```", "")
            except Exception as e:
                res = "答案解析失败"

            # res = response['response'][0]
            # print(res)
            # res_json["answer"] = line[line['answer']]
            res_json["qwen_response"] = res
            out_json.append(res_json)
            print("--"*10)

    # 将 Python 对象保存到 JSON 文件
    json.dump(out_json, fout, ensure_ascii=False, indent=4)
    # fout.write("\n")
    fout.close()
    return

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    ####################  需要校验下面这几个参数  ###################
    in_path_json = "simpleVQA/difficulty_simplevqa_mme_req_g4o_doubao.json"
    out_path_json = "simpleVQA/difficulty_simplevqa_mme_req_g4o_doubao_qwen.json"

    # temperature_list = [0.7, 0.9, 1.0, 1.2]
    # top_p_list = [0.8, 1.0]  # top_p 值范围在[0,1]之间

    main(in_path_json, out_path_json)