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

def read_txt(file_path):
    """读取txt文件内容并返回问题和答案"""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        q1, a1 = lines[0].strip().split('\t')
        q2, a2 = lines[1].strip().split('\t')
        return q1, a1, q2, a2

def txt_to_json(folder_path):
    """将文件夹下的txt文件转换为json文件"""
    data_list = []
    idx = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            jpg_path = os.path.join(folder_path, filename.replace('.txt', '.jpg'))
            png_path = os.path.join(folder_path, filename.replace('.txt', '.png'))
            image_name = ""
            if os.path.isfile(jpg_path):
                image_name = jpg_path
            elif os.path.isfile(png_path):
                image_name = png_path
            if image_name == "":
                continue
            q1, a1, q2, a2 = read_txt(file_path)
            data_entry = {
                "data_id": idx,
                "image_name": image_name,
                "question1": q1,
                "answer1": a1,
                "question2": q2,
                "answer2": a2,
                "cate": folder_path.split('/')[-1],
            }
            data_list.append(data_entry)
            idx += 1

    # 将列表写入JSON文件
    json_filename = os.path.join("simpleVQA/MMEbench/json_files", f"{folder_path.split('/')[-1]}.json")
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)


def main(in_path_json, out_path_json):
    '''
    Func:
        请求g4o获取输出
    '''
    pool = ThreadPoolExecutor(THREAD_NUM)
    futures = []

    out_json = []
    fout = open(out_path_json, 'w', encoding='utf-8')
    with open(in_path_json, 'r', encoding='utf-8') as f:
        lines = json.load(f) # 解析json文件
        for line in tqdm(lines):
            # print(line['question'])
            image_path = os.path.join("/mnt/personal-code/", line['image_name'])
            if image_path == "":
                continue

            question1 = line['question1']
            answer1 = line['answer1']
            question2 = line['question2']
            answer2 = line['answer2']
            prompt = f"""你是一个多模态领域的数据标注师，负责整理任务中的图片问答标注数据，用于优化一款多模态自动问答系统。本次的数据标注任务是给定图片、两个判断对错的问题和两个答案(包含一个“Yes”表示正确判断)，现在需要你来将问题和答案改写，按照给出的要求把判断型问答改写为针对特定对象或主体的一个询问型问答。注意不要改变问答的原本语言，答案不要出现在问题中。
            ## 问题1
            [{question1}]

            ## 答案1
            [{answer1}]

            ## 问题2
            [{question2}]

            ## 答案2
            [{answer2}]

            ## 改写要求
            1. 首先对比提供的2个问题，确定目标的提问句式并改写出一个问题，从答案为"Yes"的问题中抽取出目标问题的答案，答案不要出现在问题中；
            ## 举一个例子
            {{
                "question1": "Does this artwork belong to the type of religious? Please answer yes or no.",
                "answer1": "Yes",
                "question2": "Does this artwork belong to the type of landscape? Please answer yes or no.",
                "answer2": "No"
            }}
            ## 改写得到的问答
            {{
                "question": "What type does this artwork belong to?",
                "answer": "Religious"
            }}
            
            2. 判断改写后的「问题」是否有效，下面是几种无效问题的类型：
            - 问题必须是从原问题改写而来，是一个看图问答风格的提问；
            - 问题的语义不流畅，有明显语病。
            - 问题太简单，或者对图片的理解有问题以至于提出了不合理的问题。
            - 提出的问题即使看不到图片也能正确作答，导致图片信息没有价值。
            - 提问的问题基于现有图片无法回答。
            3. 然后判断选出的「答案」是否合理，下面是几种无效答案：
            - 答案不是从判定为"Yes"的原问题中抽取得来；
            - 答非所问，答案的内容和改写后问题所问的内容不匹配；
            - 答案空洞、没有意义，或者对图片的理解有问题以至于给出了不合理的答案；
            4. 如果改写后的「问题」和「答案」都有效，才是一条合格的数据。

            返回格式如下：
            {{
                "question": "不改变原问题的语种，内容不包含答案",
                "answer": "保持原语种，保证从原问题中抽取得到的、正确的那个"
                "qualified": "是否合格，不合格给出原因。"
            }}
            下面请严格按照以上格式生成回复。"""
            # print(query)
            data = {"query": prompt, "image_path":image_path, "use_g4o":True}
            # response = requests.post(api_url_list[0], json=data).json()
            response = pool.submit(get_case_refine, prompt, image_path)
            # futures.append([prompt, image_path, pool.submit(get_case_refine, prompt, image_path)])
            # print(response.result())
            try:
                res = response.result()['response'][0].replace("```json", "").replace("```", "")
                # res = response['response'][0].replace("```json", "").replace("```", "")
                json_res = json.loads(res)
            except Exception as e:
                res = "答案解析失败"
                json_res = {} # "答案解析失败"

            print(res)
            
            res_json = {
                "data_id": line['data_id'],
                "image_name": line['image_name'],
                "question": json_res['question'],
                "answer": json_res['answer'],
                "cate": line['cate'],
                "qualified": json_res['qualified']
            }
            
            # res_json["g4o_response"] = res
            out_json.append(res_json)
            print("--"*20)

    # 将 Python 对象保存到 JSON 文件
    json.dump(out_json, fout, ensure_ascii=False, indent=4)
    # fout.write("\n")
    fout.close()
    return

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    ####################  需要校验下面这几个参数  ###################
    commen_path = "simpleVQA/MMEbench/json_files/"
    for file_name in os.listdir(commen_path):
        if 'refine' in file_name:
            continue
        if 'artwork' in file_name:
            continue
        if os.path.isfile('refine_'+file_name):
            continue
        in_path_json = os.path.join(commen_path, file_name)
        out_path_json = os.path.join(commen_path, 'refine_'+file_name)
        main(in_path_json, out_path_json)
    # in_path_json = "simpleVQA/MMEbench/json_files/artwork.json"
    # out_path_json = "simpleVQA/MMEbench/json_files/artwork_refine.json"

    # temperature_list = [0.7, 0.9, 1.0, 1.2]
    # top_p_list = [0.8, 1.0]  # top_p 值范围在[0,1]之间

    # main(in_path_json, out_path_json)
    
    # commen_path = "simpleVQA/MMEbench/"
    # for dir_name in os.listdir(commen_path):
    #     folder_path = os.path.join(commen_path, dir_name)
    #     txt_to_json(folder_path)