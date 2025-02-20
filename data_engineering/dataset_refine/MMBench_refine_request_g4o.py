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

api_url_list = [""]

sk = ""
url = ""
RETRY_TIMES = 16 # 如果请求失败了，尝试的次数
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
                    "qwen-vl-max", # "gpt-4o",
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
    cnt = 0
    simple_query = []
    special_query = []

    out_json = []
    fout = open(out_path_json, 'a', encoding='utf-8')

    with open(in_path_json, 'r', encoding='utf-8') as f:
        lines = json.load(f) # 解析json文件
        for line in tqdm(lines):
            # print(line['question'])
            image_path = os.path.join("/mnt/personal-code/simpleVQA/images/MMBench_V11/", line['image'])
            if image_path == "":
                continue
            if len(line['hint']) < 20:
                simple_query.append(line)
                cnt += 1
                continue
                # prompt = f"""请理解图中内容，然后回答下面的问题，要求答案尽可能简单明了且专业，不超过35个字符：
                # # 问题: {line["question"]}
                # """
            if "图" == line['hint'][0]:
                Hint = line['hint'].split('\n')[0]
            elif "阅读文本" == line['hint'][:4] or line['hint'].split('\n')[-1] == "":
                Hint = line['hint']
            else:
                Hint = line['hint'].split('\n')[-1]
            print(Hint)
            # continue

            query = line['question']
            if "完成文本" in query:
                special_query.append(line)

            prompt = f"""你是一个多模态领域的数据标注师，负责整理图片问答任务的标注数据，用于优化一款多模态自动问答系统。本次的数据标注任务是给定图片、一个关于图片内容或任务的描述(Hint)，现在需要你根据提供的信息生成一组新的问题(question)和答案(answer)，问题和答案严格遵循下面给出的要求。注意不要改变问答的原本语言，答案不要出现在问题中。

            ## 描述(Hint)
            [{Hint}]

            ## 图片(已上传)
            [<image>]

            ## 对于问题和答案的要求
            1. 首先理解并结合提供的Hint和图片信息，生成一个问题，从Hint有关图片内容描述的内容中抽取或推理出一个能正确回答问题的答案，答案不要出现在问题中；
            ## 第1个例子(假如给定了图片)
            {{
            "Hint": "图：准备混凝土坍落度试验。",
            }}
            ## 生成的问答
            {{
                "question": "图中两人正在进行什么试验?",
                "answer": "混凝土坍落度试验"
            }}
            ## 第2个例子(假如给定了图片)
            {{
            "Hint": "图：松饼正在冷却。",
            }}
            ## 生成的问答
            {{
                "question": "图中展示的是什么糕点？",
                "answer": "松饼"
            }}
            2. 判断生成的「问题」是否有效，下面是几种无效问题的类型：
            - 问题不是从Hint改写或推理而来，不是看图问答风格的提问；
            - 问题中提问的对象代词和答案所属的类别不一致；
            - 问题的语义不流畅，有明显语病；
            - 问题对图片的理解有问题以至于提出了不合理的问题；
            - 提出的问题即使看不到图片也能正确作答，导致图片信息没有价值；
            3. 然后判断生成的「答案」是否合理，下面是几种无效答案：
            - 生成的答案不是问题的唯一合理答案；
            - 答案不是从Hint中抽取得来，也不是从它们描述的上下文语境中推理得出；
            - 答非所问，答案的内容和问题所问的不匹配；
            - 答案空洞、没有意义，或者对图片的理解有问题以至于给出了不合理的答案；
            - 答案本身没有充分的依据支撑，存在很大的不确定性。
            - 答案中存在幻觉问题，出现胡说八道或者严重的逻辑问题。
            4. 如果生成的「问题」和「答案」都有效，才是一条合格的数据。

            返回格式如下：
            {{
                "question": "不改变原问题的语种，内容不包含答案",
                "answer": "保持原语种，保证从Hint中抽取或推理得到的、正确的那个"
                "qualified": "生成的问答是否合格，不合格给出原因。"
            }}
            下面请严格按照以上格式生成回复，尽量返回一组合格的问答。"""

            # print(query)
            # data = {"query": prompt, "image_path":image_path, "use_g4o":True}
            # response = requests.post(api_url_list[0], json=data).json()
            response = pool.submit(get_case_refine, prompt, image_path)

            # futures.append([prompt, image_path, pool.submit(get_case_refine, prompt, image_path)])
            # print(response)
            try:
                res = response.result()['response'][0].replace("```json", "").replace("```", "")
                # res = response['response'][0].replace("```json", "").replace("```", "")
            except Exception as e:
                res = {
                    "question": "question error",
                    "answer": "answer error",
                    "qualified": "不合格，parsing error"
                }
            json_res = json.loads(res, strict=False)
            print(res)
            res_json = {
                "data_id": line['data_id'],
                "query": line['question'],
                "hint": line['hint'],
                "question": json_res['question'],
                "answer": json_res['answer'],
                "qualified": json_res['qualified'],
                "category": line['category'],
                "image": line['image'],
                "l2-category": line['l2-category'],
                "split": line['split']
            }
            
            # res_json["g4o_response"] = res
            out_json.append(res_json)
            print("--"*20)

    print("简易问答个数：",cnt)
    # 将 Python 对象保存到 JSON 文件
    json.dump(out_json, fout, ensure_ascii=False, indent=4)
    json.dump(special_query, fout, ensure_ascii=False, indent=4)
    json.dump(simple_query, fout, ensure_ascii=False, indent=4)
    # fout.write("\n")
    fout.close()
    return

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    ####################  需要校验下面这几个参数  ###################
    # commen_path = "simpleVQA/MMEbench/json_files/"
    # for file_name in os.listdir(commen_path):
    #     if 'refine' in file_name:
    #         continue
    #     if 'artwork' in file_name:
    #         continue
    #     if os.path.isfile('refine_'+file_name):
    #         continue
    #     in_path_json = os.path.join(commen_path, file_name)
    #     out_path_json = os.path.join(commen_path, 'refine_'+file_name)
    #     main(in_path_json, out_path_json)

    in_path_json = "simpleVQA/MMBench_DEV_CN_V11.json"
    out_path_json = "simpleVQA/MMBench_g4o_refine_temp.json"

    # temperature_list = [0.7, 0.9, 1.0, 1.2]
    # top_p_list = [0.8, 1.0]  # top_p 值范围在[0,1]之间

    with open(in_path_json, 'r', encoding='utf-8') as f:
        lines = json.load(f) # 解析json文件
        print(len(lines))
    main(in_path_json, out_path_json)
    
    # commen_path = "simpleVQA/MMEbench/"
    # for dir_name in os.listdir(commen_path):
    #     folder_path = os.path.join(commen_path, dir_name)
    #     txt_to_json(folder_path)