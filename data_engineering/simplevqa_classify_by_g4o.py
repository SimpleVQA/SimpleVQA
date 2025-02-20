import os
import json
import time
import datetime
import requests
import base64
from tqdm import tqdm
import traceback
import collections
from concurrent.futures import ThreadPoolExecutor
from prompt.prompt_classify import get_refine_prompt

sk = "API-sk"
url = "API-url"
retry_times = 10
thread_num = 4

# 数据过滤
refine_prompt_version = "v0.1"

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

def parse_data_validation(response):
    res = {}
# - **针对「任务类别」的分析**：...
# - **所属「任务类别」**：从可选的任务类别里选择
# - **针对「问答形式」的分析**：...
# - **所属「问答形式」**：从可选的问答形式里选择
# - **针对「主题类别」的分析**：...
# - **所属「主题类别」**：结合背景知识判别本问答涉及的领域
# - **针对「实体类别」的分析**：...
# - **问答主体所属「实体类别」**：结合图片和问答判别被提问的主体是什么分类

    task_category_analysis = "- **针对「任务类别」的分析**："
    task_category = "- **所属「任务类别」**："
    # vqa_format_analysis = "- **针对「问答形式」的分析**："
    # vqa_format = "- **所属「问答形式」**："
    subject_category_analysis = "- **针对「主题类别」的分析**："
    subject_category = "- **所属「主题类别」**："
    entity_class_analysis = "- **针对「实体类别」的分析**："
    entity_class = "- **问答主体所属「实体类别」**："

    for line in response.split("\n"):
        if task_category_analysis in line:
            res["task_category_analysis"] = line.replace(task_category_analysis, "").strip()
        if task_category in line:
            res["task_category"] = line.replace(task_category, "").strip()
        # if vqa_format_analysis in line:
        #     res["vqa_format_analysis"] = line.replace(vqa_format_analysis, "").strip()
        # if vqa_format in line:
        #     res["vqa_format"] = line.replace(vqa_format, "").strip()
        if subject_category_analysis in line:
            res["subject_category_analysis"] = line.replace(subject_category_analysis, "").strip()
        if subject_category in line:
            res["subject_category"] = line.replace(subject_category, "").strip()
        if entity_class_analysis in line:
            res["entity_class_analysis"] = line.replace(entity_class_analysis, "").strip()
        if entity_class in line:
            res["entity_class"] = line.replace(entity_class, "").strip()
    # res["org_result"] = response
    return res

def get_response(image_path, input):
    res = ""
    for i in range(retry_times):
        try:
            base64_image = image_to_base64(image_path)
            block = {
                "model":
                    "gpt-4o", # 
                "messages": [{
                    "role":
                        "user",
                    "content": [{
                        "type": "text",
                        "text": input
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}"
                        }
                    }]
                }]
            }
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {sk}'}
            response = requests.post(url, headers=headers, json=block).json()
            # print(response)
            res = response["choices"][0]["message"]["content"]
            break
        except Exception:
            res = "访问异常，需重试"
            print("访问异常，重试中...")
            #print("ERROR image_path:{} error:{} input:{} responose:{}".format(image_path, traceback.format_exc(), input, response))
    if res == "":
        print("无返回结果，请注意！！！")                                                             
    return res

def get_refine_response(image_path, query, answer):
    # caption = "无" if caption == "" else caption
    prompt = get_refine_prompt(refine_prompt_version)
    prompt = prompt.replace("[<question>]", query)
    prompt = prompt.replace("[<answer>]", answer)

    response = get_response(image_path, prompt)
    # print(response)

    res = parse_data_validation(response)
    print(res)
    # print("image_path:{} caption:{}\nresponse:{}\nquery:{} answer:{}\nparse_data_validation:{}".format(
    #     image_path, caption, response, query, answer, res))
    return response, prompt, res

def get_case_refine(tcase, image_root):
    # capiton = tcase["prediction"][0]
    # bos_url = os.path.basename(tcase["image_path"].split('?')[0])
    image_path = os.path.join(image_root, tcase["image"])
    question = tcase["question"]
    answer = tcase["answer"]
    response, prompt, res = get_refine_response(image_path, question, answer)
    tcase['vqa_category'] = res
    # if 'validation' in res:
    #     tcase['qa_validation'] = res['validation']
    # else:
    #     tcase['qa_validation'] = res
    return tcase

def run_craw_gpt4o_classify():
    file_path = 'simpleVQA/SimpleVQA_EN_add_atomic.json'

    image_root = 'simpleVQA/'
    output_data = []

    pool = ThreadPoolExecutor(thread_num)
    with open(file_path, "r", encoding='UTF-8') as fin:
        data = json.load(fin)
        for line in data:
            # tcase = json.loads(line)
            # if 'validation' in line['qualified']:
            #     continue
            output_data.append([line['data_id'], pool.submit(get_case_refine, line, image_root)])

    output_path = "simpleVQA/SimpleVQA_EN_add_atomic_classified.json"

    filter_data = []
    for i in tqdm(output_data):
        # fout.write(json.dumps(i[1].result(), ensure_ascii=False)+"\n")
        filter_data.append(i[1].result())

    with open(output_path, "a") as fout:
        json.dump(filter_data, fout, ensure_ascii=False, indent=4)

def data_filter():
    file_path = 'simpleVQA/0.json'
    output_path = 'mmvet_for_simplevqa_qwen.json'
    with open(file_path) as fin:
        data = json.load(fin)
    cnt = 0
    result = []
    for item in data:
        if item['qa_validation'] == '是':
            cnt += 1
            result.append(item)
    print(cnt)
    with open(output_path, "w") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    now = datetime.datetime.now()
    print("start time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    # vqa分类
    run_craw_gpt4o_classify()

    # 过滤得到合格的simplevqa
    # data_filter()

    end = datetime.datetime.now()
    print("end time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    time_difference = end - now
    total_seconds = time_difference.total_seconds()
    print('Total seconds: ', total_seconds)
