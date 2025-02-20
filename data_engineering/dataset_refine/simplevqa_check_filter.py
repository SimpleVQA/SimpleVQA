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
from prompt.prompt_simplevqa import get_refine_prompt

sk = ""
url = ""
retry_times = 16
thread_num = 8

# 数据过滤
refine_prompt_version = "v0.0"

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

def parse_data_validation(response):
    res = {}
    question_analysis_pattern = "- **针对「问题」的分析**："
    question_valid = "- **「问题」是否有效**"
    answer_analysis_pattern = "- **针对「答案」的分析**："
    answer_valid = "- **「答案」是否有效**"
    validation = "- **该条数据是否合格**："

    for line in response.split("\n"):
        if question_analysis_pattern in line:
            res["question_analysis"] = line.replace(question_analysis_pattern, "").strip()
        if question_valid in line:
            res["question_valid"] = line.replace(question_valid, "").strip()
        if answer_analysis_pattern in line:
            res["answer_analysis"] = line.replace(answer_analysis_pattern, "").strip()
        if answer_valid in line:
            res["answer_valid"] = line.replace(answer_valid, "").strip()
        if validation in line:
            res["validation"] = line.replace(validation, "").strip()
    # res["org_result"] = response
    return res

def get_response(image_path, input):
    res = ""
    for i in range(retry_times):
        try:
            base64_image = image_to_base64(image_path)
            block = {
                "model":
                    "gpt-4o", # "qwen-vl-max", # "gpt-4o", # 
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
    tcase['qualified'] = res
    if 'validation' in res:
        tcase['qa_validation'] = res['validation']
    else:
        tcase['qa_validation'] = res
    return tcase

def run_craw_gpt4o_refine():
    file_path = 'simpleVQA/mmvet_refined.json'

    with open(file_path) as fin:
        data = json.load(fin)
    image_root = 'simpleVQA/images/MMVet/'
    output_data = []

    pool = ThreadPoolExecutor(thread_num)
    for line in data:
        # tcase = json.loads(line)
        # if 'validation' in line['qualified']:
        #     continue
        output_data.append([line['data_id'], pool.submit(get_case_refine, line, image_root)])

    output_path = "simpleVQA/craw_qa_g4o_for_" + file_path.split("/")[-1].split(
            ".")[0] + "_" + refine_prompt_version + ".json"

    filter_data = []
    for i in tqdm(output_data):
        # fout.write(json.dumps(i[1].result(), ensure_ascii=False)+"\n")
        filter_data.append(i[1].result())

    with open(output_path, "a") as fout:
        json.dump(filter_data, fout, ensure_ascii=False, indent=4)

def data_filter():
    file_path = 'simpleVQA/mmvet_refined.json'
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

    # 问题抓取
    # run_craw_gpt4o_question()
    # 答案抓取
    # run_craw_gpt4o_answer()
    # refine抓取
    # run_craw_gpt4o_refine()

    # 过滤得到合格的simplevqa
    # data_filter()

    end = datetime.datetime.now()
    print("end time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    time_difference = end - now
    total_seconds = time_difference.total_seconds()
    print('Total seconds: ', total_seconds)

    # file_path = './mmbench_for_simplevqa_qwen.json'
    # output_path = 'mmvet_for_simplevqa_qwen.json'
    # with open(file_path) as fin:
    #     data = json.load(fin)
    # cnt = 0
    # result = []
    # for item in data:
    #     if item['qa_validation'] == '是':
    #         cnt += 1
    #         result.append(item)
    # print(cnt)