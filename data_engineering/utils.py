import requests
import time
from tqdm import tqdm
import random
import json
import base64
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import os
from config import Config

def get_query2image_path(query_image_path_json):
    '''
    Func:
        读取文件，获取 query => image_path 的映射
    '''
    query2image_path = {}
    with open(query_image_path_json, 'r') as f:
        lines = f.readlines()
        for line in lines:
            cur_dic = json.loads(line)
            query = cur_dic['query']
            image_path = cur_dic['image_path']
            query2image_path[query] = image_path
    return query2image_path


def image_to_base64(image_path):
    '''
    Func:
        将一张图片转化成base64 编码
    
    Args:
        image_path: 图片的地址
    '''
    # print(image_path)
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string


def filter_query(query):
    '''
    Func: 过滤query。因为有的query质量不高，或者说query本身存在格式问题，需要过滤。

    Args:
        query: 待过滤的 query
    '''
    if "<img>" in query:
        return True
    return False


def get_querys_from_json(in_path_json):
    '''
    Func:
        从文件中读取所有的query，输入文件是json格式
    '''
    querys = []
    with open(in_path_json, 'r' ) as f:
        lines = f.readlines()
        for line in lines:
            dic = json.loads(line)
            # 去除\n \r 空格等
            cur_query = dic['query'].strip("\n \r")
            querys.append(cur_query)
    return querys

def request_by_query_and_image_path(api_url, image_path, query, temperature = 0.2, top_p = 0.8):
    '''
    Func:
        根据query、image_path 向模型请求，获取结果
    Args:
        api_url: 调用的api url
        image_path: 需要处理的图片的路径
        query: 处理的query
        temperature：
        top_p: 
    '''
    base64_image = image_to_base64(image_path)
    # print(f"base64_image = {base64_image}")
    info = {
    "context": [
        {
        "role": "user",
        "utterance": [
            {
            # !!!!!!!!! 一定要注意这个地方的参数是 image_url !!!!!!!!!
            "type": "image_url",
            "image_url": {
                "url": f"{base64_image}",
                "detail": "high"
            }
            },
            {
            "type": "text",
            "text": ""
            }
        ]
        }
    ],
    "penalty_score": 1,
    "frequency_score": 0,
    "presence_score": 0,
    "max_dec_len": 1000,
    }
    info['context'][0]['utterance'][1]['text'] = query
    # info['context'][0]['utterance'][0]['text'] = query
    info['top_p'] = top_p
    info['temperature'] = temperature
    payload = json.dumps(info)
    headers = {
    'Content-Type': 'application/json',
    "Connection": "close"
    }
    # print("payload = ",payload)
    response = requests.request("POST", api_url, headers=headers, data=payload)

    answer = json.loads(response.text)
    # print(answer)
    return answer


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_path_json = "./data/evaluation/sample_infer_{}.json".format(timestamp)
    # get_query_cate([], out_path_json)
