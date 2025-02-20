import requests
import traceback
import random
import logging
import json 
import time

# sk = ""  # 多模态专项
sk = ""  # 问答专项
# sk_list = []
sk_list = []
url = ""
retry_times = 10
thread_num = 5


def get_random_sk():
    sk = random.choice(sk_list)
    print(f">>> get_random_sk:{sk}")
    return sk

def get_o1_response(input):
    res = ""
    response = ""
    cnt = 1
    while res == "":
        if cnt % 50 == 0:
            time.sleep(300)
        for i in range(retry_times):
            try:
                cnt += 1
                tsk = get_random_sk()
                # base64_image = image_to_base64(image_path)

                block = {"model": "o1-preview", "messages": [{"role": "user", "content": input}]}
                # block = {"model": "o1-mini", "messages": [{"role": "user", "content": input}]}
                headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {tsk}'}
                response = requests.post(url, headers=headers, json=block).json()

                res = response["choices"][0]["message"]["content"]
                return res
            except Exception:
                print("ERROR:\n >>> error:{}\n>>> input:{}\n>>> response:{}".format(traceback.format_exc(), input, response))
                if cnt % 50 ==1:
                    print("ERROR:\n >>> get_o1_response error:{}\n>>> input:{}\n>>> response:{}".format(traceback.format_exc(), input, response))
    return res

def get_gpt4_response(input, model="gpt-4", cretry_times=0, temperature=-1):
    res = ""
    response = ""
    rtime = cretry_times if cretry_times > 0 else retry_times
    cnt = 1
    while res == "":
        if cnt % 50 == 0:
            time.sleep(300)
        for i in range(rtime):
            try:
                cnt += 1
                tsk = get_random_sk()
                if temperature >= 0 and cnt == 2:
                    block = {
                        "model": model,
                        "temperature": temperature,
                        "messages": [{"role": "user", "content": input}]
                    }
                else:
                    block = {"model": model, "messages": [{"role": "user", "content": input}]}
                headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {tsk}'}
                response = requests.post(url, headers=headers, json=block, timeout=180).json()
                res = response["choices"][0]["message"]["content"]
                return res
            except Exception:
                if cnt % 50 ==1:
                    print("ERROR:\n >>> get_gpt4_response {} error:{}\n>>> input:{}\n>>>  response:{}".format(model, traceback.format_exc(), input, response))
    return res


def get_g4o_response(input, cretry_times=0):
    res = ""
    rtime = cretry_times if cretry_times > 0 else retry_times
    cnt = 1
    while res == "":
        if cnt % 50 == 0:
            time.sleep(300)
        for i in range(rtime):
            try:
                cnt += 1
                tsk = get_random_sk()
                block = {
                    "model":
                        "gpt-4o",
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": input
                        }]
                    }]
                }
                headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {tsk}'}
                response = requests.post(url, headers=headers, json=block)
                # print(f">>> get_g4o_response:{response.text}")
                response = response.json()
                res = response["choices"][0]["message"]["content"]
                return res
            except Exception:
                print("ERROR times:{} get_gpt4o_response error:{} responose:{}".format(i,   traceback.format_exc(), response))
                if cnt % 50 ==1:
                    print("ERROR:\n >>> get_gpt4o_response error:{}\n>>> input:{}\n>>> response:{}".format(traceback.format_exc(), input, response))
    return res



if __name__ == "__main__":
    query = """2024百度世界大会附近的酒店价格是多少"""
    print(get_g4o_response(query))
