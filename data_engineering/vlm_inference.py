from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.nn as nn
import os
import json
import base64
from tqdm import tqdm
import time
import traceback

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 

ANSWER_NUM = 1 # 需要几个答案

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "models/Qwen/Qwen2-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)


# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model, device_ids=[0,1,2])
# model.to(device)
    
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

def image_to_base64(image_path):
    # print(image_path)
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

def get_response(image_path, prompt):
    image = Image.open(image_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(output_text)
    return output_text[0]

def get_refine_response(image_path, query):
    response = []
    for i in range(ANSWER_NUM):
        res = get_response(image_path, query)
        response.append(res)
    return response

def main(in_path_json, out_path_json):
    out_json = []
    fout = open(out_path_json, 'a', encoding='utf-8')
    with open(in_path_json, 'r', encoding='utf-8') as f:
        lines = json.load(f) # 解析json文件
        for line in tqdm(lines):
            # print(line['question'])
            image_path = os.path.join("datasets/CCBench", line['image'])
            if image_path == "":
                continue
            res_json = line # {"query": line["query"], "image_path": line['image_path']}
            prompt = f"""请理解图片内容，并回答给出的问题，要求只能回答一个命名实体：
            ## 问题: 
            {line["question"]}
            """
        
            try:
                # res = response.result()['response'][0]
                response = get_refine_response(image_path, prompt)
                print(response)
                res = response[0]
            except Exception as e:
                res = "答案解析失败"

            # res = response['response'][0]
            # print(res)
            res_json["qwen72B_response"] = res
            out_json.append(res_json)
            print("--"*10)

    # 将 Python 对象保存到 JSON 文件
    json.dump(out_json, fout, ensure_ascii=False, indent=4)
    # fout.write("\n")
    fout.close()
    return

if __name__ == "__main__":

    ####################  需要校验下面这几个参数  ###################
    in_path_json = "datasets/ovall_CCBench_for_simplevqa_qwen_g4o_eb_doubao_response.json"
    out_path_json = "datasets/CCBench_for_simplevqa_req_qwen_g4o_eb_doubao_qwen72B.json"

    main(in_path_json, out_path_json)