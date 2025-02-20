from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn as nn
import os
import json
import base64
from tqdm import tqdm
import time
import traceback
# import jsonlines
from prompt import get_vqa_prompt

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 

ANSWER_NUM = 1 # 需要几个答案

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "models/Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)
# Qwen2VLForConditionalGeneration.from_pretrained(
#     "models/Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
# )


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
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

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
    # print(output_text)
    return output_text[0]

def get_refine_response(image_path, query):
    response = []
    for i in range(ANSWER_NUM):
        res = get_response(image_path, query)
        response.append(res)
    return response

def main(in_path_json, out_path_json, model_name, with_confidence=False):
    out_json = []
    prompt = ""
    fout = open(out_path_json, 'w', encoding='utf-8')
    with open(in_path_json, 'r', encoding='utf-8') as f:
        # lines = json.load(f) # 解析json文件
        lines = f.readlines()
        for item in tqdm(lines):
            line = json.loads(item)
            
            image_path = os.path.join("datasets/", line['image'])
            if not os.path.isfile(image_path):
                print("ERROR image!!!")
                out_json.append(line)
                continue
            
            res_json = line.copy()
            
            ## 获取置信度
            if line["data_id"] < 1012: ## 中文
                if with_confidence:
                    prompt = get_vqa_prompt("vqa_cn_with_confidence")
                else:
                    prompt = get_vqa_prompt("vqa_cn_without_confidence")
            else:  ## 英文
                if with_confidence:
                    prompt = get_vqa_prompt("vqa_en_with_confidence")
                else:
                    prompt = get_vqa_prompt("vqa_en_without_confidence")
                    
            prompt = prompt.replace("[<question>]", line["question"])
            # print(prompt)
            try:
                # res = response.result()['response'][0]
                response = get_refine_response(image_path, prompt)
                # print(response)
                res = response[0] # 从一组答案中获取一个
                res = res.replace("```json","").replace("```python","").replace("```","").strip()
                res = json.loads(res)
            except Exception as e:
                res = {"answer": "答案解析失败", "confidence_score": -1}

            res_json["{}_response".format(model_name)] = res['answer']
            if with_confidence:
                res_json["{}_confidence_score".format(model_name)] = res['confidence_score']
            
            out_json.append(res_json)
            # print("--"*10)

    # 将 Python 对象保存到 JSON 文件
    json.dump(out_json, fout, ensure_ascii=False, indent=4)
    # fout.write("\n")
    fout.close()
    return

if __name__ == "__main__":

    ####################  需要校验下面这几个参数  ###################
    in_path_json = "datasets/SimpleVQA_final_atomicQA_add_2025_without_reclassified.jsonl"
    out_path_json = "datasets/SimpleVQA_final_qwen2_5_72B_conf.json"
    
    model_name = "qwen2_5_72B"
    with_confidence = True
    # gen_prompt_dict = {
    #     "vqa_cn_without_confidence": vqa_cn_without_confidence,
    #     "vqa_en_without_confidence": vqa_en_without_confidence,
    #     "vqa_cn_with_confidence": vqa_cn_with_confidence,
    #     "vqa_en_with_confidence": vqa_en_with_confidence,
    # }
    
    main(in_path_json, out_path_json, model_name, with_confidence)