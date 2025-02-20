import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import time
import traceback
from o1_api_nostop import get_o1_response, get_gpt4_response, get_g4o_response
import random
import uuid

sk = ""
url = ""
RETRY_TIMES = 10 # 如果请求失败了，尝试的次数
THREAD_NUM = 4
model = "gpt-4o" # "doubaopro32k-241215" # "gpt-4-turbo" # api1_32k  #o1-preview # gpt-4o
model_keys = ["g4o_response", "qwen_response", "doubao_response"]

def call_llm(model, query, temperature=1):
    res = ""
    if "o1" in model:
        res = get_o1_response(query)
    elif "4o" in model:
        res = get_g4o_response(query)
    elif "doubao" in model or "gpt-4" in model:
        res= get_gpt4_response(query, model) #, temperature=temperature)
    return res

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
            # if line["data_id"] not in [346]: # [201, 203, 360, 428, 430, 501]):
            #     continue
            res_json = line
            # prompt = """
            # 假设你是一个专业的看图问答标注师，能依据用户给定的原始问题和答案，来为图片生成一个相关的询问原子事实的问题。原子事实是关于对象的最简单、最原始、不可分割的经验，原子问题被定义为揭示原子事实的提问。现在用户提供一个原始问题，提问主题和某一张图片的内容或相关的背景知识相符，但是不给出图片。请你从原始问题中确定提问针对的实体对象，并结合该对象所属的类别生成一个原子问题。要求生成的原子问题逻辑合理，表达通顺，提问的口吻是在引导用户做看图问答任务。
            # ## 下面是几个根据原始问题生成原子问题的示例：
            # ## 示例1（原始问题围绕主体的某些属性提问）：
            # {{
            #     "original_question":"图中的文物属于我国哪个朝代？", 
            #     "atomic_question":"图中的文物叫什么？"
            # }}
            # ## 示例2（原始问题包含长上下文描述）：
            # {{
            #     "original_question":"该图描绘了xxxxx，它是一部电影的镜头，这部电影的导演是谁？", 
            #     "atomic_question":"该图来自于哪部电影？"
            # }}
            # ## 示例3（原始问题是根据上下文内容的填空题）:
            # {{
            #     "original_question":"完成文本以描述图表。\n溶质粒子在可渗透膜上双向移动。但是更多的溶质粒子通过膜向（）侧移动。当两侧浓度相等时，粒子达到平衡。", 
            #     "atomic_question":"完成文本以描述图表。\n溶质粒子在可渗透膜上双向移动。但是更多的溶质粒子通过膜向（）侧移动。当两侧浓度相等时，粒子达到平衡。"
            # }}
            # ## 示例4（原始问题是一个直观的原子问题）：
            # {{
            #     "original_question":"图中人物是谁？", 
            #     "atomic_question":"图中人物是谁？"
            # }}
            # {{
            #     "original_question":"这是一道看图猜成语题目，请问图中画面对应的是哪个成语？", 
            #     "atomic_question":"这是一道看图猜成语题目，请问图中画面对应的是哪个成语？"
            # }}
            # 这是一道看图猜成语题目，请问图中画面对应的是哪个成语？
            # ## 示例5（原始问题不是一个直观的原子问题）：
            # {{
            #     "original_question":"这是一道看图猜古诗的题目，请问图中画面对应的是首古诗，请回答诗名", 
            #     "atomic_question":"这是一道看图猜古诗的题目，请问图中画面对应的诗句是什么？"
            # }}
            # ## 现在任务正式开始，用户提供的原始问题为：
            # {question}
            # ## 请严格按照下面json格式输出，不要包含注释：
            # ```json
            # {{
            # "original_question": "xxxxx?"
            # "atomic_question": "xxxxx?"
            # }}
            # ```
            # """

            prompt = """
            Suppose you are a professional tagger who can generate an atomic fact-related question for the picture based on the original question and answer given by the user. Atomic facts are the simplest, most primitive, indivisible experiences about objects, and atomic questions are defined as questions that reveal atomic facts. Now the user provides an original question with a topic that matches the content of an image or relevant background information, but does not give the image. You identify the entity object from the original question and combine it with the class to which the object belongs to generate an atomic question. The generated atomic questions are required to be logical and smooth, and the tone of the questions is to guide the user to do the picture question and answer task.
            Here are a few examples of generating an atomic problem from the original problem:
            ## Example 1 (the original question was asked around some attribute of the body) :
            {{
            "original_question": "Which dynasty do the relics in the picture belong to in our country?" ,
            "atomic_question": "What is the artifact in the picture?"
            }}
            ## Example 2 (the original question contained a long context description) :
            {{
            "original_question": "The picture depicts xxxxx. It is a shot of a movie. Who is the director of this movie?" ,
            "atomic_question": "Which movie is this image from?"
            }}
            ## Example 3 (the original question was a fill-in-the-blank based on context) :
            {{
            "original_question": "Complete the text to describe the chart. The solute particles move bidirectionally on the permeable membrane. But more solute particles move through the membrane to the () side. When the concentrations on both sides are equal, the particles reach equilibrium. ,
            "atomic_question": "Completes the text to describe the chart. The solute particles move bidirectionally on the permeable membrane. But more solute particles move through the membrane to the () side. When the concentrations on both sides are equal, the particles reach equilibrium.
            }}
            ## Example 4 (the original problem was an intuitive atomic problem) :
            {{
            "original_question": "What is x in the equation?" ,
            "atomic_question": "What is x in the equation?"
            }}
            ## Example 5 (the original problem is not an intuitive atomic problem) :
            {{
            "original_question": "This is a question about guessing an ancient poem by looking at pictures. Please answer the name of the poem."
            "atomic_question": "This is a picture-guessing ancient poem question, may I ask the picture in the picture corresponding to the poem?"
            }}
            ## Now the task is officially started, the original question provided by the user is:
            {question}

            ## Please output strictly in the following json format, without comments.
            ## If the original question is in Chinese, please translate it back to English. The original question in English is not dealt with, and is directly returned.
            ## The generated atomic question must be in English:
            ```json
            {{
                "original_question": "xxxxx?"
                "atomic_question": "xxxxx?"
            }}
            ```
            """
            question = line["question"]
            answer = line["answer"]
            prompt = prompt.format(question=question)

            response = pool.submit(call_llm, model, prompt)
            # print(response.result())
            try:
                print(response.result())
                res = response.result()
                # res = response['response'][0].replace("```json", "").replace("```", "")
                res = res.replace("```json","").replace("```python","").replace("```","")
                res = json.loads(res)
            except Exception as e:
                res = {"original_question": question, "atomic_question": "答案解析失败"}
                print(traceback.print_exc())
            
            # res_json["question"] = res["original_question"]
            res_json["atomic_question"] = res["atomic_question"]
            out_json.append(res_json)
            print("--"*10)

    # 将 Python 对象保存到 JSON 文件
    json.dump(out_json, fout, ensure_ascii=False, indent=4)
    # fout.write("\n")
    fout.close()
    return

if __name__ == "__main__":

    ####################  需要校验下面这几个参数  ###################
    in_path_json = "simpleVQA/SimpleVQA_EN.json"
    out_path_json = "simpleVQA/SimpleVQA_EN_add_atomic.json"

    main(in_path_json, out_path_json)