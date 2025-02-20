import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import time
import traceback
from o1_api_nostop import get_o1_response, get_gpt4_response, get_g4o_response
import random
import uuid

sk = "API-sk"
url = "API-url"
RETRY_TIMES = 16 # 如果请求失败了，尝试的次数
THREAD_NUM = 4
model = "gpt-4o" # "doubaopro32k-241215" # "gpt-4-turbo" # api1_32k  #o1-preview
model_keys = ["g4o_response"]

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
            res_json = line.copy()

            # prompt = """
            # 你是一位审核专家，你会收到一个[题目]、[题目]对应的[标准答案]，以及[多个考生答案]，你的任务是根据[题目]，[标准答案]逐一分析这些[考生答案]是否正确，具体要求如下：
            # 1. 仔细阅读并充分理解[题目]和[标准答案]的内容与含义。
            # 2. 仔细阅读每条[考生答案]的内容。
            # 3. 考生答案只有下面三种情况：
            # - 【正确】：[考生答案]中的事实在[标准答案]中出现过，且与[标准答案]中的描述一致。或可以通过[标准答案]中的信息推断出[考生答案]中的事实是正确的。
            # - 【错误】：[考生答案]中的事实与[标准答案]中的信息相矛盾。或可以通过[标准答案]中的信息推断出[考生答案]中的事实是错误的。
            # - 【未尝试】：[考生答案]中的事实与这些[标准答案]中的信息既不蕴含也不矛盾，或无法根据这些[标准答案]给出明确的结论，或[考生答案]为空。
            # ---
            # # 题目：
            # {question}
            # # 标准答案：
            # {answer}
            # # 考生答案：
            # 针对每个考生作答的答案，你的回答都必须是一个含有一对键值的字典，这三个键值是-"考生答案i"、 "整体结论", 分别对应着考生模型的回答、最终结论（【正确】/【错误】/【未尝试】）。
            # 以下是给定的考生答案
            # {candidates}

            # 请严格按照以下格式回复，以JSON格式返回一个字典,而且字典第一层key不要替换为具体答案。不要返回其他任何内容。
            # [回复格式]: 
            # ```json
            # {{
            #     "考生答案0": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
            #     "考生答案1": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
            #     ...
            # }}
            # ```
            # """.strip()

            # ```json
            # {{
            #     "考生答案0": {{
            #         "reasoning": "分析这条[考生回答]与给定[标准答案]之间的关系，然后推导出最终结论。写出你的分析过程",
            #         "conclusion": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
            #     }},
            #     "考生答案1": {{
            #         "reasoning": "分析这条[考生回答]与给定[标准答案]之间的关系，然后推导出最终结论。写出你的分析过程",
            #         "conclusion": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
            #     }}...
            # }}
            # ```

            prompt = """
            请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。 
            首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。 
            ## 以下是【正确】的答复示例：
            ```
            问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？ 
            标准答案：玛丽亚·奥巴马和萨莎·奥巴马 
            模型预测1：Malia Obama and Sasha Obama 
            模型预测2：玛丽亚和萨沙 
            模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认 
            模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。 
            ```
            这些答复均为【正确】，因为： 
            - 完整地包含了标准答案中的重要信息。 
            - 不包含任何与标准答案矛盾的信息。 
            - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。 
            - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。 

            ## 以下是【错误】的答复示例：
            ```
            问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？ 
            标准答案：玛丽亚·奥巴马和萨莎·奥巴马 
            模型预测1：玛丽亚 
            模型预测2：玛丽亚、萨莎和苏珊 
            模型预测3：巴拉克·奥巴马没有孩子 
            模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。 
            模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。 
            模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？ 
            ```
            这些答复均为【错误】，因为： 
            - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。 

            ## 以下是【未尝试】的答复示例： 
            ```
            问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？ 
            标准答案：玛丽亚·奥巴马和萨莎·奥巴马 
            模型预测1：我不知道。 
            模型预测2：我需要更多关于您所指奥巴马的上下文。 
            模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。 
            模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。 
            模型预测5：我无法识别图中的人物。
            模型预测6：N/A。
            ```
            这些答复均为【未尝试】，因为： 
            - 没有包含标准答案中的重要信息。 
            - 回复中没有与标准答案矛盾的陈述。 
            只返回字母”A”、”B”或”C”，无须添加其他文本。

            另外注意以下几点： 
            - 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”： 
            - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。 
            - 预测答案“3520”和“3600”均为【错误】。 
            - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。 
            - 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。 
            - 例如，考虑问题“菱镁矿的主要化学成分是什么？”,标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。 
            - 如果从问题中明显可以推断出预测答案省略的信息，那么算作【正确】。 
            - 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。 
            - 如果能明显看出名字翻译版本不同但是是同一个人也认为【正确】。 
            - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均【正确】。 
            - 预测答案和标准答案对应的是同一事物，但是称呼不同，如“天坛”和“祈年殿”，那么算作【正确】

            ## 下面是一个新的问题示例。对每一个预测答案，请只回复"正确"、"错误"、"未尝试"之一，不要道歉或纠正自己的错误，只需要评估该回答。 
            ```
            问题: {question} 
            正确答案: {answer} 
            预测答案: {candidates} 
            ```

            请严格按照以下格式回复，以JSON格式返回一个字典,而且字典第一层key不要替换为具体答案。不要返回其他任何内容。
            [回复格式]: 
            ```json
            {{
                "预测答案0": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
                "预测答案1": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
                ...
            }}
            ```
            """.strip()
            
            # ```json
            # {{
            #     "预测答案0": {{
            #         "reasoning": "分析这条[预测答案]与给定[正确答案]之间的关系，然后推导出最终结论。写出你的分析过程",
            #         "conclusion": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
            #     }},
            #     "预测答案1": {{
            #         "reasoning": "分析这条[预测答案]与给定[正确答案]之间的关系，然后推导出最终结论。写出你的分析过程",
            #         "conclusion": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
            #     }}...
            # }}
            # ```
            question = line["question"]
            answer = line["answer"]
            candidates = ""
            for idx, model_key in enumerate(model_keys):
                # candidates += "\n[考生答案%d]：%s"%(idx, line[model_key])
                candidates += "\n[预测答案{}]：{}".format(idx, line[model_key])
            prompt = prompt.format(question=question, answer=answer, candidates=candidates)

            response = pool.submit(call_llm, model, prompt)
            # futures.append([prompt, image_path, pool.submit(get_case_refine, prompt, image_path)])
            
            try:
                # print(response.result())
                res = response.result()
                # res = response['response'][0].replace("```json", "").replace("```", "")
                res = res.replace("```json","").replace("```python","").replace("```","").strip()
                if res[-1] != "}":
                    res += "}"
                res = json.loads(res)
            except Exception as e:
                res = {"预测答案0":{"conclusion":"答案解析失败"}}
                print(traceback.print_exc())

            # res = response['response'][0]
            # print(res)
            new_res = {}
            for idx, key in enumerate(res.keys()):
                if idx >= len(model_keys):
                    break
                new_res[model_keys[idx]] = res[key]
                # new_res[model_keys[idx]] = res[key]['conclusion']
        
            res_json["judge_res"] = new_res
            out_json.append(res_json)
            # print("--"*10)

    # 将 Python 对象保存到 JSON 文件
    json.dump(out_json, fout, ensure_ascii=False, indent=4)
    # fout.write("\n")
    fout.close()
    return

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    ####################  需要校验下面这几个参数  ###################
    in_path_json = "simpleVQA/SimpleVQA_CN_qwen72B_janus7B_qw2572B_recheck_clip2.json"
    out_path_json = "simpleVQA/SimpleVQA_CN_qwen72B_janus7B_qw2572B_recheck_clip2_judge2.json"

    main(in_path_json, out_path_json)
