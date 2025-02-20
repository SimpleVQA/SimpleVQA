vqa_cn_without_confidence = """假如你是一个计算机视觉专家，请理解图片内容，并回答给出的问题，要求只能回答一个命名实体：
## 下面我们提出问题: 
[<question>]
"""

vqa_en_without_confidence = """If you are a computer vision expert, please understand the picture and answer the question below. The answer must be a short named entity:
## Now let's ask the question:
[<question>]
"""

vqa_cn_with_confidence = """假如你是一个计算机视觉专家，请理解图片内容，并回答给出的问题：
## 下面我们提出问题: 
[<question>]
请基于此问题提供你的最佳答案，要求只能回答一个命名实体，并用0到100的分数表示你对该答案的信心（置信度）。请严格按照如下的JSON格式给出回复：
```json
{{
    "answer": "你的答案",
    "confidence_score": 你的置信度
}}
```
"""

vqa_en_with_confidence = """If you are a computer vision expert, understand the content of the image and answer the given questions:
## Below we ask the question. 
[<question>]
Please provide your best answer based on this question, which should be answered by only one named entity, and indicate your confidence (level of confidence) in the answer with an integer score from 0 to 100. Please give your response strictly in the following JSON format:
```json
{{
    "answer": "your answer",
    "confidence_score": your confidence level
}}
```
"""

def get_vqa_prompt(version):
    gen_prompt_dict = {
        "vqa_cn_without_confidence": vqa_cn_without_confidence,
        "vqa_en_without_confidence": vqa_en_without_confidence,
        "vqa_cn_with_confidence": vqa_cn_with_confidence,
        "vqa_en_with_confidence": vqa_en_with_confidence,
    }
    if version not in gen_prompt_dict:
        raise ValueError("Invalid version")
    return gen_prompt_dict[version]