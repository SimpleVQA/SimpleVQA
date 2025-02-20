from vlmeval.dataset import ImageMCQDataset
from vlmeval.smp import mmqa_display
import json
import os

prefix = 'CCBench'
# Load MMBench_DEV_EN
dataset = ImageMCQDataset(prefix)

# To build multi-modal prompt for samples in dataset (by index)
# item = dataset.build_prompt(0)
# print(type(item)) # list
""" 
The output will be:
[
    {'type': 'image', 'value': '/root/LMUData/images/MMBench/241.jpg'},  # The image will be automatically saved under ~/LMUData/
    {'type': 'text', 'value': "Hint: The passage below describes an experiment. Read the passage and then follow the instructions below.\n\nMadelyn applied a thin layer of wax to the underside of her snowboard and rode the board straight down a hill. Then, she removed the wax and rode the snowboard straight down the hill again. She repeated the rides four more times, alternating whether she rode with a thin layer of wax on the board or not. Her friend Tucker timed each ride. Madelyn and Tucker calculated the average time it took to slide straight down the hill on the snowboard with wax compared to the average time on the snowboard without wax.\nFigure: snowboarding down a hill.\nQuestion: Identify the question that Madelyn and Tucker's experiment can best answer.\nOptions:\nA. Does Madelyn's snowboard slide down a hill in less time when it has a thin layer of wax or a thick layer of wax?\nB. Does Madelyn's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?\nPlease select the correct answer from the options above. \n"}
]
"""

# To visualize samples in dataset (by index)
# print(dataset.__len__())
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(prefix):
    os.makedirs(prefix)
    print(f"目录 '{prefix}' 创建成功")
else:
    print(f"目录 '{prefix}' 已存在")
    
for i in range(dataset.__len__()):
    # dataset.display(i)
    item = dataset.build_prompt(i)
    img_index = os.path.basename(item[0]['value']).split('.')[0]
    filename = f'vqa_{img_index}.json'
    
    with open(os.path.join(prefix, filename), 'w', encoding='utf-8') as json_file:
        json.dump(item, json_file, ensure_ascii=False, indent=4)
""" 
The output will be:
<image>
QUESTION. Identify the question that Madelyn and Tucker's experiment can best answer.
HINT. The passage below describes an experiment. Read the passage and then follow the instructions below.

Madelyn applied a thin layer of wax to the underside of her snowboard and rode the board straight down a hill. Then, she removed the wax and rode the snowboard straight down the hill again. She repeated the rides four more times, alternating whether she rode with a thin layer of wax on the board or not. Her friend Tucker timed each ride. Madelyn and Tucker calculated the average time it took to slide straight down the hill on the snowboard with wax compared to the average time on the snowboard without wax.
Figure: snowboarding down a hill.
A. Does Madelyn's snowboard slide down a hill in less time when it has a thin layer of wax or a thick layer of wax?
B. Does Madelyn's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?
ANSWER. B
CATEGORY. identity_reasoning
SOURCE. scienceqa
L2-CATEGORY. attribute_reasoning
SPLIT. dev
"""
