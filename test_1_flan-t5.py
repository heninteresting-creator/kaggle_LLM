import pandas as pd
import os
import warnings
warnings.simplefilter("ignore")

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from string import Template
from pathlib import Path

# -------------------- 自动查找 test.csv --------------------
def find_file(root_dir, target_filename):
    for dirpath, _, filenames in os.walk(root_dir):
        if target_filename in filenames:
            return os.path.join(dirpath, target_filename)
    return None

data_root = '/kaggle/input'
test_file = find_file(data_root, 'test.csv')
if test_file is None:
    raise FileNotFoundError("未找到 test.csv，请检查输入数据。")
print(f"找到测试文件: {test_file}")

# 读取测试数据，将 id 列设为索引
df = pd.read_csv(test_file, index_col='id')
print(f"测试集样本数: {len(df)}")

# -------------------- 加载模型和tokenizer --------------------
# 模型路径（假设你已将flan-t5-base数据集添加到notebook中）
# 如果使用不同版本，请修改路径
model_path = '/kaggle/input/models/google/flan-t5/pytorch/base/4'  # 根据实际路径调整
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# -------------------- 构造输入模板 --------------------
preamble = (
    'Answer the following question by outputting the letters A, B, C, D, and E '
    'in order of the most likely to be correct to the to least likely to be correct.'
)
template = Template('$preamble\n\n$prompt\n\nA) $a\nB) $b\nC) $c\nD) $d\nE) $e')

def format_input(row):
    """从一行数据构造输入文本"""
    return template.substitute(
        preamble=preamble,
        prompt=row['prompt'],
        a=row['A'],
        b=row['B'],
        c=row['C'],
        d=row['D'],
        e=row['E']
    )

# -------------------- 后处理函数 --------------------
def post_process(predictions):
    valid = set(['A', 'B', 'C', 'D', 'E'])
    # 如果没有任何有效字母，返回默认排序
    if set(predictions).isdisjoint(valid):
        return 'A B C D E'
    else:
        final_pred = []
        for pred in predictions:
            if pred in valid:
                final_pred.append(pred)  # 注意这里用 append，原代码用 += 对字符串会拆成字符
        # 补全缺失的字母
        to_add = valid - set(final_pred)
        final_pred.extend(list(to_add))
        # 空格连接
        return ' '.join(final_pred)

# -------------------- 推理循环 --------------------
# 初始化一个列表存放预测结果

predictions = []

for idx in df.index:
    # 获取当前行的数据
    row = df.loc[idx]
    # 构造输入文本
    input_text = format_input(row)
    # 编码
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    # 生成
    with torch.no_grad():
        outputs = model.generate(**inputs)
    # 解码
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  # 得到字符串，如 "A"
    # 后处理（注意模型可能输出单个字母，也可能输出多个，后处理函数期望列表）
    # 这里简单将 answer 按空格分割，如果模型输出 "A B C"，则 split 得到 ["A","B","C"]
    pred_list = answer.split()
    final_pred = post_process(pred_list)
    predictions.append(final_pred)
    

# 将预测结果添加到 DataFrame
df['prediction'] = predictions

# -------------------- 保存提交文件 --------------------
submission = df[['prediction']]  
submission.to_csv('submission.csv')
print("提交文件已生成: submission.csv")