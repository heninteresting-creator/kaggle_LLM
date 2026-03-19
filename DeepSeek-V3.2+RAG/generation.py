# generation.py
import re
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm
from api_client import call_llm_api
from config import TEMPERATURE, MAX_TOKENS, NUM_TRIALS
from concurrent.futures import ThreadPoolExecutor, as_completed

def build_prompt(row):
    """根据一行数据构造prompt"""
    return f"""You are an expert scientist answering multiple-choice questions. 
Please read the background context and answer the question.
Background knowledge:
{row['context']}
---
Based on the above background, answer the following multiple-choice question. 

Question: {row['prompt']}
Options:
A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}
E. {row['E']}

First, do not repeat the background. Then, briefly analyze the question in 2-3 sentences. Next, output your reasoning and the final answer strictly in the following JSON format:
{{
  "reasoning": "Your 2-3 sentence analysis here",
  "answer": "X"  # X is one of A, B, C, D, E
}}
"""

def extract_answer(text):
    """从生成文本中提取答案字母"""
    if not text:
        return "A"
    # 优先匹配JSON中的answer字段
    match = re.search(r'"answer"\s*:\s*"([A-E])"', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 备选：找最后一个出现的字母
    letters = re.findall(r'[A-E]', text.upper())
    return letters[-1] if letters else "A"

def _majority_vote(letters):
    letters = [x for x in letters if x]
    if not letters:
        return ""
    counts = Counter(letters)
    max_count = max(counts.values())
    # 若平票，按出现顺序取第一个达到最大票数的
    for x in letters:
        if counts[x] == max_count:
            return x
    return letters[0]


def generate_for_test(test_df, cache_path, output_path, *, answer_extractor=None, max_workers=10):
    """
    主生成流程：读取缓存，对每个问题生成NUM_TRIALS次答案，保存结果
    
    Args:
        max_workers: 并发线程数（默认10）
    """
    # 读取检索缓存
    cache_df = pd.read_csv(cache_path)
    context_dict = dict(zip(cache_df['id'], cache_df['context']))
    test_df['context'] = test_df['id'].map(context_dict)

    results_dict = {idx: {'row': row, 'raw_outputs': []} for idx, row in test_df.iterrows()}

    for trial in range(NUM_TRIALS):
        print(f"\n▶️ 第 {trial+1}/{NUM_TRIALS} 遍生成...")
        
        # 🚀 并发调用LLM
        def process_single(row):
            prompt = build_prompt(row)
            output = call_llm_api(prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            return row.name, output
        
        outputs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single, row) for idx, row in test_df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Trial {trial+1}"):
                outputs.append(future.result())
        
        # 按顺序保存结果
        for idx, output in outputs:
            results_dict[idx]['raw_outputs'].append(output)

    # 汇总 - 按照您的要求组织输出
    final = []
    for idx, data in results_dict.items():
        row = data['row']
        raw = data['raw_outputs']
        extractor = answer_extractor or extract_answer
        letters = [extractor(t) for t in raw]
        model_final = _majority_vote(letters)
        
        final.append({
            'id': row['id'],
            'prompt': row['prompt'],
            'correct': row.get('answer', ''),
            'trial1': letters[0] if len(letters) > 0 else '',
            'trial2': letters[1] if len(letters) > 1 else '',
            'trial3': letters[2] if len(letters) > 2 else '',
            'raw_trial1': raw[0] if len(raw) > 0 else '',
            'raw_trial2': raw[1] if len(raw) > 1 else '',
            'raw_trial3': raw[2] if len(raw) > 2 else '',
        })

    res_df = pd.DataFrame(final)
    res_df.to_csv(output_path, index=False)
    print(f"✅ 最终结果保存至 {output_path}")
    return res_df