# evaluation.py
import pandas as pd
import numpy as np

# 全局变量设置
PREDICTIONS_PATH = "final_predictions_hybrid_all.csv"
K_VALUE = 3

def apk(y_true, y_pred, k):
    """
    计算单个样本的 Average Precision@K
    关键修复：预测列表去重，避免重复答案多次计分
    """
    # 去重并保持顺序
    seen = set()
    y_pred_dedup = []
    for pred in y_pred:
        if pred not in seen:
            seen.add(pred)
            y_pred_dedup.append(pred)
    
    if len(y_pred_dedup) > k:
        y_pred_dedup = y_pred_dedup[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i, pred in enumerate(y_pred_dedup):
        if pred == y_true:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            break  # 只计算第一次命中（因为只有一个正确答案）
    
    # 对于单个正确答案，分母为1
    return score / 1.0

def map_at_k(y_true_list, y_pred_list, k=3):
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true_list and y_pred_list must have the same length")
    
    ap_scores = [apk(y_true, y_pred, k) for y_true, y_pred in zip(y_true_list, y_pred_list)]
    return np.mean(ap_scores)

def evaluate_predictions(predictions_path, k=3):
    df = pd.read_csv(predictions_path)
    
    y_true_list = df['correct'].tolist()
    
    y_pred_list = []
    for idx, row in df.iterrows():
        pred_list = []
        for i in range(1, k+1):
            col_name = f'trial{i}'
            if col_name in df.columns and pd.notna(row[col_name]):
                pred_list.append(str(row[col_name]).strip())
        y_pred_list.append(pred_list)
    
    map_score = map_at_k(y_true_list, y_pred_list, k)
    
    print(f"MAP@{k} Score: {map_score:.4f}")
    print(f"总样本数: {len(y_true_list)}")
    
    # 统计
    match_count = 0
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        if y_true in y_pred:
            match_count += 1
    print(f"前{k}个预测中包含正确答案的样本数: {match_count}/{len(y_true_list)}")
    
    # 保存
    eval_result = pd.DataFrame({'metric': [f'MAP@{k}'], 'score': [map_score]})
    eval_output_path = predictions_path.replace('.csv', f'_eval_MAP@{k}.csv')
    eval_result.to_csv(eval_output_path, index=False)
    print(f"评估结果已保存至 {eval_output_path}")
    
    return map_score

if __name__ == "__main__":
    evaluate_predictions(PREDICTIONS_PATH, K_VALUE)