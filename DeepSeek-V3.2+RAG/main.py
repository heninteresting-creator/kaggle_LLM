# main.py
import pandas as pd
from config import TEST_CSV, TRAIN_CSV, CACHE_PATH, OUTPUT_CSV
from data_loader import load_embedding_model, load_faiss_index, load_wiki_dataset
from retrieval import batch_retrieve
from generation import generate_for_test

def main():
    # 加载数据（可根据需要选择加载哪些）
    print("=== 步骤1: 加载数据 ===")
    emb_model = load_embedding_model(device='cpu')   # 使用cpu
    faiss_index = load_faiss_index()
    wiki_dataset = load_wiki_dataset()

    # 读取测试集
    test_df = pd.read_csv(TRAIN_CSV).head(2)

    # 步骤2: 检索（如果已有缓存可跳过）
    print("\n=== 步骤2: 检索上下文 ===")
    # 如果想跳过检索，注释下面这一行
    batch_retrieve(test_df, emb_model, faiss_index, wiki_dataset, CACHE_PATH)

    # 步骤3: 生成答案
    print("\n=== 步骤3: 生成答案 ===")
    generate_for_test(test_df, CACHE_PATH, OUTPUT_CSV)

if __name__ == "__main__":
    main()