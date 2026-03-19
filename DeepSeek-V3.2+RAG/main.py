# main.py (示例)
import pandas as pd
from data_loader import load_embedding_model, load_faiss_index, load_wiki_dataset, load_bm25s_index
from retrieval import batch_retrieve
from config import TEST_CSV, TRAIN_CSV, CACHE_PATH, RAG_MODE
from generation import generate_for_test

def main():
    print(f"当前RAG模式: {RAG_MODE}")
    
    # 加载数据
    test_df = pd.read_csv(TRAIN_CSV)
    emb_model = load_embedding_model(device='cpu')
    faiss_index = load_faiss_index()
    wiki_dataset = load_wiki_dataset()
    
    # 只有在需要BM25S时才加载
    bm25s_retriever = None
    if RAG_MODE in ['hybrid']:
        bm25s_retriever = load_bm25s_index()
    
    # 执行批量检索
    batch_retrieve(
        test_df,
        emb_model,
        faiss_index,
        wiki_dataset,
        bm25s_retriever,
        CACHE_PATH
    )

    output_path = f"final_predictions_{RAG_MODE}_all.csv"
    generate_for_test(test_df, CACHE_PATH, output_path)


if __name__ == "__main__":
    main()