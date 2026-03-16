# retrieval.py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from api_client import call_rerank_api
from config import TOP_K, RETRIEVE_N

def retrieve_context(query, emb_model, faiss_index, wiki_dataset, top_k=TOP_K, retrieve_n=RETRIEVE_N):
    """
    双阶段检索
    """
    # 1. 查询向量化
    q_vec = emb_model.encode([query], normalize_embeddings=True)
    q_vec = q_vec.astype(np.float32)

    # 2. Faiss搜索
    distances, indices = faiss_index.search(q_vec, retrieve_n)
    indices = indices[0]

    # 3. 获取原始文本
    initial_contexts = []
    for idx in indices:
        text = wiki_dataset[int(idx)]['text'].replace('\n', ' ')
        initial_contexts.append(text)

    # 4. Rerank
    pairs = [[query, ctx] for ctx in initial_contexts]
    scores = call_rerank_api(pairs)

    # 5. 排序并取top_k
    sorted_idx = np.argsort(-np.array(scores))[:top_k]
    final_contexts = [initial_contexts[i] for i in sorted_idx]

    return "\n\n".join(final_contexts)

def batch_retrieve(test_df, emb_model, faiss_index, wiki_dataset, cache_path):
    """
    对测试集所有问题进行检索，并保存缓存
    """
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Retrieving"):
        try:
            context = retrieve_context(row['prompt'], emb_model, faiss_index, wiki_dataset)
        except Exception as e:
            print(f"检索失败 id={row.get('id', idx)}: {e}")
            context = "No context available."
        results.append({'id': row.get('id', idx), 'context': context})

    pd.DataFrame(results).to_csv(cache_path, index=False)
    print(f"检索结果已缓存至 {cache_path}")
    return results

# 测试
# if __name__ == "__main__":
#     from data_loader import load_embedding_model, load_faiss_index, load_wiki_dataset
#     emb = load_embedding_model(device='cpu')
#     idx = load_faiss_index()
#     wiki = load_wiki_dataset()
#     context = retrieve_context("What is the capital of France?", emb, idx, wiki)
#     print(context)