# retrieval.py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from api_client import call_rerank_api
from config import (
    TOP_K,
    RETRIEVE_N,
    RETRIEVE_N_BM25,
    HYBRID_WEIGHT_VECTOR,
    HYBRID_WEIGHT_BM25,
    CONTEXT_N_VECTOR,
    CONTEXT_N_BM25,
    CONTEXT_N_TOTAL,
    RAG_MODE
)
import bm25s
import pickle

def _dedup_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def retrieve_context_no_rag(query, wiki_dataset, top_k=CONTEXT_N_TOTAL):
    """不使用RAG，返回空上下文"""
    return ""

def retrieve_context_vector_only(
    query,
    emb_model,
    faiss_index,
    wiki_dataset,
    top_k=CONTEXT_N_TOTAL,
    retrieve_n=RETRIEVE_N,
):
    """
    纯向量检索：返回 top_k 个段落
    """
    q_vec = emb_model.encode([query], normalize_embeddings=True).astype(np.float32)
    _, indices = faiss_index.search(q_vec, max(retrieve_n, top_k))
    indices_vec = indices[0][:top_k]
    contexts = [wiki_dataset[int(idx)]["text"].replace("\n", " ") for idx in indices_vec]
    return "\n\n".join(contexts)

def retrieve_context_vector_rerank(
    query,
    emb_model,
    faiss_index,
    wiki_dataset,
    top_k=CONTEXT_N_TOTAL,
    retrieve_n=RETRIEVE_N,
):
    """
    向量检索 + 重排：先用向量检索获取候选，再用重排模型排序
    """
    # 先用向量检索获取更多候选
    q_vec = emb_model.encode([query], normalize_embeddings=True).astype(np.float32)
    _, indices = faiss_index.search(q_vec, max(retrieve_n, top_k))
    indices_vec = indices[0][:retrieve_n]
    
    # 获取候选文档
    candidates = [wiki_dataset[int(idx)]["text"].replace("\n", " ") for idx in indices_vec]
    
    # 构造重排对
    pairs = [(query, doc) for doc in candidates]
    
    # 调用重排API
    scores = call_rerank_api(pairs)
    
    # 按分数排序
    scored_docs = list(zip(candidates, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 取top_k
    top_docs = [doc for doc, score in scored_docs[:top_k]]
    return "\n\n".join(top_docs)

def retrieve_context_hybrid(
    query,
    emb_model,
    faiss_index,
    wiki_dataset,
    bm25s_retriever,
    *,
    n_vector=CONTEXT_N_VECTOR,
    n_bm25=CONTEXT_N_BM25,
    total_k=CONTEXT_N_TOTAL,
    retrieve_n=RETRIEVE_N,
    retrieve_n_bm25=RETRIEVE_N_BM25,
    weight_vector=HYBRID_WEIGHT_VECTOR,
    weight_bm25=HYBRID_WEIGHT_BM25,
    strategy="concat",
):
    """
    混合检索：向量检索 + BM25S
    - strategy="concat": 先取向量 n_vector，再取 BM25S n_bm25，去重后截断到 total_k
    """
    n_vector = int(n_vector)
    n_bm25 = int(n_bm25)
    total_k = int(total_k)

    # 1) 向量检索
    q_vec = emb_model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = faiss_index.search(q_vec, max(retrieve_n, n_vector, total_k))
    indices_vec = indices[0]

    # 2) BM25S 检索
    query_tokens = bm25s.tokenize([query])
    results_bm25s, scores_bm25s = bm25s_retriever.retrieve(query_tokens, k=max(retrieve_n_bm25, n_bm25, total_k))
    indices_bm25s = results_bm25s[0]

    if strategy == "concat":
        chosen = _dedup_keep_order(list(indices_vec[:n_vector]) + [int(i) for i in indices_bm25s[:n_bm25]])
        if len(chosen) < total_k:
            chosen = _dedup_keep_order(chosen + list(indices_vec) + [int(i) for i in indices_bm25s])
        chosen = chosen[:retrieve_n]  # 🚀 先保留更多候选用于重排
        
        # 3) 获取候选文档内容
        candidates = [wiki_dataset[int(idx)]["text"].replace("\n", " ") for idx in chosen]
        
        # 4) 🆕 调用重排API
        pairs = [(query, doc) for doc in candidates]
        scores = call_rerank_api(pairs)
        
        # 5) 按重排分数排序
        scored_docs = list(zip(chosen, candidates, scores))
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        # 6) 取top_k
        top_docs = [doc for idx, doc, score in scored_docs[:total_k]]
        return "\n\n".join(top_docs)

def retrieve_context(
    query,
    emb_model,
    faiss_index,
    wiki_dataset,
    bm25s_retriever,
    *,
    n_vector=CONTEXT_N_VECTOR,
    n_bm25=CONTEXT_N_BM25,
    total_k=CONTEXT_N_TOTAL,
    retrieve_n=RETRIEVE_N,
    retrieve_n_bm25=RETRIEVE_N_BM25,
    weight_vector=HYBRID_WEIGHT_VECTOR,
    weight_bm25=HYBRID_WEIGHT_BM25,
):
    """
    根据配置的RAG模式进行检索
    """
    if RAG_MODE == 'no_rag':
        return retrieve_context_no_rag(query, wiki_dataset, total_k)
    elif RAG_MODE == 'vector_only':
        return retrieve_context_vector_only(
            query, emb_model, faiss_index, wiki_dataset, 
            top_k=total_k, retrieve_n=retrieve_n
        )
    elif RAG_MODE == 'vector_rerank':
        return retrieve_context_vector_rerank(
            query, emb_model, faiss_index, wiki_dataset,
            top_k=total_k, retrieve_n=retrieve_n
        )
    elif RAG_MODE == 'hybrid':
        if bm25s_retriever is None:
            raise ValueError("BM25S retriever is required for hybrid mode but not loaded")
        return retrieve_context_hybrid(
            query, emb_model, faiss_index, wiki_dataset, bm25s_retriever,
            n_vector=n_vector, n_bm25=n_bm25, total_k=total_k,
            retrieve_n=retrieve_n, retrieve_n_bm25=retrieve_n_bm25,
            weight_vector=weight_vector, weight_bm25=weight_bm25,
            strategy="concat"
        )
    else:
        raise ValueError(f"Unknown RAG mode: {RAG_MODE}")

def batch_retrieve(
    test_df,
    emb_model,
    faiss_index,
    wiki_dataset,
    bm25s_retriever,
    cache_path,
    *,
    n_vector=CONTEXT_N_VECTOR,
    n_bm25=CONTEXT_N_BM25,
    total_k=CONTEXT_N_TOTAL,
):
    """
    对测试集所有问题进行检索，并保存缓存
    """
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Retrieving"):
        try:
            context = retrieve_context(
                row["prompt"],
                emb_model,
                faiss_index,
                wiki_dataset,
                bm25s_retriever,
                n_vector=n_vector,
                n_bm25=n_bm25,
                total_k=total_k,
            )
        except Exception as e:
            print(f"检索失败 id={row.get('id', idx)}: {e}")
            context = "No context available."
        results.append({'id': row.get('id', idx), 'context': context})

    pd.DataFrame(results).to_csv(cache_path, index=False)
    print(f"检索结果已缓存至 {cache_path}")
    return results


