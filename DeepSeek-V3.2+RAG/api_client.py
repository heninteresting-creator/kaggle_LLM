# api_client.py
import time
import requests
from openai import OpenAI
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, RERANKER_MODEL, LLM_MODEL

client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)

def call_rerank_api(pairs, batch_size=100):
    """
    重排API调用 - 支持分批处理
    
    Args:
        pairs: [(query, doc1), (query, doc2), ...]
        batch_size: 每批发送的文档数量
    """
    query = pairs[0][0]
    documents = [p[1] for p in pairs]
    url = f"{SILICONFLOW_BASE_URL}/rerank"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    all_scores = []
    
    # 🚀 分批处理
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        payload = {
            "model": RERANKER_MODEL,
            "query": query,
            "documents": batch_docs,
            "top_n": len(batch_docs)
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            batch_scores = [0.0] * len(batch_docs)
            for res in data["results"]:
                batch_scores[res["index"]] = res["relevance_score"]
            all_scores.extend(batch_scores)
        except Exception as e:
            print(f"[Rerank Error] Batch {i//batch_size}: {type(e).__name__}: {e}")
            all_scores.extend([1.0] * len(batch_docs))
    
    return all_scores

def call_llm_api(prompt, temperature=0.5, max_tokens=1024, retries=3):
    url = f"{SILICONFLOW_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM Error] Attempt {attempt+1}/{retries}: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return ""

# 测试
# if __name__ == "__main__":
#     test_prompt = "某计算机的主存地址为32位，按字节编址。其缓存采用三路组相联方式，缓存总容量为48KB，每个缓存块（行）的大小为64B。（1）该缓存共有多少组？（2）假设CPU依次访问主存地址 A=0x12345678 和 B=0x12345678+64，问这两个地址映射到同一组吗？请计算它们对应的组索引，并说明理由。"
#     # test_prompt = "Hello, how are you?"
#     print(call_llm_api(test_prompt))