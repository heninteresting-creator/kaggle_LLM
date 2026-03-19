# data_loader.py
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from config import FAISS_PATH, WIKI_DIR, EMBEDDING_MODEL_PATH
import bm25s
from config import BM25S_INDEX_PATH

def load_embedding_model(device='cpu'):
    """加载本地embedding模型，可指定device为'cpu'或'cuda'"""
    print(f"加载Embedding模型到 {device}...")
    return SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)

def load_faiss_index():
    """加载faiss索引（使用cpu版本）"""
    print("加载Faiss索引...")
    try:
        import faiss
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "未安装 faiss。请安装 `faiss-cpu`（或 GPU 环境安装 `faiss-gpu`）后再运行向量检索。"
        ) from e

    index = faiss.read_index(FAISS_PATH)
    print(f"索引包含 {index.ntotal} 个向量，维度 {index.d}")
    return index

def load_wiki_dataset():
    """加载维基百科文本数据集"""
    print("加载Wiki数据集...")
    dataset = load_from_disk(WIKI_DIR)
    print(f"数据集大小: {len(dataset)}，列: {dataset.column_names}")
    return dataset

def load_bm25s_index():
    """加载bm25s索引（使用新版本API）"""
    print("加载 BM25S 索引...")
    try:
        retriever = bm25s.BM25.load(BM25S_INDEX_PATH)
        print("BM25S 索引加载成功")
        return retriever
    except Exception as e:
        print(f"加载BM25S索引失败: {e}")
        return None


