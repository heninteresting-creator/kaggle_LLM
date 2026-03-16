# data_loader.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from config import FAISS_PATH, WIKI_DIR, EMBEDDING_MODEL_PATH

def load_embedding_model(device='cpu'):
    """加载本地embedding模型，可指定device为'cpu'或'cuda'"""
    print(f"加载Embedding模型到 {device}...")
    return SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)

def load_faiss_index():
    """加载faiss索引（使用cpu版本）"""
    print("加载Faiss索引...")
    index = faiss.read_index(FAISS_PATH)
    print(f"索引包含 {index.ntotal} 个向量，维度 {index.d}")
    return index

def load_wiki_dataset():
    """加载维基百科文本数据集"""
    print("加载Wiki数据集...")
    dataset = load_from_disk(WIKI_DIR)
    print(f"数据集大小: {len(dataset)}，列: {dataset.column_names}")
    return dataset