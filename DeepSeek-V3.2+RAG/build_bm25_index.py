import bm25s
from datasets import load_from_disk
import os
from config import WIKI_DIR, BM25S_INDEX_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("加载数据集...")
dataset = load_from_disk(WIKI_DIR)
texts = [item['text'] for item in dataset]

print("分词并构建 BM25S 索引...")
tokenized_corpus = bm25s.tokenize(texts)
bm25_retriever = bm25s.BM25()
bm25_retriever.index(tokenized_corpus)   # 使用 index() 代替 fit()

print("保存 BM25S 索引...")
bm25_retriever.save(BM25S_INDEX_PATH)    # 只传路径，不传 corpus
print(f"BM25S 索引已保存至 {BM25S_INDEX_PATH}")