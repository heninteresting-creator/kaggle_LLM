# config.py
import os

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取 config.py 所在目录，即项目根目录

FAISS_PATH = os.path.join(BASE_DIR, "data", "all-para-270k-wiki-stem_faiss_roberta", "paraphs_parse_index.faiss")
WIKI_DIR = os.path.join(BASE_DIR, "data", "all-para-270k-wiki-stem_faiss_roberta", "all-para-expanded-270k-wiki-stem")  # 请确认这个路径是否正确
TEST_CSV = os.path.join(BASE_DIR, "data", "test.csv")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "train.csv")
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "data", "stsb-roberta-large")  # 用 BASE_DIR 构建，保持统一
CACHE_PATH = os.path.join(BASE_DIR, "rag_contexts_cache.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "eval_api_results.csv")

# ========== API配置 ==========
SILICONFLOW_API_KEY = "sk-sctpwmxcoygvdyprtfeqkfoovcsijmozgrcvrheunckuwbqw"              # 替换为真实key
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"

# ========== 检索参数 ==========
TOP_K = 10
RETRIEVE_N = 100

# ========== 生成参数 ==========
TEMPERATURE = 0.5
MAX_TOKENS = 1024
NUM_TRIALS = 3