# config.py 修改建议
from pathlib import Path
import os

# 确保使用绝对路径
BASE_DIR = Path(__file__).parent.parent.absolute()

# 数据目录配置
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "local_models"
FAISS_INDEX_DIR = DATA_DIR / "faiss"

# 自动创建必要目录
required_dirs = [DATA_DIR, PROCESSED_DIR, MODEL_DIR, FAISS_INDEX_DIR]
for dir_path in required_dirs:
    dir_path.mkdir(parents=True, exist_ok=True)

# 文件路径配置
TRAIN_DATA_PATH = PROCESSED_DIR / "train_data.csv"  # 训练数据路径
TEST_DATA_PATH = PROCESSED_DIR / "test_data.csv"    # 测试数据路径
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "scam_index.bin"

# 模型配置
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L6-v2"

# 评估参数
DEFAULT_THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4]
BATCH_SIZE = 128