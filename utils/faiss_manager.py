import os
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

logger = logging.getLogger(__name__)

class FAISSManager:
    """FAISS Index Manager with Enhanced Error Handling"""
    
    def __init__(self, persist_path: str = "data/faiss_index.bin"):
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.persist_path = persist_path
        
    def encode_text(self, text: str) -> np.ndarray:
        """编码单个文本并归一化"""
        try:
            vec = self.encoder.encode(text, convert_to_tensor=False).astype(np.float32)
            faiss.normalize_L2(vec.reshape(1, -1))
            logger.debug(f"Encoded text: '{text}' -> shape: {vec.shape}")
            return vec
        except Exception as e:
            logger.error(f"Encoding failed: {str(e)}")
            raise

    def build_index(self, texts: List[str]) -> None:
        """构建新的FAISS索引"""
        try:
            if not texts:
                raise ValueError("Empty text list")
            embeddings = self.encoder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            ).astype(np.float32)
            faiss.normalize_L2(embeddings)
            # 使用内积作为度量（归一化后内积即余弦相似度，范围[-1,1]）
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.add(embeddings)
            self._save_index()
            logger.info(f"Built index with {len(texts)} vectors")
        except Exception as e:
            logger.error(f"Index building failed: {str(e)}")
            raise

    def search_similar(self, query: str, k: int = 3) -> List[Dict]:
        """搜索相似文本，并返回列表，每个元素包含 index 与经过缩放的相似度"""
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            query_vec = self.encode_text(query).reshape(1, -1)
            scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))
            results = []
            scaling_factor = 0.5  # 调整缩放因子，可根据需要修改
            for idx, score in zip(indices[0], scores[0]):
                # 归一化：内积归一化后范围为[-1,1]，转换到[0,1]
                normalized = (score + 1) / 2
                similarity = normalized * scaling_factor
                results.append({"index": int(idx), "score": float(similarity)})
                logger.debug(f"Search result - index: {idx}, raw score: {score}, normalized: {normalized}, final similarity: {similarity}")
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def _save_index(self) -> None:
        """保存索引到文件"""
        if self.index and self.persist_path:
            faiss.write_index(self.index, self.persist_path)
            logger.info(f"Saved index to {self.persist_path}")

    def _load_index(self) -> None:
        """从文件加载索引"""
        if os.path.exists(self.persist_path):
            self.index = faiss.read_index(self.persist_path)
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
