import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import logging
import asyncio
from tqdm import tqdm
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
from config import *

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FAISSManager:
    """增强路径验证的FAISS管理"""
    def __init__(self, index_path: Path, train_data_path: Path):
        self.index_path = index_path
        self.train_data_path = train_data_path
        
        # 路径存在性验证
        if not train_data_path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {train_data_path}")
        
        if not index_path.exists():
            self._build_index()
            
        self.index = self._load_index()
    
    def _build_index(self):
        """构建索引时添加进度提示"""
        from tqdm import tqdm
        
        logger.info(f"开始构建FAISS索引，使用训练数据: {self.train_data_path}")
        
        # 加载数据
        df = pd.read_csv(self.train_data_path)
        spam_texts = df[df['type'] == 'spam']['text'].tolist()
        
        if not spam_texts:
            raise ValueError("训练数据中未找到spam样本，无法构建索引")
        
        # 生成嵌入
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        embeddings = []
        batch_size = 32  # 减小批处理以降低内存消耗
        
        for i in tqdm(range(0, len(spam_texts), batch_size), desc="生成嵌入"):
            batch = spam_texts[i:i+batch_size]
            embeddings.extend(model.encode(batch, show_progress_bar=False))
        
        embeddings = np.array(embeddings)
        
        # 创建索引
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        
        # 保存索引
        faiss.write_index(index, str(self.index_path))
        logger.info(f"索引构建完成，保存至: {self.index_path}")

class FraudDetector:
    """纯本地风险评估器"""
    def __init__(self):
        # 加载本地模型
        self.faiss = FAISSManager(
            index_path=FAISS_INDEX_PATH,
            train_data_path=TRAIN_DATA_PATH  # 使用配置路径
        )
        # 加载FAISS索引
        self.faiss = FAISSManager(
            index_path=FAISS_INDEX_PATH,
            train_data_path=DATA_DIR / "processed" / "train_data.csv"  # 指定训练数据路径
        )
        
        # 配置相似度转换函数（相似度越高风险越大）
        self.sim_to_risk = lambda sim: 1 - np.exp(-5 * sim)
    
    async def analyze_text(self, text: str) -> Dict:
        """分析文本风险（无LLM调用）"""
        try:
            # 生成文本嵌入
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # FAISS相似度检索
            distances, _ = self.faiss.search(embedding.reshape(1, -1))
            avg_similarity = np.mean(distances)
            
            # 转换为风险分 [0,1]
            risk_score = float(self.sim_to_risk(avg_similarity))
            return {"risk_score": risk_score}
        except Exception as e:
            logger.error(f"分析失败: {str(e)}")
            return {"risk_score": 0.0}

class ThresholdOptimizer:
    def __init__(
        self,
        data_path: Path = TEST_DATA_PATH,  # 默认使用配置的测试数据路径
        test_size: int = 200,
        thresholds: List[float] = DEFAULT_THRESHOLDS,
        batch_size: int = BATCH_SIZE
    ):
        # 添加路径验证
        if not data_path.exists():
            raise FileNotFoundError(f"测试数据文件不存在: {data_path}")
        
        
        # 修复路径创建：递归创建目录
        self.output_dir = DATA_DIR / "eval_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)  # <- 关键修改
        
        # 初始化检测器
        self.detector = FraudDetector()

    async def optimize(self):
        """执行优化流程"""
        # 加载数据
        df = pd.read_csv(self.data_path)
        test_data = self._sample_data(df)
        
        # 获取预测结果
        y_true, y_probs = await self._get_predictions(test_data)
        
        # 分析阈值性能
        results = self._analyze_thresholds(y_true, y_probs)
        
        # 保存结果
        self._save_results(results, y_true, y_probs)
        return results

    def _sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """采样测试数据"""
        actual_size = min(len(df), self.test_size)
        return df.sample(n=actual_size, random_state=42)

    async def _get_predictions(self, data: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """批量获取预测结果"""
        y_true, y_probs = [], []
        
        for i in tqdm(range(0, len(data), self.batch_size), desc="处理进度"):
            batch = data.iloc[i:i+self.batch_size]
            
            # 批量处理
            tasks = [self._process_row(row) for _, row in batch.iterrows()]
            batch_results = await asyncio.gather(*tasks)
            
            # 收集结果
            for true_label, risk_score in batch_results:
                y_true.append(true_label)
                y_probs.append(risk_score)
        
        return y_true, y_probs

    async def _process_row(self, row) -> Tuple[int, float]:
        """处理单行数据"""
        result = await self.detector.analyze_text(row['text'])
        true_label = 1 if row['type'] == 'spam' else 0
        return true_label, result['risk_score']

    def _analyze_thresholds(self, y_true: List[int], y_probs: List[float]) -> Dict:
        """分析不同阈值性能"""
        results = {}
        for thresh in self.thresholds:
            y_pred = (np.array(y_probs) >= thresh).astype(int)
            report = classification_report(y_true, y_pred, output_dict=True)
            
            results[thresh] = {
                'accuracy': report['accuracy'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1': report['1']['f1-score'],
                'support': report['1']['support']
            }
            logger.info(f"阈值 {thresh} - F1: {results[thresh]['f1']:.4f}")
        return results

    def _save_results(self, results: Dict, y_true: List[int], y_probs: List[float]):
        """保存评估结果"""
        # 保存指标
        with open(self.output_dir / 'threshold_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(self.output_dir / 'roc_curve.png')
        plt.close()

async def main():
    optimizer = ThresholdOptimizer(
        data_path=DATA_DIR / "processed" / "test_data.csv",
        test_size=200
    )
    await optimizer.optimize()

if __name__ == "__main__":
    asyncio.run(main())