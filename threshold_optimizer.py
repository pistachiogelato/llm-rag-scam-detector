import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import asyncio
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
from rag.rag_system import FraudDetector
from utils.faiss_manager import FAISSManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    def __init__(
        self,
        data_file: str,
        test_size: int = 200,
        thresholds: List[float] = [0.2, 0.3, 0.4],
        batch_size: int = 32
    ):
        self.data_file = data_file
        self.test_size = test_size
        self.thresholds = thresholds
        self.batch_size = batch_size
        self.output_dir = Path("threshold_optimization")
        self.output_dir.mkdir(exist_ok=True)
        
        self.faiss_manager = FAISSManager(persist_path="data/faiss_index.bin")
        self.detector = FraudDetector(self.faiss_manager)

    async def optimize_thresholds(self):
        """核心优化流程"""
        df = pd.read_csv(self.data_file)
        actual_test_size = min(len(df), self.test_size)
        test_data = df.sample(n=actual_test_size, random_state=42)
        logger.info(f"实际使用测试集大小: {actual_test_size} (总数据量: {len(df)})")
        
        y_true, y_probs = await self._get_predictions(test_data)
        
        results = {}
        for threshold in self.thresholds:
            y_pred = (np.array(y_probs) >= threshold).astype(int)
            report = classification_report(y_true, y_pred, output_dict=True)
            
            results[threshold] = {
                'accuracy': report['accuracy'],
                'precision': report.get('1', {}).get('precision', 0),
                'recall': report.get('1', {}).get('recall', 0),
                'f1': report.get('1', {}).get('f1-score', 0),
                'support': report.get('1', {}).get('support', 0)
            }
            
            logger.info(f"\n阈值 {threshold} 性能:")
            logger.info(f"准确率: {results[threshold]['accuracy']:.4f}")
            logger.info(f"精确率: {results[threshold]['precision']:.4f}")
            logger.info(f"召回率: {results[threshold]['recall']:.4f}")
            logger.info(f"F1分数: {results[threshold]['f1']:.4f}")

        self._save_results(results, y_true, y_probs)
        return results

    async def _get_predictions(self, test_data: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """批量获取预测概率（修复缺失的方法）"""
        y_true = []
        y_probs = []
        
        for i in tqdm(range(0, len(test_data), self.batch_size), desc="预测进度"):
            batch = test_data.iloc[i:i+self.batch_size]
            
            tasks = [self.detector.analyze_text(row['text']) for _, row in batch.iterrows()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"预测失败: {str(result)}")
                    continue
                
                true_label = 1 if batch.iloc[idx]['type'] == 'spam' else 0
                y_true.append(true_label)
                y_probs.append(result['risk_score'])
        
        return y_true, y_probs

    def _save_results(self, results: Dict, y_true: List[int], y_probs: List[float]):
        """保存结果"""
        with open(self.output_dir / 'threshold_performance.json', 'w') as f:
            json.dump(results, f, indent=2)
        
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
        data_file="data/processed/test_data.csv",
        test_size=200,
        thresholds=[0.2, 0.25, 0.3, 0.35, 0.4]
    )
    await optimizer.optimize_thresholds()

if __name__ == "__main__":
    asyncio.run(main())