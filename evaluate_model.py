import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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


# 假设您的标签为'ham'和'spam'，并将其映射为0和1
label_mapping = {'ham': 0, 'spam': 1}
all_labels = list(label_mapping.values())

class ModelEvaluator:
    def __init__(
        self,
        data_file: str,
        n_splits: int = 2,
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
        random_state: int = 42,
        batch_size: int = 32
    ):
        self.data_file = data_file
        self.n_splits = n_splits
        self.thresholds = thresholds
        self.random_state = random_state
        self.batch_size = batch_size
        self.results = {}
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化检测器
        self.faiss_manager = FAISSManager(persist_path="data/faiss_index.bin")
        self.detector = FraudDetector(self.faiss_manager)

    async def evaluate_model(self):
        """执行完整的模型评估"""
        start_time = time.time()
        
        # 加载并预处理数据
        df = pd.read_csv(self.data_file)
        logger.info(f"原始数据分布:\n{df['type'].value_counts()}")
        
        # 数据采样（可选）
        sample_size = min(len(df), 6)  # 调整采样大小
        df = df.sample(n=sample_size, random_state=self.random_state)
        logger.info(f"采样后数据大小: {len(df)}")
        
        # 准备交叉验证
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        # 存储结果
        all_results = {t: [] for t in self.thresholds}
        all_predictions = {t: {'y_true': [], 'y_pred': [], 'y_score': []} for t in self.thresholds}
        
        # 执行交叉验证
        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['type']), 1):
            logger.info(f"\n处理折叠 {fold}/{self.n_splits}")
            
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            
            # 更新检测器的训练数据
            train_texts = train_data[train_data['type'] == 'spam']['text'].tolist()
            self.detector.load_scam_dataset(train_texts)
            
            # 对每个阈值进行评估
            for threshold in self.thresholds:
                results = await self._evaluate_threshold(test_data, threshold)
                all_results[threshold].append(results)
                
                # 收集预测结果
                all_predictions[threshold]['y_true'].extend(results['y_true'])
                all_predictions[threshold]['y_pred'].extend(results['y_pred'])
                all_predictions[threshold]['y_score'].extend(results['y_score'])
                
                # 输出当前阈值的性能
                metrics = classification_report(
                    results['y_true'],
                    results['y_pred'],
                    labels=all_labels,  # 指定所有可能的类别
                    output_dict=True,
                    zero_division=0  # 避免除零错误
                )
                logger.info(f"\n阈值 {threshold} 的性能指标:")
                logger.info(f"准确率: {metrics['accuracy']:.4f}")
                logger.info(f"精确率: {metrics['1']['precision']:.4f}")
                logger.info(f"召回率: {metrics['1']['recall']:.4f}")
                logger.info(f"F1分数: {metrics['1']['f1-score']:.4f}")

        # 生成评估报告
        self._generate_evaluation_report(all_results, all_predictions)
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"\n评估完成! 总耗时: {total_time/60:.2f} 分钟")
        
        return all_results, all_predictions

    async def _evaluate_threshold(self, test_data: pd.DataFrame, threshold: float) -> Dict:
        """评估单个阈值的性能"""
        y_true = []
        y_pred = []
        y_score = []
        
        # 批处理评估
        for i in tqdm(range(0, len(test_data), self.batch_size), desc=f"阈值 {threshold}"):
            batch = test_data.iloc[i:i+self.batch_size]
            
            for _, row in batch.iterrows():
                try:
                    result = await self.detector.analyze_text(row['text'])
                    confidence = result['risk_score']
                    
                    y_true.append(1 if row['type'] == 'spam' else 0)
                    y_pred.append(1 if confidence >= threshold else 0)
                    y_score.append(confidence)
                    
                    # 添加短暂延迟以防止API限制
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"处理错误: {row['text'][:50]}... - {str(e)}")
                    continue
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_score': y_score,
            'threshold': threshold
        }

    def _generate_evaluation_report(self, all_results: Dict, all_predictions: Dict):
        """生成详细的评估报告和可视化"""
        # 1. ROC曲线
        plt.figure(figsize=(10, 8))
        for threshold in self.thresholds:
            y_true = all_predictions[threshold]['y_true']
            y_score = all_predictions[threshold]['y_score']
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'阈值 {threshold} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('不同阈值的ROC曲线')
        plt.legend()
        plt.savefig(self.output_dir / 'roc_curves.png')
        plt.close()
        
        # 2. 混淆矩阵
        for threshold in self.thresholds:
            y_true = all_predictions[threshold]['y_true']
            y_pred = all_predictions[threshold]['y_pred']
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'阈值 {threshold} 的混淆矩阵')
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.savefig(self.output_dir / f'confusion_matrix_{threshold}.png')
            plt.close()
        
        # 3. 详细性能指标
        performance_summary = {}
        for threshold in self.thresholds:
            y_true = all_predictions[threshold]['y_true']
            y_pred = all_predictions[threshold]['y_pred']
            report = classification_report(y_true, y_pred, output_dict=True)
            
            performance_summary[str(threshold)] = {
                'accuracy': report['accuracy'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1': report['1']['f1-score'],
                'support': report['1']['support']
            }
        
        # 保存结果
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump({
                'performance_summary': performance_summary,
                'configuration': {
                    'n_splits': self.n_splits,
                    'thresholds': self.thresholds,
                    'random_state': self.random_state,
                    'batch_size': self.batch_size
                }
            }, f, indent=2)

async def main():
    evaluator = ModelEvaluator(
        data_file="data/processed/test_data.csv",
        n_splits=2,
        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
        batch_size=32
    )
    await evaluator.evaluate_model()

if __name__ == "__main__":
    asyncio.run(main()) 