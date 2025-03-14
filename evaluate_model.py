import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
from rag.rag_system import detect_and_generate_report, faiss_index
logging.basicConfig(level=logging.INFO)

class ModelEvaluator:
    def __init__(
        self,
        data_file: str,
        n_splits: int = 2,
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
        random_state: int = 42
    ):
        self.data_file = data_file
        self.n_splits = n_splits

        self.thresholds = thresholds
        self.random_state = random_state
        self.results = {}
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)

    def evaluate_model(self):
        """执行完整的模型评估"""
        df = pd.read_csv(self.data_file)
        print("数据分布:\n", df['label'].value_counts())
        print("正样本示例:\n", df[df['label']=='spam'].head(3))
        print("负样本示例:\n", df[df['label']=='ham'].head(3))
        df = df.sample(n=300, random_state=self.random_state)  # 全局只保留1万条样本
        # 准备交叉验证
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        # 存储所有结果
        all_results = {t: [] for t in self.thresholds}
        all_predictions = {t: {'y_true': [], 'y_pred': [], 'y_score': []} for t in self.thresholds}
        
        # 执行交叉验证
        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['label']), 1):
            logging.info(f"\nProcessing fold {fold}/{self.n_splits}")
            
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            
            # 对每个阈值进行评估
            for threshold in self.thresholds:
                results = self._evaluate_threshold(train_data, test_data, threshold)
                all_results[threshold].append(results)
                
                # 收集预测结果
                all_predictions[threshold]['y_true'].extend(results['y_true'])
                all_predictions[threshold]['y_pred'].extend(results['y_pred'])
                all_predictions[threshold]['y_score'].extend(results['y_score'])
        
        # 生成评估报告
        self._generate_evaluation_report(all_results, all_predictions)

    def _evaluate_threshold(self, train_data: pd.DataFrame, test_data: pd.DataFrame, threshold: float) -> Dict:
        """评估单个阈值的性能"""
        # 使用训练数据更新模型
        scam_texts = train_data[train_data['label'] == 'spam']['text'].tolist()
        
        y_true = []
        y_pred = []
        y_score = []
        
        # 评估测试数据
        for _, row in tqdm(test_data.iterrows(), desc=f"Threshold {threshold}"):
            try:
                result = detect_and_generate_report(row['text'], scam_texts, faiss_index)
                print(f"Text: {row['text'][:30]} | Confidence: {result['confidence']}")  # 添加这行
                y_true.append(1 if row['label'] == 'spam' else 0)
                y_pred.append(1 if result['confidence'] >= threshold else 0)
                y_score.append(result['confidence'])
                
                #time.sleep(0.1)  # 防止API限制
                
            except Exception as e:
                logging.error(f"Error processing: {row['text']} - {str(e)}")
                continue
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_score': y_score,
            'threshold': threshold
        }

    def _generate_evaluation_report(self, all_results: Dict, all_predictions: Dict):
        """生成详细的评估报告和可视化"""
        # 1. 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        for threshold in self.thresholds:
            y_true = all_predictions[threshold]['y_true']
            y_score = all_predictions[threshold]['y_score']
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Threshold {threshold} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Thresholds')
        plt.legend()
        plt.savefig(self.output_dir / 'roc_curves.png')
        plt.close()
        
        # 2. 性能指标比较
        metrics = {t: {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        } for t in self.thresholds}
        
        for threshold in self.thresholds:
            y_true = all_predictions[threshold]['y_true']
            y_pred = all_predictions[threshold]['y_pred']
            report = classification_report(y_true, y_pred, output_dict=True)
            
            metrics[threshold]['accuracy'].append(report['accuracy'])
            metrics[threshold]['precision'].append(report['1']['precision'])
            metrics[threshold]['recall'].append(report['1']['recall'])
            metrics[threshold]['f1'].append(report['1']['f1-score'])
        
        # 绘制性能指标对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Across Thresholds')
        
        for (metric, values), ax in zip(metrics[self.thresholds[0]].items(), axes.flat):
            for threshold in self.thresholds:
                ax.plot(metrics[threshold][metric], label=f'Threshold {threshold}')
            ax.set_title(f'{metric.capitalize()} Score')
            ax.set_xlabel('Fold')
            ax.set_ylabel('Score')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png')
        plt.close()
        
        # 3. 保存详细结果
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump({
                'thresholds': self.thresholds,
                'metrics': metrics
            }, f, indent=2)

if __name__ == "__main__":
    evaluator = ModelEvaluator(
        data_file="tests/data/processed_test_data.csv",
        n_splits=5,
        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]
    )
    evaluator.evaluate_model() 