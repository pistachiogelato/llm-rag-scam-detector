# utils/data_processor.py
import os
import pandas as pd
import logging
import json
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from datetime import datetime
from enum import Enum

class LabelStrategy(Enum):
    """标签处理策略 (2025轻量级版)"""
    STRATIFIED = "stratified"  # 保持原始分布
    OVERSAMPLING = "oversampling"  # 基础过采样
    UNDERSAMPLING = "undersampling"  # 基础欠采样

class DataProcessor:
    def __init__(
        self,
        data_dir: str = "data",
        text_column: str = "text",
        label_column: str = "type",
        label_strategy: LabelStrategy = LabelStrategy.STRATIFIED
    ):
        """
        参数说明:
        - test_size: 小数据集自动设为0.3，大数据集可手动调整
        - label_strategy: 默认保持原始分布，避免不必要计算
        """
        self.data_dir = data_dir
        self.text_column = text_column
        self.label_column = label_column
        self.label_strategy = label_strategy
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.evaluation_dir = os.path.join(data_dir, "evaluation")
        self._ensure_directories()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def _ensure_directories(self):
        for dir_path in [self.raw_dir, self.processed_dir, self.evaluation_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
    def analyze_dataset(self, file_path: str) -> Dict:
        df = pd.read_csv(file_path)
        if self.label_column not in df.columns:
            raise ValueError(f"标签列 '{self.label_column}' 不存在于数据集中")
        
        # 转换数值类型为Python原生类型
        label_counts = df[self.label_column].value_counts()
        label_percentages = df[self.label_column].value_counts(normalize=True)
        text_lengths = df[self.text_column].str.len()
        
        stats = {
            'total_samples': int(len(df)),  # 转换为int
            'label_distribution': label_counts.astype(int).to_dict(),  # 转int
            'label_percentages': label_percentages.round(4).astype(float).to_dict(),  # 转float并保留4位小数
            'text_length_stats': {
                'mean': float(text_lengths.mean()),  # 转float
                'max': int(text_lengths.max()),      # 转int
                'min': int(text_lengths.min())
            }
        }
        
        analysis_path = os.path.join(
            self.evaluation_dir, 
            f'dataset_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(analysis_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logging.info(f"数据集分析结果已保存至 {analysis_path}")
        return stats
        
    def split_dataset(
        self, 
        input_file: str = "merged_sms.csv",
        test_size: Optional[float] = None,  # 自动判断数据集大小
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        智能分割策略：
        - 样本量<10k时：test_size=0.3
        - 样本量≥10k时：test_size=0.2
        - 始终使用分层抽样保持分布
        """
        input_path = os.path.join(self.processed_dir, input_file)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"关键文件缺失：{input_path}")
        
        df = pd.read_csv(input_path)
        logging.info(f"已加载数据：{len(df)}条样本 | 路径：{input_path}")
        
        # 自动设置测试集比例
        auto_test_size = 0.3 if len(df) < 10000 else 0.2
        final_test_size = test_size if test_size is not None else auto_test_size
        
        # 执行分层分割
        train_df, test_df = train_test_split(
            df,
            test_size=final_test_size,
            stratify=df[self.label_column],  # 强制分层保持分布
            random_state=random_state
        )
        
        # 智能保存数据集
        save_paths = {
            'train': os.path.join(self.processed_dir, 'train_data.csv'),
            'test': os.path.join(self.processed_dir, 'test_data.csv')
        }
        train_df.to_csv(save_paths['train'], index=False)
        test_df.to_csv(save_paths['test'], index=False)
        
        logging.info(f"训练集已保存 → {save_paths['train']} ({len(train_df)}条)")
        logging.info(f"测试集已保存 → {save_paths['test']} ({len(test_df)}条)")
        
        # 生成评估报告
        self.analyze_dataset(input_path)
        return train_df, test_df

    # 精简版数据加载方法
    def load_dataset(self, dataset_type: str) -> pd.DataFrame:
        """加载数据集（支持'train'/'test'）"""
        valid_types = ['train', 'test']
        if dataset_type not in valid_types:
            raise ValueError(f"数据集类型错误，可选：{valid_types}")
            
        path = os.path.join(self.processed_dir, f'{dataset_type}_data.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据集未找到：{path}")
            
        return pd.read_csv(path)
    
if __name__ == "__main__":
    # 自动处理数据
    processor = DataProcessor(label_column="type")
    train_df, test_df = processor.split_dataset()
    
    # 打印分布验证
    print("\n=== 数据分布 ===")
    print("训练集标签分布:")
    print(train_df['type'].value_counts(normalize=True))
    print("\n测试集标签分布:")
    print(test_df['type'].value_counts(normalize=True))