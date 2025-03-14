import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import re
import logging
from langdetect import detect, LangDetectException
from typing import Tuple, Dict
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.df = None
        self.stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """加载并初步清理数据"""
        try:
            # 使用更安全的数据加载方式
            self.df = pd.read_csv(
                self.input_file,
                encoding='latin-1',  # 使用通用编码
                low_memory=False,    # 防止混合类型警告
                usecols=[0, 1],      # 只读取前两列
                names=['label', 'text'],  # 指定列名
                dtype={'label': str, 'text': str}  # 指定数据类型
            )
            
            # 标准化标签
            self.df['label'] = self.df['label'].str.lower()
            
            logging.info(f"Successfully loaded {len(self.df)} rows")
            self._update_stats('initial_load', len(self.df))
            return self.df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """增强的文本清理"""
        if not isinstance(text, str):
            return ""
        
        # 统一换行和制表符
        text = re.sub(r'[\r\n\t]+', ' ', text)
        
        # 保留基本URL结构但简化
        text = re.sub(r'http\S+', '[URL]', text)
        
        # 处理常见缩写
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        
        # 保留基本标点但规范化
        text = re.sub(r'[^\w\s.,!?@$%-]', ' ', text)
        
        # 处理重复字符
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def is_valid_english(self, text: str) -> bool:
        """增强的英文验证"""
        try:
            if not text or len(text.split()) < 3:
                return False
            
            # 检查英文字符比例
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            if english_chars / len(text) < 0.5:
                return False
            
            # 检查数字比例
            if len(re.findall(r'\d', text)) / len(text) > 0.3:
                return False
            
            # 语言检测
            return detect(text) == 'en'
        except:
            return False

    def preprocess(self) -> Tuple[pd.DataFrame, Dict]:
        """增强的预处理流程"""
        if self.df is None:
            self.load_data()
        
        # 1. 删除空值和极短文本
        self.df = self.df.dropna()
        self.df = self.df[self.df['text'].str.len() > 10]
        self._update_stats('after_basic_cleaning', len(self.df))
        
        # 2. 清理文本
        self.df['text'] = self.df['text'].apply(self.clean_text)
        
        # 3. 验证标签和文本
        valid_mask = (
            self.df['label'].isin(['spam', 'ham']) &
            self.df['text'].apply(self.is_valid_english)
        )
        self.df = self.df[valid_mask]
        self._update_stats('after_validation', len(self.df))
        
        # 4. 移除重复和近似重复
        self.df = self.df.drop_duplicates(subset=['text'])
        self._update_stats('after_deduplication', len(self.df))
        
        # 5. 平衡数据集
        min_count = min(
            len(self.df[self.df['label'] == 'spam']),
            len(self.df[self.df['label'] == 'ham'])
        )
        spam_df = self.df[self.df['label'] == 'spam'].sample(n=min_count, random_state=42)
        ham_df = self.df[self.df['label'] == 'ham'].sample(n=min_count, random_state=42)
        self.df = pd.concat([spam_df, ham_df])
        self._update_stats('final_balanced', len(self.df))
        
        # 生成详细统计
        self.generate_statistics()
        
        return self.df, self.stats

    def _update_stats(self, stage: str, count: int):
        """更新处理统计"""
        self.stats[stage] = {
            'total': count,
            'spam': len(self.df[self.df['label'] == 'spam']),
            'ham': len(self.df[self.df['label'] == 'ham'])
        }

    def generate_statistics(self):
        """生成详细统计和可视化"""
        output_dir = Path('evaluation_results')
        output_dir.mkdir(exist_ok=True)
        
        # 1. 文本长度分布
        plt.figure(figsize=(10, 6))
        self.df['text_length'] = self.df['text'].str.len()
        sns.boxplot(x='label', y='text_length', data=self.df)
        plt.title('Text Length Distribution by Label')
        plt.savefig(output_dir / 'text_length_distribution.png')
        plt.close()
        
        # 2. 处理阶段统计
        stages = list(self.stats.keys())
        spam_counts = [self.stats[s]['spam'] for s in stages]
        ham_counts = [self.stats[s]['ham'] for s in stages]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(stages))
        width = 0.35
        plt.bar(x - width/2, spam_counts, width, label='Spam')
        plt.bar(x + width/2, ham_counts, width, label='Ham')
        plt.xticks(x, stages, rotation=45)
        plt.title('Data Distribution Across Processing Stages')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'processing_stages_distribution.png')
        plt.close()

    def save_processed_data(self, output_file: str):
        """保存处理后的数据和统计信息"""
        # 保存处理后的数据
        self.df.to_csv(output_file, index=False, encoding='utf-8')
        
        # 保存统计信息
        stats_file = Path(output_file).parent / 'preprocessing_stats.json'
        import json
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logging.info(f"Processed data saved to {output_file}")
        logging.info(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    # 设置输出目录
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 运行预处理
    preprocessor = DataPreprocessor("tests/data/raw_test_data.csv")
    df, stats = preprocessor.preprocess()
    
    # 保存结果
    preprocessor.save_processed_data(output_dir / "tests/data/processed_test_data.csv")
    
    # 打印最终统计
    print("\nPreprocessing Statistics:")
    for stage, data in stats.items():
        print(f"\n{stage}:")
        print(f"Total: {data['total']}")
        print(f"Spam: {data['spam']}")
        print(f"Ham: {data['ham']}")