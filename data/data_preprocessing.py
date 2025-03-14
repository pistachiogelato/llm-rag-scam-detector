import csv
import chardet
import os
import logging
import re
import pandas as pd
from typing import List, Dict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    with open(file_path, "rb") as f:
        return chardet.detect(f.read())["encoding"]

def normalize_text(text: str) -> str:
    """文本标准化处理"""
    return text.strip().lower()

def anonymize_text(text: str) -> str:
    """匿名化敏感信息"""
    text = re.sub(r'\d{10,}', '[PHONE]', text)  # 电话号码
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)  # 邮箱
    return re.sub(r'(http|https)://[^\s]+', '[URL]', text)  # 网址

def load_scam_data(csv_path: str) -> List[Dict[str, str]]:
    """
    加载CSV数据，保留原始结构
    输入文件格式：type,text
    """
    encoding = detect_encoding(csv_path)
    data = []
    
    try:
        # 检测编码
        with open(csv_path, "rb") as f:
            encoding = chardet.detect(f.read())['encoding']
        
        # 读取数据
        records = []
        with open(csv_path, 'r', encoding=encoding, errors='replace') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if len(row) < 2:
                    logging.warning(f"跳过不完整行 {row_num}: {row}")
                    continue
                
                # 按位置读取列（第一列类型，第二列文本）
                record = {
                    'type': row[0].strip().lower(),
                    'text': anonymize_text(normalize_text(row[1]))
                }
                
                # 过滤有效记录
                if record['type'] in ['spam', 'ham'] and len(record['text']) >= 10:
                    data.append(record)
                
                    
        logging.info(f"成功加载 {len(data)} 条记录来自 {csv_path}")
        return data
        
    except Exception as e:
        logging.error(f"加载文件 {csv_path} 失败: {str(e)}")
        return []

def merge_and_deduplicate(file_list: List[str]) -> pd.DataFrame:
    """合并多个数据集并进行高级去重"""
    merged_data = []
    
    # 1. 加载所有数据
    for file in file_list:
        merged_data.extend(load_scam_data(file))
    
    # 2. 转换为DataFrame
    df = pd.DataFrame(merged_data)
    
    if df.empty:
        raise ValueError("合并后的数据集为空，请检查输入文件")
    
    # 3. 高级去重（保留最新出现的记录）
    df = df.drop_duplicates(
        subset=['text'], 
        keep='last',  # 假设后加载的文件包含更新数据
    )
    
    # 4. 类型分布分析
    type_dist = df['type'].value_counts()
    logging.info(f"类型分布:\n{type_dist}")
    
    return df

if __name__ == "__main__":
    input_files = [
        "data/sms_spam.csv",
        "data/balanced_sms_dataset.csv"
    ]
    
    try:
        # 合并数据集
        merged_df = merge_and_deduplicate(input_files)
        
        # 保存结果
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "merged_sms.csv")
        
        merged_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL  # 确保含逗号的文本正确处理
        )
        
        # 输出统计信息
        logging.info(f"成功保存 {len(merged_df)} 条记录到 {output_path}")
        print("\n数据样例:")
        print(merged_df.sample(3).to_markdown(index=False))
        
    except Exception as e:
        logging.error(f"处理失败: {str(e)}", exc_info=True)