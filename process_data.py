import pandas as pd
import numpy as np
import logging
import os
import psycopg2
from psycopg2 import sql
import asyncio
import requests
from typing import List, Tuple
from dotenv import load_dotenv
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Environment variables for database configuration
DB_NAME = os.getenv("DB_NAME", "scam_detector")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "070827")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Database utilities
def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    try:
        return psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

def preprocess_text(text: str) -> str:
    """文本预处理"""
    import re
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点
    text = re.sub(r'\d+', '', text)       # 移除数字
    return text.strip().lower()

def load_sms_dataset(file_path: str = "data/merged_sms.csv") -> pd.DataFrame:
    """加载SMS数据集"""
    try:
        # 原始数据集有5列：v1,v2,Unnamed: 2,Unnamed: 3,Unnamed: 4，取前两列
        df = pd.read_csv(file_path, encoding='latin-1', usecols=[0, 1])
        df.columns = ['label', 'text']
        
        # 清理空值
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].apply(preprocess_text)
        
        logging.info(f"Loaded {len(df)} SMS records")
        return df
    except Exception as e:
        logging.error(f"Error loading SMS dataset: {e}")
        raise

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """验证数据集结构"""
    required_columns = ['label', 'text']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    if df['text'].isnull().sum() > 0:
        raise ValueError("Text column contains null values")
    return df

def generate_variations(text: str, label: str) -> List[str]:
    """使用LLM生成文本变体"""
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        raise ValueError("API_TOKEN not found in environment variables")
    
    prompt = f"""
    Task: Generate 3 phishing/spam message variations with high realism

    Original Message: {text}

    Technical Requirements (for spam/phishing):
    1. Must include at least 2 of these traits:
    - Urgency triggers ("Act now", "within 24 hours")
    - Fake rewards ("unclaimed prize", "account issue")
    - Authority impersonation ("Bank alert", "Official notice")
    - Sensitive action requests ("click link", "verify code")
    2. Use URL obfuscation techniques:
    - Replace actual links with "[LINK]" or "visit [site]"
    - Add typos in domain names (e.g., "amaz0n.com")
    3. Include 1-2 subtle grammar errors mimicking real scams
    4. Maintain natural language patterns (e.g., excessive punctuation)

    Examples of Good Output (Spam):
    1. [FedEx] Your package #8891 requires customs clearance at [LINK]
    2. Security Alert: Unusual login from new device. Confirm now: bit[.]ly/2YhGx3
    3. Final Notice: Your $120 refund expires today. Claim at irs-refund[.]net
    
    Format: Return only the variations, one per line.
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Scam Detector"
        }

        payload = {
            "model": "deepseek-chat",  # 使用免费模型
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }

        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload
        )
        
        if not response.ok:
            logging.error(f"OpenRouter API Error: {response.status_code}")
            logging.error(f"Response content: {response.text}")
            response.raise_for_status()

        result = response.json()
        logging.info(f"LLM Response: {result}")
        variations = result['choices'][0]['message']['content'].strip().split('\n')
        return [var.strip() for var in variations if var.strip()]
    except Exception as e:
        logging.error(f"Error generating variations: {e}")
        return []

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """平衡数据集，使 spam 与 ham 数量一致"""
    spam = df[df['label'] == 'spam']
    ham = df[df['label'] == 'ham']
    
    # 目标数量为 ham 的数量，避免spam扩充过多（根据实际情况可调整）
    target_count = len(ham)
    logging.info(f"Target count per class: {target_count}")
    
    # 如果 spam 数量不足，生成变体
    if len(spam) < target_count:
        new_spam_texts = []
        sample_needed = target_count - len(spam)
        logging.info(f"Generating {sample_needed} variations for spam...")
        # 这里建议批量抽样和生成，防止过多API调用
        for _, row in spam.sample(n=sample_needed, replace=True).iterrows():
            variations = generate_variations(row['text'], 'spam')
            if variations:
                new_spam_texts.extend(variations)
        new_spam_df = pd.DataFrame({
            'label': ['spam'] * len(new_spam_texts),
            'text': new_spam_texts
        })
        spam = pd.concat([spam, new_spam_df])
    
    # 如果 ham 数量不足（通常ham较多），这里一般不需要扩充
    if len(ham) < target_count:
        new_ham_texts = []
        sample_needed = target_count - len(ham)
        logging.info(f"Generating {sample_needed} variations for ham...")
        for _, row in ham.sample(n=sample_needed, replace=True).iterrows():
            variations = generate_variations(row['text'], 'ham')
            if variations:
                new_ham_texts.extend(variations)
        new_ham_df = pd.DataFrame({
            'label': ['ham'] * len(new_ham_texts),
            'text': new_ham_texts
        })
        ham = pd.concat([ham, new_ham_df])
    
    balanced_df = pd.concat([spam, ham]).sample(frac=1).reset_index(drop=True)
    logging.info(f"Balanced dataset - Spam: {len(spam)}, Ham: {len(ham)}")
    return balanced_df

def save_to_database(df: pd.DataFrame):
    """保存处理后的数据到数据库"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        records = [
            (
                'sms',
                row['text'],
                'spam' if row['label'] == 'spam' else 'ham',
                0.9 if row['label'] == 'spam' else 0.1
            )
            for _, row in df.iterrows()
        ]
        cur.executemany("""
            INSERT INTO realtime_scams 
                (source, scam_text, scam_type, confidence)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (scam_text) DO NOTHING
        """, records)
        conn.commit()
        logging.info(f"Saved {len(df)} records to database")
    except Exception as e:
        conn.rollback()
        logging.error(f"Database error: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def save_to_csv(df: pd.DataFrame, file_path: str):
    """将处理后的数据保存到 CSV 文件"""
    try:
        df.to_csv(file_path, index=False, encoding='utf-8')
        logging.info(f"Saved balanced dataset to {file_path}")
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
        raise

def main():
    """主处理流程"""
    try:
        logging.info("Loading SMS dataset...")
        df = load_sms_dataset()
        df = validate_dataset(df)
        initial_stats = df['label'].value_counts()
        logging.info(f"Initial dataset statistics:\n{initial_stats}")
        
        logging.info("Balancing dataset...")
        balanced_df = balance_dataset(df)
        final_stats = balanced_df['label'].value_counts()
        logging.info(f"Final dataset statistics:\n{final_stats}")
        
        # 将处理后的数据保存到 data/ 文件夹下
        output_csv_path = "data/balanced_sms_dataset.csv"
        balanced_df.to_csv(output_csv_path, index=False)
        logging.info(f"Balanced dataset saved to {output_csv_path}")
        
        # 可选：保存到数据库
        logging.info("Saving to database...")
        save_to_database(balanced_df)
        
        logging.info("Data processing completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
