import os
import logging
import psycopg2
from typing import Optional
from datetime import datetime, timedelta

# 数据库配置
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "scam_detector"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "070827"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def get_db_connection():
    """获取数据库连接"""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise

async def init_db():
    """初始化数据库表结构"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 创建主表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS realtime_scams (
                id SERIAL PRIMARY KEY,
                source VARCHAR(255) NOT NULL,
                scam_text TEXT NOT NULL,
                scam_type VARCHAR(255),
                confidence FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_scam_text 
            ON realtime_scams (scam_text);
            
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON realtime_scams (timestamp DESC);
        """)
        
        conn.commit()
        logging.info("Database initialized successfully")
        
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def load_scam_texts(days: int = 30, batch_size: int = 1000) -> list:
    """加载诈骗文本"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        cur.execute("""
            SELECT DISTINCT scam_text 
            FROM realtime_scams 
            WHERE scam_text IS NOT NULL 
            AND scam_text != '' 
            AND timestamp >= %s
            ORDER BY timestamp DESC
        """, (since,))
        
        texts = [row[0] for row in cur.fetchall()]
        logging.info(f"Loaded {len(texts)} texts from database")
        return texts
        
    except Exception as e:
        logging.error(f"Error loading scam texts: {e}")
        raise
    finally:
        cur.close()
        conn.close()

async def seed_initial_data():
    """初始化示例数据"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 检查是否需要添加示例数据
        cur.execute("SELECT COUNT(*) FROM realtime_scams")
        if cur.fetchone()[0] == 0:
            sample_data = [
                ("sample", "Your account has been compromised, verify now", "phishing", 0.9),
                ("sample", "Congratulations! You've won $1,000,000", "lottery", 0.95),
                ("sample", "Urgent: Transfer money to secure account", "financial", 0.85)
            ]
            
            cur.executemany("""
                INSERT INTO realtime_scams (source, scam_text, scam_type, confidence)
                VALUES (%s, %s, %s, %s)
            """, sample_data)
            
            conn.commit()
            logging.info("Sample data seeded successfully")
            
    except Exception as e:
        logging.error(f"Error seeding initial data: {e}")
        raise
    finally:
        cur.close()
        conn.close() 