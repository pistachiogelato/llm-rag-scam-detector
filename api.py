import os
import time
import signal
import logging
import csv
import requests
import numpy as np
import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta
from typing import List
from contextlib import asynccontextmanager


import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import chardet
import asyncio
from langdetect import detect

# Import RAG functions from our custom module
from rag.rag_system import detect_and_generate_report, encode_text, build_faiss_index

# Set up logging
logging.basicConfig(level=logging.INFO)

# Environment variables for database configuration
DB_NAME = os.getenv("DB_NAME", "scam_detector")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "070827")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Global state variables
scam_texts = []
faiss_index = None
# Use a lighter model for faster CPU inference
encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embedding_dim = encoder.get_sentence_embedding_dimension()  # Dimension for 'paraphrase-MiniLM-L6-v2'

# Configure FAISS index using HNSW for better scalability
faiss_index = faiss.IndexHNSWFlat(embedding_dim, 32)

# Data models for API
class ScamRecord(BaseModel):
    id: int
    scam_text: str
    scam_type: str
    timestamp: str

class QueryRequest(BaseModel):
    text: str

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

def chunked_iterable(iterable, chunk_size):
    """
    Yield successive chunks of size chunk_size from iterable.
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]

def load_scam_texts_from_db(batch_size=1000, days=30) -> List[str]:
    """
    Stream scam texts from the database efficiently by using fetchmany.

    Only scam texts within the last 'days' days are retrieved.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    query = """
        SELECT scam_text FROM realtime_scams
        WHERE scam_text IS NOT NULL AND scam_text != ''
        AND timestamp >= %s
        ORDER BY timestamp DESC
        LIMIT 10000
    """
    cur.execute(query, (since,))
    
    texts = []
    while True:
        batch = cur.fetchmany(batch_size)
        if not batch:
            break
        texts.extend([row[0] for row in batch])
    logging.info(f"Loaded {len(texts)} scam texts from database.")
    cur.close()
    conn.close()
    return texts

# Data source handling for URLhaus data
def fetch_urlhaus_data() -> List[str]:
    """
    Fetch malicious URLs from URLhaus.
    """
    url = "https://urlhaus.abuse.ch/downloads/text/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # Filter out comment lines and empty lines
        return [line.strip() for line in response.text.splitlines() if line.strip() and not line.startswith("#")]
    except Exception as e:
        logging.error(f"Error fetching URLhaus data: {e}")
        return []

def seed_urlhaus_data():
    """
    Seed URLhaus data into the database with timeout control and batch processing.
    """
    start_time = time.time()
    max_duration = 300  # 5 minutes timeout
    try:
        urls = list(set(fetch_urlhaus_data()))
        if not urls:
            logging.info("No URL records to process.")
            return
        records = [("URLhaus", url, "malware", 0.9) for url in urls]
        # Deduplicate records in memory
        seen = set()
        deduped_records = []
        for r in records:
            if r[1] not in seen:
                seen.add(r[1])
                deduped_records.append(r)
        total = len(deduped_records)
        logging.info(f"Preparing to insert {total} URL records into the database.")
        
        conn = get_db_connection()
        cur = conn.cursor()
        chunk_size = 5000
        for i, chunk in enumerate(chunked_iterable(deduped_records, chunk_size)):
            if time.time() - start_time > max_duration:
                logging.warning("Insertion timeout reached, stopping batch insertion.")
                break
            try:
                query = sql.SQL("""
                    INSERT INTO realtime_scams (source, scam_text, scam_type, confidence)
                    VALUES {}
                    ON CONFLICT (scam_text) DO NOTHING
                """).format(sql.SQL(',').join([sql.Literal(r) for r in chunk]))
                cur.execute(query)
                conn.commit()
                logging.debug(f"Inserted batch {i+1} / {total//chunk_size+1}.")
            except Exception as e:
                logging.error(f"Batch {i} insertion failed: {e}", exc_info=True)
                conn.rollback()
        duration = time.time() - start_time
        logging.info(f"URL data insertion completed in {duration:.2f}s, rate: {total/duration:.1f} records/s")
    except Exception as e:
        logging.error(f"URL data seeding failed: {e}", exc_info=True)
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

def seed_sms_scams():
    """
    Seed SMS scam data from a CSV file into the database.
    """
    if not os.path.exists("data/sms_spam.csv"):
        logging.warning("SMS spam data file not found.")
        return
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        with open("data/sms_spam.csv", "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'latin1'
            logging.info(f"Detected file encoding: {encoding}")
        with open("data/sms_spam.csv", encoding=encoding) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            records = [("sms", row[1], row[0], 0.8) for row in reader if row and row[0].strip().lower() == "spam"]
        if records:
            for chunk in chunked_iterable(records, 1000):
                cur.executemany(
                    "INSERT INTO realtime_scams (source, scam_text, scam_type, confidence) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (scam_text) DO NOTHING",
                    chunk
                )
                conn.commit()
            logging.info(f"Inserted {len(records)} SMS scam records.")
    except Exception as e:
        logging.error(f"Error seeding SMS scams data: {e}", exc_info=True)
        conn.rollback()
    finally:
        cur.close()
        conn.close()

# FAISS index management
def update_faiss_index():
    """
    Update the FAISS index by loading recent scam texts and encoding them.
    Uses incremental update to reduce CPU load.
    """
    global scam_texts, faiss_index
    new_texts = load_scam_texts_from_db(days=30)  # Only load texts from the last 30 days
    if not new_texts:
        logging.warning("No new data available for FAISS index update.")
        return
    embedding_dim = encoder.get_sentence_embedding_dimension()
    faiss_index = faiss.IndexHNSWFlat(embedding_dim, 32)

    # Encode the texts into embeddings
    embeddings = encoder.encode(new_texts, convert_to_numpy=True)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Optionally, if you wish to preserve an existing index, you could add new embeddings:
    # Here we rebuild the index for simplicity.
    #faiss_index.reset()  # Clear the existing index
    faiss_index.add(embeddings)
    
    scam_texts = new_texts  # Update the global scam_texts variable
    logging.info(f"FAISS index updated with {len(new_texts)} entries.")
    #logging.info(f"FAISS index updated with {faiss_index.ntotal} entries.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    优化的应用生命周期管理
    """
    global scam_texts, faiss_index
    try:
        # 1. 快速启动必要服务
        logging.info("Starting application...")
        
        # 2. 异步初始化数据库
        async def init_db():
            conn = await asyncio.to_thread(get_db_connection)
            cur = conn.cursor()
            await asyncio.to_thread(
                cur.execute,
                """
                CREATE TABLE IF NOT EXISTS realtime_scams (
                    id SERIAL PRIMARY KEY,
                    source VARCHAR(255),
                    scam_text TEXT UNIQUE,
                    scam_type VARCHAR(255),
                    confidence FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await asyncio.to_thread(conn.commit)
            cur.close()
            conn.close()
        
        # 3. 异步加载初始数据
        async def init_data():
            global scam_texts, faiss_index
            # 只加载最近7天数据用于快速启动
            initial_texts = await asyncio.to_thread(
                load_scam_texts_from_db, 
                batch_size=1000, 
                days=7
            )
            if initial_texts:
                embeddings = await asyncio.to_thread(
                    encoder.encode,
                    initial_texts,
                    convert_to_numpy=True
                )
                faiss_index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
                faiss_index.add(embeddings)
                scam_texts = initial_texts
                logging.info(f"Initial index built with {len(initial_texts)} entries")
        
        # 4. 创建后台任务
        background_tasks = [
            asyncio.create_task(init_db()),
            asyncio.create_task(init_data())
        ]
        
        # 5. 等待基本服务启动
        await asyncio.gather(*background_tasks)
        
        # 6. 启动完整数据加载
        asyncio.create_task(load_complete_data())
        
        logging.info("Application startup complete")
        yield
        
    except Exception as e:
        logging.error(f"Startup error: {e}")
        yield
    finally:
        logging.info("Shutting down...")

async def async_load_initial_data():
    """
    Asynchronously load initial data after API starts
    """
    try:
        # 先加载一小部分数据用于快速启动
        await asyncio.to_thread(load_scam_texts_from_db, days=7)  # 只加载最近7天数据
        update_faiss_index()
        
        # 后台加载完整数据
        asyncio.create_task(load_complete_data())
    except Exception as e:
        logging.error(f"Error in initial data loading: {e}")

async def load_complete_data():
    """
    Load complete dataset in background
    """
    try:
        await asyncio.sleep(5)  # 等待服务完全启动
        await asyncio.to_thread(seed_sms_scams)
        await asyncio.to_thread(seed_urlhaus_data)
        await asyncio.to_thread(update_faiss_index)
        logging.info("Complete data loading finished")
    except Exception as e:
        logging.error(f"Error in complete data loading: {e}")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="LLM-RAG Scam Detector API",
    description="Scam Detector API based on LLM-RAG with optimized DB and FAISS index updates.",
    version="1.0",
    lifespan=lifespan
)

#Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API endpoints
@app.get("/", summary="Service status check")
def home():
    """
    Return a status message indicating the service is running.
    """
    return {"message": "LLM-RAG Scam Detector API is running!"}

@app.get("/scams", response_model=List[ScamRecord], summary="Retrieve all scam records")
def get_scams():
    """
    Retrieve all scam records from the database.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, scam_text, scam_type, timestamp FROM realtime_scams;")
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "scam_text": row[1],
                "scam_type": row[2],
                "timestamp": row[3].isoformat() if isinstance(row[3], datetime) else str(row[3])
            }
            for row in rows
        ]
    except Exception as e:
        logging.error(f"Error in /scams endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error querying database: {e}")
    finally:
        cur.close()
        conn.close()

'''
@app.get("/debug_index", summary="Return FAISS index size for debugging")
def debug_index():
    return {"faiss_index_size": faiss_index.ntotal}
'''

@app.post("/detect", summary="Detect potential scams in text")
def detect_scam(query: QueryRequest):
    """
    增强的诈骗检测端点
    """
    text = query.text.lower()
    try:
        # 检测文本语言
        lang = detect(text)
        logging.info(f"Input text: {text}")
        logging.info(f"Detected language: {lang}")
        
        # 检查数据准备情况
        logging.info(f"FAISS index size: {faiss_index.ntotal if faiss_index else 0}")
        logging.info(f"Scam texts count: {len(scam_texts) if scam_texts else 0}")
        
        if not scam_texts or not faiss_index:
            raise HTTPException(
                status_code=503,
                detail="Service not ready. Please try again later."
            )
        
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Empty text received."
            )
        
        # 获取RAG检测结果
        rag_result = detect_and_generate_report(text, scam_texts, faiss_index)
        
        # 记录检测结果
        logging.info(f"Detection result: {rag_result}")
        
        return rag_result
        
    except Exception as e:
        logging.error(f"Error in detect_scam: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/update_index", summary="Manually update FAISS index")
def update_index():
    """
    Manually trigger an update of the FAISS index.
    """
    update_faiss_index()
    return {"message": "FAISS index updated successfully."}

# Check for required database indexes before starting the server
def check_index():
    """
    Verify that the required unique index exists in the database.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'realtime_scams' 
            AND indexname = 'idx_scam_text_unique'
        """)
        if not cur.fetchone():
            logging.critical("Missing unique index idx_scam_text_unique!")
            raise RuntimeError("Database indexes not properly configured")
    finally:
        cur.close()
        conn.close()

# Graceful shutdown handler
def shutdown(signum, frame):
    """
    Handle shutdown signals and clean up resources.
    """
    logging.info("Received shutdown signal, cleaning up...")
    if faiss_index:
        faiss_index.reset()
    os._exit(0)

@app.get("/health")
async def health_check():
    """
    服务健康检查
    """
    return {
        "status": "healthy",
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "scam_texts_count": len(scam_texts) if scam_texts else 0,
        "encoder_ready": encoder is not None
    }

if __name__ == "__main__":
    def check_index():
        """
        Verify required database indexes exist
        """
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'realtime_scams' 
                AND indexname = 'idx_scam_text_unique'
            """)
            if not cur.fetchone():
                logging.critical("Missing unique index idx_scam_text_unique!")
                raise RuntimeError("Database indexes not properly configured")
        finally:
            cur.close()
            conn.close()
    
    # 执行索引检查
    check_index()
    
    def shutdown(signum, frame):
        """
        Graceful shutdown handler
        """
        logging.info("Received shutdown signal, cleaning up...")
        if faiss_index:
            faiss_index.reset()
        os._exit(0)
    
    # 设置信号处理
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # 启动应用
    uvicorn.run(app, host="127.0.0.1", port=8000)
