import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from typing import List
from datetime import datetime
import logging
from rag.rag_system import detect_and_generate_report
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# environment configuration
DB_NAME = os.getenv("DB_NAME", "scam_detector")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "070827")  
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# --- FastAPI  ---
app = FastAPI(
    title="LLM-RAG Scam Detector API",
    description=" Scam Detector API base on LLM-RAG",
    version="1.0"
)

# Initialize the embedding model
encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# --- define data model ---
class ScamRecord(BaseModel):
    id: int
    scam_text: str
    scam_type: str
    timestamp: str  

# request model
class QueryRequest(BaseModel):
    text: str

# --- connect to database ---
def get_db_connection():
    
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logging.error("Database connection error: %s", e)
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

def encode_text(text: str) -> np.ndarray:
    """
    Convert input text to a vector embedding.
    :param text: The input text.
    :return: A numpy array representing the text embedding.
    """
    # Note: convert_to_tensor=False returns a numpy array by default
    vector = encoder.encode(text, convert_to_tensor=False).astype('float32')
    return vector



def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS index for the given set of vectors.
    :param vectors: A numpy array of shape (n_samples, vector_dim)
    :return: A FAISS index.
    """
    # Determine the dimension from vectors
    d = vectors.shape[1]
    # Create a flat (brute-force) L2 index
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index

# --- API port ---
@app.get("/", summary="Service status checks")
def home():
    
    return {"message": "LLM-RAG Scam Detector API is running!"}

@app.get("/scams", response_model=List[ScamRecord], summary="Get all scam records")
def get_scams():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, scam_text, scam_type, timestamp FROM realtime_scams;")
        rows = cur.fetchall()
    except Exception as e:
        # Log error details and raise HTTPException
        logging.error("Error in /scams endpoint: %s", e)
        raise HTTPException(status_code=500, detail=f"Error querying database: {e}")
    finally:
        cur.close()
        conn.close()
    
    # Convert datetime to ISO string
    return [
        {
            "id": row[0],
            "scam_text": row[1],
            "scam_type": row[2],
            "timestamp": row[3].isoformat() if isinstance(row[3], datetime) else str(row[3])
        }
        for row in rows
    ]


@app.post("/detect", summary="Detect Scam Texts")
def detect_scam(query: QueryRequest):
    text = query.text.lower()
    scam_texts = [
        "http://mitrashopee.com/",
        "https://pub-d4ba3ddf19254ddeadc01c5590e107d5.r2.dev/index.html",
        "http://yonggevl3k.la-xrdn.workers.dev/",
        "https://mkup-a.kroin.top/"
    ]
    # Build FAISS index for demo
    try:
        vectors = np.array([encode_text(s) for s in scam_texts])
        index = build_faiss_index(vectors)
    except Exception as e:
        logging.error("Error building FAISS index: %s", e)
        raise HTTPException(status_code=500, detail="Error processing scam data.")
    
    
    # --- Rule-based detection ---
    keyword_weights = {
        "account": 0.3,
        "verify": 0.3,
        "lottery": 0.3,
        "immediately": 0.2,
        "urgent": 0.2,
        "now": 0.1
    }
    score = 0.0
    for keyword, weight in keyword_weights.items():
        if keyword in text:
            score += weight
    rule_confidence = min(score, 1.0)
    
    # --- RAG module detection ---
    try:
        rag_result = detect_and_generate_report(text, scam_texts, index)
    except Exception as e:
        logging.error("Error in RAG module: %s", e)
        raise HTTPException(status_code=500, detail="Error processing RAG detection.")

    # Combine the results: for example, use the maximum confidence and merge reports.
    final_confidence = max(rule_confidence, rag_result.get("confidence", 0))
    final_result = {
        "scam_detected": rag_result.get("scam_detected"),
        "scam_type": rag_result.get("scam_type"),
        "confidence": final_confidence,
        "report": rag_result.get("report"),
        "retrieved_cases": rag_result.get("retrieved_cases"),
        "rule_based_confidence": rule_confidence  # optional: include rule score for参考
    }
    return final_result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
