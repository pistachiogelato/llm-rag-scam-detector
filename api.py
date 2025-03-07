import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from typing import List
from datetime import datetime

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

# --- define data model ---
class ScamRecord(BaseModel):
    id: int
    scam_text: str
    scam_type: str
    detected_at: str  

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
        #  HTTP 500 ERROR CODE
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

# --- API port ---
@app.get("/", summary="Service status checks")
def home():
    
    return {"message": "LLM-RAG Scam Detector API is running!"}

@app.get("/scams", response_model=List[ScamRecord], summary="Get all the scam records")
def get_scams():
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, scam_text, scam_type, detected_at FROM realtime_scams;")
        rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying database: {e}")
    finally:
        cur.close()
        conn.close()

    
    return [
        {
            "id": row[0],
            "scam_text": row[1],
            "scam_type": row[2],
            "detected_at": row[3].isoformat() if isinstance(row[3], datetime) else str(row[3])
        }
        for row in rows
    ]

@app.post("/detect", summary="Detect scam texts")
def detect_scam(query: QueryRequest):
    """
    Simulate fraud detection interface. 
    Currently, for preliminary implementation, it is determined whether it is fraudulent information based on whether the input text contains specific keywords (such as "account", "verify", "lottery"). 
    The RAG module and LLM will be integrated for more accurate detection in the future.
    """
    text = query.text.lower()
    # Simple rule: If you include keywords, you will be considered a scam
    if "account" in text or "verify" in text or "lottery" in text:
        result = {"scam_detected": True, "scam_type": "phishing", "confidence": 0.9}
    else:
        result = {"scam_detected": False, "scam_type": "", "confidence": 0.0}
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
