import os
import asyncio
import logging
import signal
import uuid
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from typing import List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Local module imports
from utils.faiss_manager import FAISSManager
from rag.rag_system import FraudDetector

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "scam_detector"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "070827"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# Global components
faiss_manager = FAISSManager(persist_path="data/faiss_index.bin")
fraud_detector = FraudDetector(faiss_manager)

class TextRequest(BaseModel):
    text: str

def load_training_data() -> List[str]:
    """Load and clean training data from CSV"""
    try:
        train_path = os.path.join('data', 'processed', 'train_data.csv')
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        df = pd.read_csv(train_path)
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].str.strip().str.lower()
        return df[df['text'].str.len().between(10, 1000)]['text'].tolist()
    except Exception as e:
        logger.error(f"Training data error: {str(e)}")
        return []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced initialization"""
    try:
        logger.info("Initializing service...")
        if os.path.exists(faiss_manager.persist_path):
            faiss_manager._load_index()
            logger.info("Loaded existing index")
        else:
            logger.info("Building new index")
            texts = load_training_data()
            if not texts:
                raise ValueError("Initial training data required")
            faiss_manager.build_index(texts)
        texts = load_training_data()
        fraud_detector.load_scam_dataset(texts)
        logger.info(f"Loaded {len(texts)} scam samples")
        yield
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        raise

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_scam(request: TextRequest):
    """诈骗检测端点"""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text received")
        result = await fraud_detector.analyze_text(text)
        # 转换所有 numpy 类型为 Python 内置类型
        risk_score = float(result["risk_score"])
        risk_detected = bool(risk_score > 0.5)
        response = {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "input_text": text,
            "risk_detected": risk_detected,
            "confidence": round(risk_score, 4),
            "risk_level": result["risk_level"],
            "pattern_analysis": {
                "total_score": float(result["pattern_analysis"]["total_score"]),
                "matched_patterns": [
                    str(pattern)
                    for category in result["pattern_analysis"]["categories"].values()
                    for pattern in category.get("matched_patterns", [])
                ]
            },
            "similar_texts": [
                {
                    "text": case["text"],
                    "similarity": float(case["similarity"])
                }
                for case in result["similar_cases"]
            ],
            "llm_analysis": {
                "risk_assessment": float(result["llm_analysis"].get("risk_score", 0)),
                "indicators": result["llm_analysis"].get("indicators", []),
                "recommendations": result["llm_analysis"].get("recommendations", []),
                "confidence": result["llm_analysis"].get("confidence", "Low")
            }
        }
        logger.info(f"Analysis completed for text: {text[:50]}...")
        return response
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """System health endpoint"""
    return {
        "status": "operational",
        "components": {
            "database": "connected",
            "index_status": {
                "vectors": faiss_manager.index.ntotal if faiss_manager.index else 0,
                "last_updated": datetime.utcnow().isoformat()
            },
            "model": "active"
        }
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
