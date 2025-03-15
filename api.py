import os
import asyncio
import logging
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
import json
import re

# Import FAISS manager and FraudDetector from project modules
from utils.faiss_manager import FAISSManager
from rag.rag_system import FraudDetector

# Configure logging for debugging purposes
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration parameters
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "scam_detector"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "070827"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# Initialize FAISS manager and FraudDetector instances
faiss_manager = FAISSManager(persist_path="data/faiss_index.bin")
fraud_detector = FraudDetector(faiss_manager)

# Pydantic model for text request payload
class TextRequest(BaseModel):
    text: str

# Function to load training data from CSV file
def load_training_data() -> List[str]:
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

# Lifespan context manager to initialize resources at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
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

# Initialize FastAPI app with lifespan for startup and shutdown events
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to detect scam text from the input payload
@app.post("/detect")
async def detect_scam(request: TextRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text received")
        logger.info(f"Received text for analysis: {text[:100]}...")
        result = await fraud_detector.analyze_text(text)
        
        risk_score = float(result["risk_score"])
        risk_detected = bool(risk_score > 0.5)
        report = result.get("report", "No detailed report available")
        if isinstance(report, str):
            report = report.replace('\\', '\\\\')
        
        matched_patterns = [
            str(pattern)
            for category in result["pattern_analysis"]["categories"].values()
            for pattern in category.get("matched_patterns", [])
        ]
        
        response = {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "input_text": text,
            "risk_detected": risk_detected,
            "confidence": round(risk_score, 4),
            "risk_level": result["risk_level"],
            "report": report,
            "patterns": matched_patterns,  # For frontend compatibility
            "pattern_analysis": {
                "total_score": float(result["pattern_analysis"]["total_score"]),
                "matched_patterns": matched_patterns
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

# Health check endpoint to monitor API status
@app.get("/health")
async def health_check():
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
