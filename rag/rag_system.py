import os, json, requests
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from openai import OpenAI
import numpy as np
import faiss
import logging
import re
import math
import random
from collections import defaultdict
import httpx
from typing import List, Optional, Dict

# ---------------------------
# 配置常量与环境初始化
# ---------------------------
DEFAULT_MODEL = "google/gemma-3-27b-it:free"  # 使用新模型
MAX_RETRIES = 3      # 重试次数
TIMEOUT = 10.0       # 超时时间（秒）
os.environ["FORCE_REGION"] = "default"  # 可选: asia/europe/default
load_dotenv()
DEBUG_MODE = True  # 通过环境变量控制更佳


def detect_region() -> str:
    forced = os.getenv("FORCE_REGION")
    return forced if forced else "default"

# ---------------------------
# 全局模型初始化
# ---------------------------
encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ---------------------------
# 文本编码函数
# ---------------------------

# def encode_text(text: str) -> np.ndarray:
#     """将输入文本转换为向量嵌入"""
#     return encoder.encode(text, convert_to_tensor=False).astype('float32')

def encode_text(text: str) -> np.ndarray:
    # 清洗特殊字符和URL
    clean_text = re.sub(r'http\S+|[@#]\w+', '', text)  # 移除URL和社交标签
    return encoder.encode(clean_text, convert_to_tensor=False).astype('float32')


def encode_text_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """批量编码文本"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = encoder.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# ---------------------------
# FAISS 索引构建与相似度检索
# ---------------------------
def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """构建 FAISS 索引以便高效的相似度检索"""
    dim = encoder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    print(f"Building index with dimension: {dim}")
    print(f"Input vectors shape: {vectors.shape}")
    index.add(vectors)
    return index

def retrieve_similar(query_text: str, index: faiss.IndexFlatL2, scam_texts: list, k: int = 3) -> list:
    """根据输入文本检索相似的诈骗文本"""
    if not scam_texts:
        logging.warning("No scam texts available for retrieval")
        return []
    query_vec = encode_text(query_text).reshape(1, -1)
    try:
        distances, indices = index.search(query_vec, k)
        logging.info(f"Query: {query_text}")
        logging.info(f"Distances: {distances}")
        logging.info(f"Indices: {indices}")
        # 距离阈值，距离越小代表越相似
        distance_threshold = 40.0
        valid_indices = [
            i for i, d in zip(indices[0], distances[0])
            if 0 <= i < len(scam_texts) and d < distance_threshold
        ]
        similar_texts = [scam_texts[i] for i in valid_indices]
        logging.info(f"Found {len(similar_texts)} similar texts")
        return similar_texts
    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        return []

# ---------------------------
# LLM 调用函数（使用 OpenRouter API 调用方式）
# ---------------------------
def llm_predict(
    prompt: str,
    system_prompt: Optional[str] = "You are a helpful assistant specialized in fraud detection analysis",
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    """
    使用 OpenRouter API 生成专业响应。
    根据示例，采用 URL：https://openrouter.ai/api/v1/chat/completions，
    模型为 "google/gemma-3-27b-it:free"，且 user 消息的 content 为列表，包含一个文本消息。
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if not (0 <= temperature <= 2):
        raise ValueError("Temperature must be between 0 and 2")
    
    api_key = os.getenv("API_TOKEN")
    if not api_key:
        logging.error("API_TOKEN environment variable is required")
        raise ValueError("Missing API_TOKEN")
    
    # 使用 OpenRouter API 的 URL
    router_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
         "Authorization": f"Bearer {api_key}",
         "Content-Type": "application/json",
         "HTTP-Referer": os.getenv("SITE_URL", "http://localhost"),
         "X-Title": os.getenv("SITE_NAME", "LLM-RAG Scam Detector")
    }
    payload = {
         "model": DEFAULT_MODEL,
         "messages": [
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": [{"type": "text", "text": prompt}]}
         ],
         "temperature": temperature,
         "max_tokens": max_tokens,
         "stream": False
    }
    try:
         r = requests.post(router_url, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
         if r.status_code != 200:
              logging.error(f"API请求失败，状态码: {r.status_code}, 响应: {r.text}")
              raise RuntimeError(f"API request failed with status {r.status_code}")
         data = r.json()
         logging.info(f"API请求成功，响应: {data}")
         return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
         logging.error(f"API连接失败 | 错误: {str(e)}")
         raise RuntimeError("API connection failed") from e

# ---------------------------
# 基于规则的增强检测：关键词规则与风险评估
# ---------------------------
KEYWORD_RULES = {
    "financial": {
        "patterns": [
            (r"\btransfer\b", 0.3),
            (r"\baccount\b", 0.2),
            (r"\$\d+", 0.4),
            (r"\bpayment\b", 0.3)
        ],
        "max_score": 0.6
    },
    "urgency": {
        "patterns": [
            (r"\burgen(t|cy)\b", 0.4),
            (r"\bimmediately\b", 0.3),
            (r"\blimited time\b", 0.3)
        ],
        "max_score": 0.5
    },
    "suspicious": {
        "patterns": [
            (r"\bverify\b", 0.4),
            (r"\bclick here\b", 0.3),
            (r"\bsecure\b", 0.2)
        ],
        "max_score": 0.4
    }
}

def calculate_pattern_score(text: str) -> float:
    """根据关键词规则计算得分"""
    
    text_lower = text.lower()
    total = 0.0
    for category, config in KEYWORD_RULES.items():
        category_score = 0.0
        for pattern, weight in config["patterns"]:
            if re.search(pattern, text_lower):
                category_score = max(category_score, weight)
        total += min(category_score, config["max_score"])
    print(f"分析文本: {text[:50]}")  # 查看被处理文本
    print(f"最终得分: {total}")      # 查看规则得分
    return min(total, 1.0)

def calibrate_confidence(raw_score: float) -> float:
    """使用 sigmoid 函数校准置信度，防止极端值"""
    return 1 / (1 + math.exp(-6 * (raw_score - 0.5)))
    #return 1 / (1 + math.exp(-8 * (raw_score - 0.6)))

def llm_prompt_template(text: str, cases: list) -> str:
    """构造用于 LLM 分析的提示模板"""
    return f"""As a cybersecurity expert, analyze this message:
    
Message: {text}

Evaluation Criteria:
1. Financial terminology prevalence (0-1)
2. Urgency pressure intensity (0-1)
3. Similarity to known scams (0-1)

Format Requirements:
- Final score = (Criteria1*0.4 + Criteria2*0.3 + Criteria3*0.3)
- Score MUST be between 0.0-1.0
- Include EXACTLY ONE line: Confidence: x.xx

Examples:
Criteria1: 0.8 (3 financial terms detected)
Criteria2: 0.6 ("urgent" found but no timeframe)
Criteria3: 0.7 (matches 2 phishing patterns)
Confidence: 0.72
"""

def generate_risk_report(confidence: float, indicators: list) -> str:
    """生成专业的风险评估报告"""
    risk_level = "High Risk" if confidence > 0.7 else "Medium Risk" if confidence > 0.4 else "Low Risk"
    return f"""
[Fraud Risk Assessment]
Risk Level: {risk_level} (Confidence: {confidence*100:.1f}%)

Key Indicators:
{"• " + "\n• ".join(indicators) if indicators else "No strong indicators detected"}

Recommended Actions:
1. Do NOT share sensitive information
2. Verify through official channels
3. Report suspicious messages
"""

# ---------------------------
# 核心检测与报告生成函数
# ---------------------------
def detect_and_generate_report(user_text: str, scam_texts: list, faiss_index: faiss.IndexFlatL2) -> dict:
    """
    综合使用关键词匹配、相似度检索和 LLM 分析生成诈骗检测报告
    """
    if DEBUG_MODE:
        print("\n=== 输入文本分析 ===")
        print(f"原始文本: {user_text}")
        #print(f"清洗后文本: {clean_text}")  # 需先在encode_text中返回clean_text
    try:
        # 1. 规则匹配得分
        pattern_score = calculate_pattern_score(user_text)
        # 2. 相似案例检索
        retrieved_cases = retrieve_similar(user_text, faiss_index, scam_texts)
        
        # 3. LLM 分析（仅在有足够可疑信息时调用）
        llm_confidence = None
        llm_analysis = ""
        use_llm = False  # 新增控制开关
        if use_llm and (pattern_score > 0.3 or retrieved_cases):
            try:
                prompt = llm_prompt_template(user_text, retrieved_cases[:2])
                llm_response = llm_predict(prompt)
                if match := re.search(r"Confidence:\s*([0-1]\.\d{2})", llm_response):
                    llm_confidence = float(match.group(1))
                else:
                    llm_confidence = 0.5
                logging.info(f"LLM Confidence: {llm_confidence}")
            except Exception as e:
                logging.error(f"LLM analysis failed: {str(e)}")
                llm_confidence = None
        
        # 4. 综合评分计算
        if llm_confidence is not None:
            raw_score = (pattern_score * 0.4) + (llm_confidence * 0.6)
        else:
            raw_score = min(pattern_score * 1.2, 0.85)
            raw_score += random.uniform(-0.05, 0.05)
        final_confidence = max(round(calibrate_confidence(raw_score), 2), 0.2)
        
        # 5. 生成风险指标描述
        indicators = []
        if pattern_score > 0.5:
            indicators.append("Suspicious keyword patterns detected")
        if retrieved_cases:
            indicators.append(f"Similar to {len(retrieved_cases)} known scams")
        
        return {
            "risk_detected": final_confidence > 0.5,
            "confidence": final_confidence,
            "report": generate_risk_report(final_confidence, indicators),
            "triggers": {
                "keywords": list(set(re.findall(
                    r"\b(" + "|".join(p for rules in KEYWORD_RULES.values() for p, _ in rules["patterns"]) + r")\b",
                    user_text, re.IGNORECASE
                )))
            }
        }
        
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        return {
            "risk_detected": False,
            "confidence": 0.0,
            "report": "System error occurred - contact support",
            "triggers": {}
        }

# ---------------------------
# 初始化 OpenAI 客户端（备用，目前未被 llm_predict 使用）
# ---------------------------
client = OpenAI(
    api_key=os.getenv("API_TOKEN"),
    base_url=os.getenv("API_URL", "https://api.deepseek.com/v1"),
    timeout=httpx.Timeout(TIMEOUT),
    http_client=httpx.Client(
        limits=httpx.Limits(max_connections=50)
    )
)

# ---------------------------
# 示例：FastAPI 应用（简化版）
# ---------------------------
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import psycopg2
import asyncio
from langdetect import detect
from contextlib import asynccontextmanager

class ScamRecord(BaseModel):
    id: int
    scam_text: str
    scam_type: str
    timestamp: str

class QueryRequest(BaseModel):
    text: str

# 这里假设 scam_texts 和 faiss_index 已初始化
scam_texts = [
    "urgent! your account needs verification. click here: http://bank-secure.com",
    "your account has been compromised, send money immediately to secure it",
    "please update your bank details to avoid suspension"
]
embeddings = encode_text_batch(scam_texts)
faiss_index = build_faiss_index(embeddings)

app = FastAPI(
    title="LLM-RAG Scam Detector API",
    description="Scam Detector API based on LLM-RAG with optimized DB and FAISS index updates.",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="Service status check")
def home():
    return {"message": "LLM-RAG Scam Detector API is running!"}

@app.post("/detect", summary="Detect potential scams in text")
def detect_scam(query: QueryRequest):
    text = query.text.lower()
    try:
        lang = detect(text)
        logging.info(f"Input text: {text}")
        logging.info(f"Detected language: {lang}")
        if not scam_texts or not faiss_index:
            raise HTTPException(status_code=503, detail="Service not ready. Please try again later.")
        if not text:
            raise HTTPException(status_code=400, detail="Empty text received.")
        rag_result = detect_and_generate_report(text, scam_texts, faiss_index)
        logging.info(f"Detection result: {rag_result}")
        return rag_result
    except Exception as e:
        logging.error(f"Error in detect_scam: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "scam_texts_count": len(scam_texts),
        "encoder_ready": encoder is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
