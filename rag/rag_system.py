from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import numpy as np
import faiss
import os
import logging
import faiss
from dotenv import load_dotenv
from typing import List
import re 
from collections import defaultdict

load_dotenv()

# Initialize the embedding model globally
encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def encode_text(text: str) -> np.ndarray:
    """
    Convert input text into a vector embedding using the SentenceTransformer model.
    
    Args:
        text (str): The input text to encode.
    
    Returns:
        np.ndarray: A numpy array representing the text embedding.
    """
    return encoder.encode(text, convert_to_tensor=False).astype('float32')

def encode_text_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    批量编码文本
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = encoder.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS index for efficient similarity search over a set of vectors.
    
    Args:
        vectors (np.ndarray): Array of shape (n_samples, vector_dim) containing embeddings.
    
    Returns:
        faiss.IndexFlatL2: A FAISS index for L2 distance-based retrieval.
    """
    #dim = vectors.shape[1]  # Vector dimension
    #dim = vectors.shape[1] if vectors.shape[1] else 384
    dim = encoder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    print(f"Building index with dimension: {dim}")
    print(f"Input vectors shape: {vectors.shape}")
    index.add(vectors)
    return index

def retrieve_similar(query_text: str, index: faiss.IndexFlatL2, scam_texts: list, k: int = 3) -> list:
    """
    优化相似度检索函数
    """
    if not scam_texts:
        logging.warning("No scam texts available for retrieval")
        return []
        
    query_vec = encode_text(query_text)
    query_vec = query_vec.reshape(1, -1)
    
    try:
        # 修改相似度阈值
        distances, indices = index.search(query_vec, k)
        logging.info(f"Query: {query_text}")
        logging.info(f"Distances: {distances}")
        logging.info(f"Indices: {indices}")
        
        # 设置距离阈值，距离越小越相似
        distance_threshold = 2.0
        valid_indices = [
            i for i, d in zip(indices[0], distances[0]) 
            if i >= 0 and i < len(scam_texts) and d < distance_threshold
        ]
        
        similar_texts = [scam_texts[i] for i in valid_indices]
        logging.info(f"Found {len(similar_texts)} similar texts")
        return similar_texts
    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        return []

def llm_predict(prompt: str) -> str:
    """
    Generate a response from the LLM API based on the provided prompt.
    
    Args:
        prompt (str): The input prompt for the LLM.
    
    Returns:
        str: The generated text from the LLM.
    
    Raises:
        Exception: If API token is missing or LLM call fails.
    """
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        logging.error("API_TOKEN is missing!")
        raise Exception("API_TOKEN is not set in environment variables.")

    try:
        client = InferenceClient(provider="sambanova", api_key=api_token)
        messages = [{"role": "user", "content": prompt}]
        logging.info(f"Sending request to LLM: {prompt[:100]}...")  #debug
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=messages,
            max_tokens=150
        )
        logging.info("LLM request successful")
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"LLM API error: {str(e)}", exc_info=True)  # debug
        raise

def generate_report(user_text: str, retrieved_cases: list) -> str:
    """
    Generate a scam detection report using the user text and retrieved cases.
    
    Args:
        user_text (str): The user-provided text to analyze.
        retrieved_cases (list): List of similar scam examples retrieved.
    
    Returns:
        str: A detailed report from the LLM.
    """
    prompt = (
        f"User message: {user_text}\n"
        f"Related scam examples:\n" + "\n".join(retrieved_cases) +
        "\n\nBased on the above, analyze if the user's message is a scam, explain why, "
        "and provide recommendations."
    )
    return llm_predict(prompt)

def detect_and_generate_report(user_text: str, scam_texts: list, faiss_index: faiss.IndexFlatL2) -> dict:
    """
    优化诈骗检测和报告生成
    """
    try:
        print(f"Processing text: {user_text}")
        print(f"Available scam texts: {len(scam_texts)}")
        
         # 1. 改进关键词权重系统
        keyword_patterns = {
        # 金额相关模式（保持原有）
        "money_request": {
            "patterns": [
                (r"send.*?\$?\d+", 0.7),
                (r"send.*?money", 0.7),
                (r"need.*?\$?\d+", 0.6),
                (r"给我.*?钱", 0.7),
                (r"转账.*?\d+", 0.7)
            ],
            "category": "financial"
        },
    
        # 紧急程度相关（保持原有）
        "urgency": {
            "patterns": [
                (r"urgent|immediately|asap", 0.5),
                (r"紧急|立即|马上", 0.5)
            ],
            "category": "urgent"
        },

        # 新增通用诈骗关键词分类
        "general_fraud": {
            "patterns": [
                # 英文关键词（动态生成模式）
                *[(rf"\b{kw}\b", weight) for kw, weight in {
                    "account": 0.3, "verify": 0.4, "bank": 0.3,
                    "urgent": 0.5, "immediately": 0.4, "password": 0.5,
                    "login": 0.4, "money": 0.3, "payment": 0.3,
                    "verification": 0.4, "security": 0.3, "secure": 0.3,
                    "update": 0.3, "confirm": 0.3, "important": 0.4,
                    "limited time": 0.4, "expires": 0.4, "suspended": 0.4,
                    "blocked": 0.4, "click": 0.3, "link": 0.3
                }.items()],
                
                # 中文关键词（直接匹配）
                (r"账户", 0.3), (r"验证", 0.3), (r"银行", 0.3),
                (r"紧急", 0.4), (r"立即", 0.4), (r"密码", 0.5),
                (r"登录", 0.4), (r"钱", 0.3), (r"支付", 0.3)
            ],
            "category": "suspicious_keywords"
        }
    }

        
        # 2. 改进文本处理
        text_lower = user_text.lower()
        matched_patterns = []
        pattern_score = 0.0
        category_scores = defaultdict(float)  # 使用分类最高分

        # 3. 计算关键词匹配得分
        for category, data in keyword_patterns.items():
            max_category_score = 0.0
            for pattern, weight in data["patterns"]:
                if re.search(pattern, text_lower):
                    max_category_score = max(max_category_score, weight)
            category_scores[data["category"]] = max_category_score

        pattern_score = sum(category_scores.values())  # 每个分类只取最高分
        
        # 4. 获取相似案例
        retrieved_cases = retrieve_similar(user_text, faiss_index, scam_texts)
        
        # 5. 使用 LLM 进行深度分析
        if pattern_score > 0.3 or retrieved_cases:  # 只在有可疑情况时调用 LLM
            try:
                llm_prompt = f"""Analyze if this message is a potential scam:
                Message: {user_text}
                
                Consider:
                1. Does it request money or financial information?
                2. Is there urgency or pressure?
                3. Does it mix different languages suspiciously?
                4. Are there any red flags typical of scams?
                
                Similar cases found: {retrieved_cases[:2] if retrieved_cases else 'None'}
                
                Provide a brief analysis and confidence score (0-1).
                """
                
                llm_analysis = llm_predict(llm_prompt)
                
                # 解析 LLM 的置信度（假设 LLM 会在回复中包含数字置信度）
                
                confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', llm_analysis.lower())
                llm_confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                
                # 综合评分 置信度归一化处理
                final_confidence = min(max(pattern_score * 0.4 + llm_confidence * 0.6, 0.0), 1.0)
            except Exception as e:
                logging.error(f"LLM analysis failed: {e}")
                final_confidence = pattern_score
                llm_analysis = "LLM analysis unavailable."
        else:
            final_confidence = pattern_score
            llm_analysis = "No significant risk indicators found."
        
        # 6. 生成最终报告
        risk_level = "High Risk" if final_confidence > 0.7 else "Medium Risk" if final_confidence > 0.4 else "Low Risk"
        
        # 特征描述生成
        bullet_points = []
        if category_scores.get("financial"):
            bullet_points.append("💰 Detected financial request patterns")
        if category_scores.get("urgent"):
            bullet_points.append("⏰ Contains urgent time pressure")
        if len(matched_patterns) > 3:
            bullet_points.append(f"🔍 Found {len(matched_patterns)} suspicious keywords")

        # 专业英文报告模板
        report = f"""
            [Fraud Detection Report]
            Risk Level: {risk_level} (Confidence: {final_confidence*100:.1f}%)

            Key Indicators:
            {'\n'.join(bullet_points) if bullet_points else 'No strong indicators found'}

            Recommendations:
            1. Do NOT transfer money or share sensitive information
            2. Verify the requester's identity through official channels
            3. Report suspicious requests to platform administrators
                    """

        return {
            "scam_detected": final_confidence > 0.5,
            "scam_type": next(iter(category_scores), "unknown"),
            "confidence": round(final_confidence, 2),
            "report": report,
            # 保留调试数据
            "_debug": {
                "raw_scores": dict(category_scores),
                "llm_analysis": llm_analysis
            }
        }
        
    except Exception as e:
        logging.error(f"Error in detect_and_generate_report: {e}")
        return {
            "scam_detected": False,
            "scam_type": "error",
            "confidence": 0.0,
            "report": f"Error processing request: {str(e)}",
            "retrieved_cases": [],
            "matched_keywords": []
        }