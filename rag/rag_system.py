from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import numpy as np
import faiss
import os
import logging
import faiss
from dotenv import load_dotenv
from typing import List

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

        # 获取相似案例
        retrieved_cases = retrieve_similar(user_text, faiss_index, scam_texts)
        
        # 关键词检测
        keyword_weights = {
            # 英文关键词
            "account": 0.3, "verify": 0.3, "bank": 0.3,
            "urgent": 0.4, "immediately": 0.4,
            # 中文关键词
            "账户": 0.3, "验证": 0.3, "银行": 0.3,
            "紧急": 0.4, "立即": 0.4,
            # 通用诈骗词
            "password": 0.5, "密码": 0.5,
            "login": 0.4, "登录": 0.4,
            "money": 0.3, "钱": 0.3,
            "payment": 0.3, "支付": 0.3,
            # 添加更多关键词
            "verify": 0.4, "verification": 0.4,
            "security": 0.3, "secure": 0.3,
            "update": 0.3, "confirm": 0.3,
            "urgent": 0.5, "important": 0.4,
            "limited time": 0.4, "expires": 0.4,
            "suspended": 0.4, "blocked": 0.4,
            "click": 0.3, "link": 0.3
        }
        
        # 计算关键词匹配得分
        keyword_score = 0.0
        text_lower = user_text.lower()
        matched_keywords = []
        
        for keyword, weight in keyword_weights.items():
            if keyword in text_lower:
                keyword_score += weight
                matched_keywords.append(keyword)
        
        keyword_score = min(float(keyword_score), 1.0)
        
        # 即使没有相似案例，也要考虑关键词得分
        if keyword_score > 0.3:  # 降低关键词触发阈值
            scam_confidence = keyword_score
            scam_detected = True
        elif retrieved_cases:
            # 如果有相似案例，计算综合置信度
            query_vec = encode_text(user_text).reshape(1, -1)
            distances, _ = faiss_index.search(query_vec, k=1)
            similarity_confidence = float(1.0 / (1.0 + np.mean(distances)))
            scam_confidence = float((similarity_confidence + keyword_score) / 2)
            scam_detected = True
        else:
            scam_confidence = 0.1
            scam_detected = False

        # 确定诈骗类型
        scam_types = {
            "phishing": ["account", "login", "verify", "bank", "password", "security"],
            "financial": ["money", "payment", "credit", "transfer", "fund"],
            "urgent": ["immediately", "urgent", "quick", "limited time", "expires"],
            "identity_theft": ["identity", "personal", "ssn", "social security"],
            "tech_support": ["computer", "virus", "support", "microsoft", "apple"]
        }
        
        # 计算每种诈骗类型的匹配度
        type_scores = {}
        for scam_type, keywords in scam_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[scam_type] = score
        
        detected_type = max(type_scores.items(), key=lambda x: x[1])[0] if type_scores else "unknown"
        
        # 生成详细报告
        report = f"Analysis:\n"
        if matched_keywords:
            report += f"Suspicious keywords found: {', '.join(matched_keywords)}\n"
        if retrieved_cases:
            report += f"Similar scam patterns detected.\n"
        report += f"Scam confidence: {scam_confidence:.2f}\n"
        report += f"Detected scam type: {detected_type}\n"
        
        if scam_detected:
            report += "\nWarning: This message shows characteristics of a potential scam.\n"
            report += "Recommendations:\n"
            report += "- Do not click on any links\n"
            report += "- Do not provide personal information\n"
            report += "- Contact the supposed sender through official channels\n"
        else:
            report += "\nNo immediate scam indicators found, but always remain cautious.\n"
        
        return {
            "scam_detected": scam_detected,
            "scam_type": detected_type,
            "confidence": round(scam_confidence, 2),
        "report": report,
            "retrieved_cases": retrieved_cases,
            "matched_keywords": matched_keywords  # 添加匹配的关键词信息
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