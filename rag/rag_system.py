from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import numpy as np
import faiss
import os
import logging
import faiss
from dotenv import load_dotenv

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
    Retrieve the k most similar scam texts to the query text using the FAISS index.
    
    Args:
        query_text (str): The user-provided text to query against.
        index (faiss.IndexFlatL2): Pre-built FAISS index for scam text embeddings.
        scam_texts (list): List of original scam texts corresponding to the indexed vectors.
        k (int): Number of similar examples to retrieve (default: 3).
    
    Returns:
        list: List of k most similar scam texts.
    """
    query_vec = encode_text(query_text)#get(384,)
    query_vec = query_vec.reshape(1, -1)#get(1,384)

    print(f"Query embedding shape: {query_vec.shape}")  
    distances, indices = index.search(query_vec, k)
    print(f"Retrieved distances: {distances}")
    print(f"Retrieved indices: {indices}")

    return [scam_texts[i] for i in indices[0]]

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
    Detect potential scams and generate a report by integrating retrieval and LLM generation.
    
    Args:
        user_text (str): The user-provided text to analyze.
        scam_texts (list): List of known scam texts for retrieval.
        faiss_index (faiss.IndexFlatL2): Pre-built FAISS index for scam text embeddings.
    
    Returns:
        dict: Detection result including scam status, type, confidence, report, and retrieved cases.
    """
    print(f"Loaded scam texts: {len(scam_texts)}")

    retrieved_cases = retrieve_similar(user_text, faiss_index, scam_texts)
    report = generate_report(user_text, retrieved_cases)
    # TODO: Parse LLM response for dynamic scam_detected, scam_type, and confidence
    return {
        "scam_detected": True,  # Placeholder until LLM parsing is implemented
        "scam_type": "phishing",  # Placeholder
        "confidence": 0.9,  # Placeholder
        "report": report,
        "retrieved_cases": retrieved_cases
    }