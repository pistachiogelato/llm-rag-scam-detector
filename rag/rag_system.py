from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import requests
import logging


# Initialize the embedding model
encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

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

def retrieve_similar(query_text: str, index: faiss.IndexFlatL2, scam_texts: list, k: int = 3) -> list:
    """
    Retrieve k most similar scam examples for a given query text.
    :param query_text: The input text to query.
    :param index: FAISS index built on scam case vectors.
    :param scam_texts: A list of original scam texts corresponding to the vectors.
    :param k: Number of similar examples to retrieve.
    :return: A list of scam texts that are most similar.
    """
    query_vec = encode_text(query_text)
    # Ensure the query vector is of shape (1, vector_dim)
    query_vec = np.array([query_vec], dtype='float32')
    distances, indices = index.search(query_vec, k)
    # Retrieve scam texts based on indices
    return [scam_texts[i] for i in indices[0]]

def llm_predict(prompt: str) -> str:
    """
    Call the LLM API to generate a response based on the prompt.
    Returns the generated text.
    """
    api_url = os.getenv("DEEPEEK_API_URL")
    api_key = os.getenv("DEEPEEK_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"prompt": prompt, "max_tokens": 150}
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("generated_text", "")
        else:
            logging.error("LLM API error: %s", response.text)
            raise Exception("LLM API call failed")
    except Exception as e:
        logging.error("Exception in llm_predict: %s", e)
        raise e



def generate_report(user_text: str, retrieved_cases: list) -> str:
    """
    Generate a detailed scam detection report using the user text and retrieved scam examples.
    Integrates an LLM API call.
    """
    prompt = (
        f"User message: {user_text}\n"
        f"Related scam examples:\n" + "\n".join(retrieved_cases) +
        "\n\nBased on the above, please analyze whether the user's message is a scam, "
        "explain why, and provide recommendations."
    )
    # Call the actual LLM API
    return llm_predict(prompt)

# Example integration function
def detect_and_generate_report(user_text: str, scam_texts: list, faiss_index: faiss.IndexFlatL2) -> dict:
    """
    Integrate the retrieval and generation modules.
    :param user_text: The input text from the user.
    :param scam_texts: List of scam case texts from the database.
    :param faiss_index: FAISS index built on scam case vectors.
    :return: A dictionary with the detection result and generated report.
    """
    # Retrieve similar scam cases
    retrieved_cases = retrieve_similar(user_text, faiss_index, scam_texts)
    # Generate a report based on the user text and retrieved cases
    report = generate_report(user_text, retrieved_cases)
    # For now, we simulate a confidence score based on simple rules
    result = {
        "scam_detected": True,
        "scam_type": "phishing",
        "confidence": 0.9,
        "report": report,
        "retrieved_cases": retrieved_cases
    }
    return result

if __name__ == "__main__":
    # Example: simulate loading scam texts (in production, load from database)
    scam_texts = [
        "http://mitrashopee.com/",
        "https://pub-d4ba3ddf19254ddeadc01c5590e107d5.r2.dev/index.html",
        "http://yonggevl3k.la-xrdn.workers.dev/",
        "https://mkup-a.kroin.top/"
    ]
    # Build FAISS index for demo
    vectors = np.array([encode_text(text) for text in scam_texts])
    index = build_faiss_index(vectors)
    # Simulate detection
    user_message = "Please verify your bank account immediately!"
    output = detect_and_generate_report(user_message, scam_texts, index)
    print("Detection result:", output)
