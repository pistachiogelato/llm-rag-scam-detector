import csv
import chardet
import re

def detect_encoding(file_path: str) -> str:
    """
    Detect the encoding of a file.
    """
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
        return result["encoding"]

def normalize_text(text: str) -> str:
    """
    Strip whitespace and convert text to lowercase.
    """
    return text.strip().lower()

def anonymize_text(text: str) -> str:
    """
    Replace sensitive information (e.g., phone numbers) with placeholders.
    """
    # Replace any sequence of 10 or more digits with [PHONE]
    return re.sub(r'\d{10,}', '[PHONE]', text)

def deduplicate_texts(texts: list) -> list:
    """
    Remove duplicate texts.
    """
    return list(set(texts))

def load_scam_texts(csv_path: str) -> list:
    """
    Load scam texts from a CSV file.
    Expects CSV to have at least fields "label" and "text".
    """
    encoding = detect_encoding(csv_path)
    scam_texts = []
    with open(csv_path, newline='', encoding=encoding) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row.get("label", "").strip().lower()
            # Map different labels to "spam"
            if label in ["spam", "financial scam", "phishing"]:
                text = normalize_text(row["text"])
                text = anonymize_text(text)
                scam_texts.append(text)
    return deduplicate_texts(scam_texts)

def merge_datasets(file_list: list) -> list:
    """
    Merge scam texts from multiple CSV files and deduplicate.
    """
    merged = []
    for file in file_list:
        merged += load_scam_texts(file)
    return deduplicate_texts(merged)

if __name__ == "__main__":
    # Example usage: merge SMS, financial scams, and OpenPhish data
    files = ["data/sms_spam.csv",  "data/openphish_feed.csv"]
    all_scam_texts = merge_datasets(files)
    print("Total unique scam texts:", len(all_scam_texts))
