import requests
import csv
from datetime import datetime

def fetch_openphish_data(url: str) -> list:
    response = requests.get(url)
    if response.status_code == 200:
        # 每行一个 URL
        return [line.strip() for line in response.text.splitlines() if line.strip()]
    else:
        return []

def save_to_csv(data: list, csv_path: str):
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "url"])
        for item in data:
            writer.writerow([datetime.now().isoformat(), item])

if __name__ == "__main__":
    openphish_url = "https://openphish.com/feed.txt"
    data = fetch_openphish_data(openphish_url)
    save_to_csv(data, "data/openphish_feed.csv")
