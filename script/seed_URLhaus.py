import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from database import connect_db  
import requests
import logging

logging.basicConfig(
    filename='scam_importer.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)

URLHAUS_URL = "https://urlhaus.abuse.ch/downloads/csv_recent/"

def fetch_urlhaus_data():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/58.0.3029.110'}
        response = requests.get(URLHAUS_URL, headers=headers)
        if response.status_code == 200:
            lines = response.text.splitlines()
            urls = [line.split(',')[2].strip('"') for line in lines if line and not line.startswith('#')]
            return urls
        else:
            logging.error(f"Failed to fetch URLhaus data, status code: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Exception in fetch_urlhaus_data: {e}")
        return []

def seed_urlhaus_data():
    conn = connect_db()  #connect to database
    if not conn:
        return
    try:
        cur = conn.cursor()
        urls = fetch_urlhaus_data()
        records = [(url, "malware", 0.9) for url in urls]  # scam_type is "malware"
        if records:
            cur.executemany(
                "INSERT INTO realtime_scams (source, scam_text, scam_type, confidence) VALUES (%s, %s, %s, %s)",
                [("URLhaus", record[0], record[1], record[2]) for record in records]
            )
            conn.commit()
            print(f"Successfully inserted {len(records)} URLhaus records")
        else:
            print("No data fetched from URLhaus.")
    except Exception as e:
        logging.error(f"ERROR during URLhaus data seeding: {e}")
        print("ERROR:", e)
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    seed_urlhaus_data()