import requests
import psycopg2
import os
import logging

logging.basicConfig(
    filename = 'scam_importer.log',
    level = logging.ERROR,
    format='%(asctime)s %(levelname)s: %(message)s'
)

#DATABASE CONFIGURATION
DB_NAME = "scam_detector"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD", "070827")
DB_HOST = "localhost"
DB_PORT = "5432"

#OpenPhish url
OPENPHISH_URL = "https://openphish.com/feed.txt"

#fetch data from openphish
def fetch_openphish_data():
    try:
        response = requests.get(OPENPHISH_URL)
        if response.status_code == 200:
            urls = response.text.splitlines()
            urls = [url.strip() for url in urls if url.strip()]
            return urls
        else:
            logging.error("Failed to fetch OpenPhish data, status code: %s", response.status_code)
            return []
    except Exception as e:
        logging.error("Exception in fetch_openphish_data: %s", e)
        return []
    
#seed OpenPhish data into realtime_scams
def seed_openphish_data():
    try:
        conn = psycopg2.connect(
            dbname = DB_NAME,
            user = DB_USER,
            password = DB_PASSWORD,
            host = DB_HOST,
            port = DB_PORT
        )
        cur = conn.cursor()

        urls = fetch_openphish_data()
        records = []
        for url in urls:
            scam_text = url
            scam_type = "phishing"
            confidence = 0.8
            records.append((scam_text, scam_type, confidence))

        if records:
            args_str = b','.join(
                cur.mogrify("(%s, %s, %s, %s)",("OpenPhish", record[0],record[1],record[2]))
                for record in records
            ).decode('utf-8')
            cur.execute("INSERT INTO realtime_scams (source, scam_text, scam_type, confidence) VALUES" + args_str)
            conn.commit()
            print(f"Successfully inserted {len(records)} OpenPhish records")
        else:
            print("No data fetched from OpenPhish.")

    except Exception as e:
        logging.error("ERROR during OpenPhish data seeding: %s", e)
        print("ERROR:", e)
        if conn:
            conn.rollback()
    finally:
        if 'cur' in locals() and cur is not None:
            cur.close()
        if 'conn' in locals() and conn is not None:
            conn.close()


if __name__ =="__main__":
    seed_openphish_data()