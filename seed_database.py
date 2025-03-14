import os
import csv
import psycopg2
import logging

#configure logging
logging.basicConfig(filename='scam_importer.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s: %(message)s')

DB_NAME = "scam_detector"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD", "070827")
DB_HOST = "localhost"
DB_PORT = "5432"

def seed_sms_spam_data(csv_path):
    try:
        conn = psycopg2.connect(
            dbname = DB_NAME,
            user = DB_USER,
            password = DB_PASSWORD,
            host = DB_HOST,
            port = DB_PORT

        )
        cur = conn.cursor()


        #read spam records from csv file
        records = []
        with open(csv_path, newline='', encoding='latin1') as csvfile:
            reader = csv.Dictreader(csvfile)
            for row in reader:
                if row['label'].strip().lower() == 'spam':
                    scam_text = row['text'].strip()
                    scam_type = "phishing"
                    records.append((scam_text, scam_type))


        #insert records into scams table
        if records:
            args_str = ",".join(
                cur.mogrify("(%s, %s)", record).decode('utf-8') for record in records
            )
            cur.execute("INSERT INTO scams (scam_text, scam_type) VALUES " + args_str)
            conn.commit()
            print(f"Successfully inserted {len(records)} spam records")
        else:
            print("No spam records found in the CSV file.")


    except Exception as e:
        logging.error("ERROR during database seeding: %s", e)
        print("ERROR:", e)
        if conn:
            conn.rollback()
    finally:
        if 'cur' in locals() and cur is not None:
            cur.close()
        if 'conn' in locals() and conn is not None:
            conn.close()

if __name__ == "__main__":
    seed_sms_spam_data("data/merged_sms.csv")
