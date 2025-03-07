import psycopg2

DB_NAME = "scam_detector"
DB_USER = "postgres"
DB_PASSWORD = "070827"
DB_HOST = "localhost"
DB_PORT = "5432"

def connect_db():
    try:
        conn = psycopg2.connect(
            dbname = DB_NAME,
            user = DB_USER,
            password = DB_PASSWORD,
            host = DB_HOST,
            port = DB_PORT
        )
        print("CONNECT TO DATABASE SUCCESSFULLY")
        return conn
    except Exception as e:
        print("CONNECTION ERROR:",e)
        return None
    
if __name__ == "__main__":
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scams;")
        scams = cursor.fetchall()
        for scam in scams:
            print(scam)
        cursor.close()
        conn.close()