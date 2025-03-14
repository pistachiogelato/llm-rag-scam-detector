import psycopg2
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)

def check_database():
    """
    检查数据库内容和状态
    """
    try:
        conn = psycopg2.connect(
            dbname="scam_detector",
            user="postgres",
            password="070827",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        
        # 1. 检查表是否存在
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'realtime_scams'
            );
        """)
        table_exists = cur.fetchone()[0]
        print(f"Table 'realtime_scams' exists: {table_exists}")
        
        if table_exists:
            # 2. 检查总记录数
            cur.execute("SELECT COUNT(*) FROM realtime_scams;")
            total_count = cur.fetchone()[0]
            print(f"\nTotal records: {total_count}")
            
            # 3. 检查最近的记录
            cur.execute("""
                SELECT scam_text, timestamp 
                FROM realtime_scams 
                ORDER BY timestamp DESC 
                LIMIT 5;
            """)
            print("\nMost recent records:")
            for record in cur.fetchall():
                print(f"Text: {record[0][:100]}... | Time: {record[1]}")
            
            # 4. 检查空值情况
            cur.execute("""
                SELECT COUNT(*) 
                FROM realtime_scams 
                WHERE scam_text IS NULL OR scam_text = '';
            """)
            empty_count = cur.fetchone()[0]
            print(f"\nEmpty or NULL records: {empty_count}")
            
            # 5. 检查文本长度分布
            cur.execute("""
                SELECT 
                    CASE 
                        WHEN length(scam_text) < 100 THEN 'Short (<100)'
                        WHEN length(scam_text) < 500 THEN 'Medium (100-500)'
                        ELSE 'Long (>500)'
                    END as text_length,
                    COUNT(*)
                FROM realtime_scams
                GROUP BY text_length;
            """)
            print("\nText length distribution:")
            for record in cur.fetchall():
                print(f"{record[0]}: {record[1]} records")
            
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    check_database() 