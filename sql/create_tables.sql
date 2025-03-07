-- sql/create_tables.sql
CREATE TABLE IF NOT EXISTS scams (
    id SERIAL PRIMARY KEY,
    scam_text TEXT NOT NULL,
    scam_type VARCHAR(100),
    confidence FLOAT DEFAULT 1.0,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS realtime_scams (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    scam_text TEXT NOT NULL,
    scam_type VARCHAR(100),
    confidence FLOAT DEFAULT 0.8,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
