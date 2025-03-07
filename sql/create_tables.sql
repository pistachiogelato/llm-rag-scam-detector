CREATE TABLE scams (
    id SERIAL PRIMARY KEY,
    scam_text TEXT NOT NULL,
    scam_type VARCHAR(100),
    confidence FLOAT DEFAULT 1.0,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
