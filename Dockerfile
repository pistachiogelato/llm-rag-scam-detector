# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . /app/

# Create necessary directory
RUN mkdir -p /app/data

# Copy requirements and install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# set environment avariable
ENV DB_NAME=scam_detector \
    DB_USER=postgres \
    DB_PASSWORD=070827 \
    DB_HOST=localhost \
    DB_PORT=5432 \
    PYTHONPATH=/app

# expose the port the app runs on
EXPOSE 8000

# Command to run the application (using uvicorn)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


