# Use the official Python slim image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy script into the container
COPY __main__.py /app/data_extractor/__main__.py
COPY sinan_extractor.py /app/data_extractor/sinan_extractor.py
COPY requirements.txt .

# Install any needed packages to build pysus
RUN apt-get update && apt-get install -y build-essential libffi-dev && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip/*

ENTRYPOINT ["python3", "-m", "data_extractor"]