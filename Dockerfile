FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for API and Streamlit
EXPOSE 8000
EXPOSE 8501

# Script to run streamlit service
RUN echo "#!/bin/bash\n\
    streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0\n\
    " > /app/start.sh

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
