FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for API and Streamlit
EXPOSE 8000
EXPOSE 8501

# Script to run both services
RUN echo "#!/bin/bash\n\
uvicorn api.backend.main:app --host 0.0.0.0 --port 8000 & \n\
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0\n\
wait -n\n\
exit $?" > /app/start.sh

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
