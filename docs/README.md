# Wildfire Management System

A comprehensive system for pre-fire risk assessment and post-fire spread prediction using satellite data and deep learning.

## Features
- **Pre-fire Risk Assessment**: Uses LSTM network to predict fire risk based on weather and historical data.
- **Post-fire Spread Prediction**: Uses U-Net architecture to forecast fire spread from satellite imagery.
- **Interactive Dashboard**: Streamlit-based UI for visualizing risk maps and simulation results.
- **REST API**: FastAPI backend for model inference and data retrieval.

## Prerequisites
- **Docker & Docker Compose** (recommended) - [Install Docker](https://docs.docker.com/get-docker/)
- **OR Python 3.9+** for manual installation

## Quick Start with Docker (Recommended)

### 1. Clone and Configure
```bash
# Navigate to project directory
cd "f:\minor project wildfire management"

# Configure environment variables (optional)
# Edit .env file with your API keys
```

### 2. Build and Run with Docker Compose
```bash
# Build and start both frontend and backend together
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build
```

### 3. Access the Application
- **Frontend Dashboard**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 4. Stop the Application
```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Manual Installation (Alternative)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Edit `.env` file with your API keys:
```env
NASA_FIRMS_API_KEY=your_key_here
WEATHER_API_KEY=your_key_here
```

### 3. Run Backend and Frontend

**Terminal 1 - Backend:**
```bash
python -m uvicorn api.backend.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend:**
```bash
python -m streamlit run frontend/app.py --server.port 8501
```

### 4. Access the Application
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000/docs

## Docker Architecture

The `docker-compose.yml` orchestrates a single service that runs both:
- **FastAPI Backend** on port 8000
- **Streamlit Frontend** on port 8501

Both services start automatically using the startup script defined in the Dockerfile.

## Project Structure
```
wildfire_management/
├── api/                    # FastAPI backend
│   ├── backend/main.py    # API entry point
│   └── routes/            # API endpoints
├── frontend/app.py        # Streamlit dashboard
├── src/                   # Core logic
│   ├── data_collection/   # NASA FIRMS, Weather, GEE
│   ├── models/            # LSTM, U-Net architectures
│   ├── preprocessing/     # Feature engineering
│   └── utils/             # Visualization helpers
├── tests/                 # Test suite
├── Dockerfile             # Container definition
├── docker-compose.yml     # Service orchestration
└── requirements.txt       # Python dependencies
```

## Testing
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_models.py
pytest tests/test_integration.py
```

## Troubleshooting

### Docker Issues
```bash
# Rebuild containers from scratch
docker-compose build --no-cache

# View logs
docker-compose logs -f

# Check running containers
docker ps
```

### Port Conflicts
If ports 8000 or 8501 are already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Change host port
  - "8502:8501"
```

## Next Steps
1. Add your API keys to `.env`
2. Collect training data using `src/data_collection/` modules
3. Train models with `src/training/` scripts
4. Deploy to production using Docker
