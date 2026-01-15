# Wildfire Management System 🔥

> A comprehensive AI-powered system for wildfire risk assessment and spread prediction

## 🚀 Quick Start with Docker

The easiest way to run the complete application (backend + frontend):

```bash
# 1. Navigate to project directory
cd "f:\minor project wildfire management"

# 2. Start both services with Docker Compose
docker-compose up --build
```

**Access the application:**
- 🌐 **Dashboard**: http://localhost:8501
- 🔌 **API**: http://localhost:8000/docs

## 📋 What's Included

- **Pre-Fire Risk Assessment** - LSTM-based prediction model
- **Post-Fire Spread Forecasting** - U-Net architecture for spatial prediction
- **Interactive Dashboard** - Streamlit web interface with maps
- **REST API** - FastAPI backend for programmatic access

## 🐳 Docker Deployment

### Single Command Deployment
```bash
docker-compose up -d --build
```

This starts:
- FastAPI backend on port 8000
- Streamlit frontend on port 8501

### Manage Services
```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild from scratch
docker-compose build --no-cache
```

## 💻 Manual Installation

If you prefer running without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start backend
python -m uvicorn api.backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Start frontend
python -m streamlit run frontend/app.py --server.port 8501
```

## 📖 Documentation

See [docs/README.md](docs/README.md) for detailed documentation including:
- Architecture overview
- API endpoints
- Testing instructions
- Troubleshooting guide

## 🧪 Testing

```bash
pytest
```

## 📁 Project Structure

```
├── api/              # FastAPI backend
├── frontend/         # Streamlit dashboard
├── src/              # Core models and data processing
├── tests/            # Test suite
├── Dockerfile        # Container definition
└── docker-compose.yml # Service orchestration
```

## ⚙️ Configuration

Edit `.env` to add your API keys:
```env
NASA_FIRMS_API_KEY=your_key
WEATHER_API_KEY=your_key
```

## 🤝 Contributing

1. Train models using data in `data/`
2. Run tests: `pytest`
3. Submit improvements

---

**Built with:** Python • FastAPI • Streamlit • PyTorch • Docker
