from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predictions

app = FastAPI(title="Wildfire Management API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction routes with /predictions prefix
app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])

@app.get("/")
async def root():
    return {"message": "Wildfire Management API", "status": "online"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}
