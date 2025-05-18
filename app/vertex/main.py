from fastapi import FastAPI, Depends, APIRouter, HTTPException
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
import asyncio
# from fastapi.responses import JSONResponse # Not used
# import os # Not used


# Local module imports
from app.vertex.auth import get_api_key, validate_api_key
from app.vertex.credentials_manager import CredentialManager
from app.vertex.vertex_ai_init import init_vertex_ai
from app.utils.logging import vertex_log
# import config as app_config # Not directly used in main.py

# Routers
from app.vertex.routes import models_api
from app.vertex.routes import chat_api

app = FastAPI(title="OpenAI to Gemini Adapter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

credential_manager = CredentialManager()
app.state.credential_manager = credential_manager # Store manager on app state

# Include API routers
app.include_router(models_api.router) 
app.include_router(chat_api.router)

# Create router
vertex_router = APIRouter(prefix="/vertex")

# Security scheme
security = HTTPBearer()

# Include vertex_router in the app
app.include_router(vertex_router)

@app.on_event("startup")
async def startup_event():
    if await init_vertex_ai(credential_manager): # Added await
        vertex_log('info', "Vertex AI credential and model config initialization check completed successfully.")
    else:
        vertex_log('error', "Failed to initialize a fallback Vertex AI client. API will likely fail.")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "OpenAI to Gemini Adapter is running."
    }

@vertex_router.get("/health")
async def health_check(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Simple health check endpoint for the Vertex AI integration.
    """
    # Validate API key
    api_key = validate_api_key(credentials)
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # If we get here, API key is valid
    vertex_log('info', "Health check passed with valid API key")
    return {"status": "ok", "message": "Vertex AI integration is operational"}

@vertex_router.get("/status")
async def status():
    """
    Public status endpoint (no auth required)
    """
    vertex_log('info', "Status check requested")
    return {
        "status": "online",
        "version": "1.0.0",
        "endpoints_available": [
            "/vertex/health",
            "/vertex/predict",
            "/vertex/generate",
            "/vertex/admin/config",
            "/vertex/models/list"
        ]
    }
