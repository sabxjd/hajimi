from fastapi import HTTPException, Header, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
from app.vertex.config import API_KEY # Import API_KEY directly for use in local validation
import os
import json
from app.utils.logging import vertex_log
import app.vertex.config as config

# Function to validate API key (moved from config.py)
def validate_api_key(api_key_to_validate: str) -> bool:

    return True

# API Key security scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Dependency for API key validation
async def get_api_key(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please include 'Authorization: Bearer YOUR_API_KEY' header."
        )
    
    # Check if the header starts with "Bearer "
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format. Use 'Authorization: Bearer YOUR_API_KEY'"
        )
    
    # Extract the API key
    api_key = authorization.replace("Bearer ", "")
    
    # Validate the API key
    if not validate_api_key(api_key): # Call local validate_api_key
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key

def validate_settings():
    """Validate settings for Vertex API access."""
    
    # Check if API key is set
    if not config.API_KEY:
        vertex_log('warning', "API key is not set. Some functionality may be limited.")
    
    # Check Google credentials JSON if available
    if config.GOOGLE_CREDENTIALS_JSON:
        try:
            # Try to parse the JSON to ensure it's valid
            json.loads(config.GOOGLE_CREDENTIALS_JSON)
            vertex_log('info', "Google Credentials JSON is valid")
        except json.JSONDecodeError:
            vertex_log('error', "Google Credentials JSON is not valid JSON. Please check the format.")
            return False
    
    # Check for project ID
    if not config.PROJECT_ID:
        vertex_log('warning', "Vertex AI Project ID is not set. Required for non-API key methods.")
    
    # Check location
    if not config.LOCATION:
        vertex_log('warning', "Vertex AI Location is not set, using default: us-central1")
    
    # Validate credentials directory
    if not os.path.exists(config.CREDENTIALS_DIR):
        try:
            os.makedirs(config.CREDENTIALS_DIR, exist_ok=True)
            vertex_log('info', f"Created credentials directory at: {config.CREDENTIALS_DIR}")
        except Exception as e:
            vertex_log('error', f"Failed to create credentials directory: {e}")
            return False
    
    return True