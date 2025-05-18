import os
import pathlib
from app.config import settings
from app.utils.logging import vertex_log

# Set default directory for credentials based on storage dir
default_credentials_dir = os.path.join(settings.STORAGE_DIR, "credentials")

# Get credentials directory from env var or use default
CREDENTIALS_DIR = os.environ.get('CREDENTIALS_DIR', default_credentials_dir)


vertex_log('info', f"Using credentials directory: {CREDENTIALS_DIR}")

# Environment-based settings
API_KEY = os.environ.get('VERTEX_API_KEY', '')
if API_KEY:
    vertex_log('info', "Using API Key authentication")
else:
    vertex_log('info', "No API Key found, falling back to credentials file")

# Service account settings
GOOGLE_CREDENTIALS_JSON = os.environ.get('GOOGLE_CREDENTIALS_JSON', '')
if GOOGLE_CREDENTIALS_JSON:
    vertex_log('info', "Using GOOGLE_CREDENTIALS_JSON environment variable for authentication")

# Project/Location
PROJECT_ID = os.environ.get('VERTEX_PROJECT_ID', '')
LOCATION = os.environ.get('VERTEX_LOCATION', 'us-central1')

# URL for models configuration with default value
default_models_config_url = "https://gist.githubusercontent.com/gzzhongqi/e0b684f319437a859bcf5bd6203fd1f6/raw"
MODELS_CONFIG_URL = os.environ.get('VERTEX_MODELS_CONFIG_URL', default_models_config_url)
vertex_log('info', f"Using models config URL: {MODELS_CONFIG_URL}")

def update_env_var(name, value):
    """Update environment variable in memory."""
    os.environ[name] = value
    vertex_log('info', f"Updated environment variable: {name}")

def update_config(name, value):
    """Update config variables and environment variables."""
    global API_KEY, GOOGLE_CREDENTIALS_JSON, PROJECT_ID, LOCATION

    if name == 'VERTEX_API_KEY':
        API_KEY = value
        vertex_log('info', "Updated API Key")
    elif name == 'GOOGLE_CREDENTIALS_JSON':
        GOOGLE_CREDENTIALS_JSON = value
        vertex_log('info', "Updated Google Credentials JSON")
    elif name == 'VERTEX_PROJECT_ID':
        PROJECT_ID = value
        vertex_log('info', f"Updated Project ID to {value}")
    elif name == 'VERTEX_LOCATION':
        LOCATION = value
        vertex_log('info', f"Updated Location to {value}")
    elif name == 'VERTEX_MODELS_CONFIG_URL':
        global MODELS_CONFIG_URL
        MODELS_CONFIG_URL = value
        vertex_log('info', f"Updated Models Config URL to {value}")
    else:
        vertex_log('warning', f"Unknown config variable: {name}")
        return

    # Also update the environment variable
    update_env_var(name, value)

# API Key for Vertex Express Mode
VERTEX_EXPRESS_API_KEY_VAL = []
if settings.VERTEX_EXPRESS_API_KEY:
    VERTEX_EXPRESS_API_KEY_VAL = [key.strip() for key in settings.VERTEX_EXPRESS_API_KEY.split(',') if key.strip()]

# Fake streaming settings from settings module
FAKE_STREAMING_ENABLED = settings.FAKE_STREAMING
FAKE_STREAMING_INTERVAL_SECONDS = settings.FAKE_STREAMING_INTERVAL

# Validation logic moved to app/auth.py