# config.py
import os
import streamlit as st # Needed for accessing secrets/env vars potentially

# --- Core Settings ---
APP_VERSION = "2.0.0" # Refactored version
PAGE_TITLE = "AI Log Analyzer"
PAGE_ICON = "🧠"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"
MIN_LOGS_FOR_CLUSTER = 10
MIN_LOGS_FOR_ANALYSIS = 10 # Consistent threshold
LOG_EXPLORER_PAGE_SIZE = 15
UI_UPDATE_INTERVAL_SECONDS = 0.75
MIN_LOGS_FOR_UI_UPDATE = 10

# --- Required DataFrame Columns ---
# Standardized column names used throughout the application
REQUIRED_COLUMNS = [
    'timestamp',      # Original timestamp string
    'level',          # Log level (INFO, ERROR, etc.) - Processed to uppercase
    'module',         # Source component/service
    'message',        # Main log message content
    'raw',            # Original raw log line
    'status_code',    # HTTP status or other code (as string)
    'latency',        # Latency in milliseconds (float)
    'env',            # Environment (production, staging, etc.)
    'datetime'        # Parsed timestamp (pd.Timestamp, NaT if failed)
    # 'cluster' column added dynamically after clustering
]

# --- UI Styling ---
COLORS = {
    "primary": "#1E90FF",    # DodgerBlue
    "secondary": "#6c757d",  # Gray
    "success": "#28a745",    # Green
    "error": "#dc3545",      # Red
    "warning": "#ffc107",    # Yellow
    "info": "#17a2b8",       # Teal
    "background": "#F0F2F6", # Lighter Gray Background
    "card": "#FFFFFF",       # White Card Background
    "text": "#212529",       # Dark Text
    "light_text": "#495057"  # Slightly Lighter Text
}

# Mapping log levels to colors/styles consistently
LOG_LEVEL_STYLES = {
    "ERROR": {"color": COLORS["error"], "border": "error-card-border"},
    "WARNING": {"color": COLORS["warning"], "border": "warning-card-border"},
    "INFO": {"color": COLORS["success"], "border": "info-card-border"},
    "DEBUG": {"color": COLORS["secondary"], "border": "debug-card-border"},
    "PARSE_ERROR": {"color": "#FFA500", "border": "parse-error-card-border"}, # Orange
    "DEFAULT": {"color": COLORS["secondary"], "border": ""} # Fallback
}

CHART_THEME_OPTIONS = ["Streamlit Theme", "Plotly White", "Plotly Dark"]
CHART_THEME_MAP = {"Streamlit Theme": None, "Plotly White": "plotly_white", "Plotly Dark": "plotly_dark"}

# --- API Keys & URLs (Load from Environment Variables or Streamlit Secrets) ---
# Use st.secrets for deployment, os.getenv for local dev flexibility
def get_config_value(key: str, default: str = None) -> str:
    """Helper to get config from st.secrets or environment variables."""
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

OPENROUTER_API_KEY = get_config_value("OPENROUTER_API_KEY", "sk-or-v1-...") # Placeholder
OLLAMA_API_URL = get_config_value("OLLAMA_API_URL", "http://192.168.160.221:11434/api/generate") # Default local Ollama
DEFAULT_SSE_LOG_URL = get_config_value("SSE_LOG_URL", "http://localhost:8000/stream-logs") # Default local SSE API

# --- LLM Model Preferences ---
OLLAMA_PREFERRED_MODELS = ["gemma3:1b", "llama3:8b", "mistral", "phi3", "gemma:7b", "gemma2:9b"]
REMOTE_LLM_OPTIONS = sorted([
    "mistralai/mistral-7b-instruct:free", "meta-llama/llama-3.1-8b-instruct:free"
])
DEFAULT_REMOTE_MODEL = "mistralai/mistral-7b-instruct:free"

# --- Clustering Settings ---
DEFAULT_N_CLUSTERS = 4
MAX_CLUSTERS_HEURISTIC_DIVISOR = 5 # Max clusters = max(2, min(15, num_logs // THIS))
TFIDF_MAX_DF = 0.95
TFIDF_MIN_DF = 2
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 5000

# --- Sample Log Generation Settings (if logs_backend is used) ---
SAMPLE_LOG_COUNT = 300 # Default number of generated logs
SAMPLE_LOG_CSV_FILENAME = 'api_logs.csv' # Preferred file for sample logs
