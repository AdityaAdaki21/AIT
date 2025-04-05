# --- START OF FILE app.py ---

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import io
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime, timedelta # Added timedelta
import random
import json
import threading
import queue
import time
import os
import traceback # Added for detailed error logging

# --- Backend Import ---
try:
    from logs_backend import fetch_logs as fetch_static_logs # Rename to avoid conflict
except ImportError:
    st.warning("Could not import logs_backend.py. 'Sample Logs' and 'Upload' features might be limited.")
    def fetch_static_logs():
        st.error("logs_backend.py not found, cannot load sample logs.")
        return []

# Define parse_timestamp here as it's crucial
def parse_timestamp(ts_str):
    """Parses timestamp strings into datetime objects, handling common formats."""
    if not isinstance(ts_str, str) or not ts_str.strip():
        return pd.NaT # Return Not-a-Time for invalid input
    # Try common formats directly
    formats = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", # ISO-like
        "%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%dT%H:%M:%S.%f", # With milliseconds/microseconds
        "%Y/%m/%d %H:%M:%S", # Slash separators
        "%d/%b/%Y:%H:%M:%S", # Apache CLF (needs timezone handling if present, simplified here)
        "%b %d %H:%M:%S",   # Syslog style (assumes current year, handles Dec->Jan rollover)
        # Add other specific formats if you encounter them
    ]
    now = datetime.now()
    dt = pd.NaT
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str.strip(), fmt)
            # Handle missing year in syslog format
            if fmt == "%b %d %H:%M:%S" and dt.year == 1900:
                dt = dt.replace(year=now.year)
                if dt.month == 12 and now.month == 1: # Handle year rollover
                    dt = dt.replace(year=now.year - 1)
            # If parsing succeeded, break the loop
            break
        except ValueError:
            continue # Try the next format

    # Fallback using pandas if specific formats failed
    if pd.isna(dt):
        try:
            dt = pd.to_datetime(ts_str.strip(), errors='coerce')
        except Exception:
            dt = pd.NaT # Ensure NaT on any pandas error

    return dt

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-Powered Log Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Configuration ---
COLORS = {
    "primary": "#4169E1", "secondary": "#6C757D", "success": "#28A745",
    "error": "#DC3545", "warning": "#FFC107", "info": "#17A2B8",
    "background": "#F8F9FA", "card": "#FFFFFF", "text": "#212529"
}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-...") # Use your key or env var
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.229.153:11434/api/generate") # Adjust if needed
DEFAULT_SSE_LOG_URL = os.getenv("SSE_LOG_URL", "http://localhost:8000/stream-logs") # URL of your FastAPI SSE server

# --- Custom CSS ---
st.markdown(f"""
<style>
    .main .block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px; }}
    h1, h2, h3 {{ color: {COLORS["primary"]}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 2px; }}
    .stTabs [data-baseweb="tab"] {{ padding: 10px 20px; border-radius: 4px 4px 0px 0px; }}
    .stTabs [aria-selected="true"] {{ background-color: {COLORS["primary"]}; color: white; }}
    .error-card {{ color: {COLORS["error"]}; border-left: 5px solid {COLORS["error"]}; padding-left: 10px; background-color: #fef2f2; }}
    .warning-card {{ color: {COLORS["warning"]}; border-left: 5px solid {COLORS["warning"]}; padding-left: 10px; background-color: #fffbeb; }}
    .info-card {{ color: {COLORS["info"]}; border-left: 5px solid {COLORS["info"]}; padding-left: 10px; background-color: #f0f9ff; }}
    .log-entry-card {{ margin-bottom: 10px; background-color: {COLORS['background']}; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.9em; display: flex; justify-content: space-between; align-items: center; border: 1px solid #e0e0e0; }}
    .log-text {{ flex-grow: 1; margin-right: 15px; word-break: break-all; /* Wrap long lines */ }}
    .css-1544g2n.e1fqkh3o4 {{ padding: 2rem; border-radius: 0.5rem; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }} /* Card styling */
    .card {{ background-color: {COLORS["card"]}; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }}
    .metric-card {{ background-color: {COLORS["card"]}; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; height: 120px; /* Ensure cards have same height */ display: flex; flex-direction: column; justify-content: center; }}
    .metric-value {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
    .metric-label {{ font-size: 14px; color: {COLORS["secondary"]}; }}
    .metric-sub-label {{ font-size: 12px; color: {COLORS["secondary"]}; }} /* Smaller text for details */
    .status-indicator {{ padding: 2px 8px; border-radius: 10px; font-size: 0.8em; font-weight: bold; display: inline-block; margin-left: 10px; }}
    .status-connected {{ background-color: {COLORS['success']}; color: white; }}
    .status-disconnected {{ background-color: {COLORS['error']}; color: white; }}
    .status-connecting {{ background-color: {COLORS['warning']}; color: black; }}
    .status-error {{ background-color: {COLORS['error']}; color: white; }} /* Added for error state */
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown(f"""
<div style="background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['info']});
            padding: 20px; border-radius: 10px; margin-bottom: 25px;">
    <h1 style="color: white; margin: 0; display: flex; align-items: center;">
        <span style="font-size: 2.5rem; margin-right: 10px;">🧠</span>
        AI-Powered Log Analyzer
    </h1>
    <p style="color: white; opacity: 0.9; margin-top: 5px;">
        Intelligent log analysis: real-time stream, clustering, visualizations, and LLM insights
    </p>
</div>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
required_columns = ['timestamp', 'level', 'module', 'message', 'raw', 'status_code', 'latency', 'env', 'datetime']
if 'log_df' not in st.session_state:
    st.session_state['log_df'] = pd.DataFrame(columns=required_columns)
if 'sse_thread' not in st.session_state:
    st.session_state['sse_thread'] = None
if 'sse_stop_event' not in st.session_state:
    st.session_state['sse_stop_event'] = None
if 'log_queue' not in st.session_state:
    st.session_state['log_queue'] = queue.Queue()
if 'sse_connection_status' not in st.session_state:
    st.session_state['sse_connection_status'] = "disconnected" # disconnected, connecting, connected, error
if 'last_ui_update_time' not in st.session_state:
    st.session_state['last_ui_update_time'] = 0 # Time of the last UI refresh for stream
if 'current_data_source' not in st.session_state:
    st.session_state['current_data_source'] = "Real-time Stream" # Default to stream
if 'max_stream_logs' not in st.session_state:
     st.session_state['max_stream_logs'] = 5000 # Default max logs
if 'n_clusters' not in st.session_state:
     st.session_state['n_clusters'] = 4 # Default clusters
if 'chart_theme' not in st.session_state:
     st.session_state['chart_theme'] = 'plotly_white'
if 'sse_last_error' not in st.session_state:
    st.session_state['sse_last_error'] = None # Store last connection error
if 'log_page' not in st.session_state: # Initialize log explorer page number
    st.session_state['log_page'] = 1
if 'log_explorer_filter_hash' not in st.session_state: # Hash for filter changes
     st.session_state['log_explorer_filter_hash'] = None
if 'last_uploaded_file_info' not in st.session_state: # Track unique identifier for last uploaded file
    st.session_state['last_uploaded_file_info'] = None
if 'sample_logs_loaded' not in st.session_state: # Track if sample logs loaded
    st.session_state['sample_logs_loaded'] = False

# --- Utility Functions ---
def is_csv_file(filename=None, content=None):
    """Checks if filename suggests CSV or if content looks like CSV."""
    if filename and filename.lower().endswith('.csv'):
        return True
    if content and len(content) > 0:
        # Check header of the first non-empty line
        first_line = next((line for line in content if str(line).strip()), None)
        if first_line:
            header = str(first_line).lower().strip()
            # Simple check: more than 1 comma, common headers
            if header.count(',') >= 1 and ('timestamp' in header or 'time' in header or 'status' in header or 'level' in header):
                return True
    return False

def preprocess(logs):
    """Basic preprocessing: lowercase and simplify non-alphanumeric chars."""
    # Keep common log punctuation: :, [], -, =, ., but remove others
    return [re.sub(r'[^\w\s,:\[\]\-=\.\%]', '', str(log).lower()) for log in logs] # Added % for CPU usage etc.

@st.cache_data(show_spinner="Parsing logs...")
def extract_components(_logs_tuple, filename=None): # Use tuple for caching
    """
    Extracts structured components from logs. Handles JSON lines (expected from SSE),
    CSV (from upload/sample), and falls back to generic regex parsing.
    """
    logs = list(_logs_tuple) # Convert back to list internally
    if not logs:
        return pd.DataFrame(columns=required_columns)

    data = []
    log_format_detected = None
    is_sse_json = False

    # --- 1. Try JSON Lines Format (Primary for SSE) ---
    first_valid_line = next((line.strip() for line in logs if isinstance(line, str) and line.strip()), None)
    if first_valid_line and first_valid_line.startswith('{') and first_valid_line.endswith('}'):
        log_format_detected = "JSON"
        is_sse_json = True

        for i, log_line in enumerate(logs):
            line = str(log_line).strip()
            if not line: continue

            if not (line.startswith('{') and line.endswith('}')):
                data.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "level": "PARSE_ERROR", "module": "parser",
                    "message": f"Skipped non-JSON line {i+1}", "raw": line[:200],
                    "status_code": "N/A", "latency": 0.0, "env": "parser_error", "datetime": pd.NaT })
                continue

            try:
                entry = json.loads(line)
                timestamp = str(entry.get('timestamp', ''))
                level = str(entry.get('log_level', entry.get('level', 'INFO'))).upper()
                module = str(entry.get('components', entry.get('module', 'unknown')))
                status_code = str(entry.get('status_code', 'N/A'))
                env = str(entry.get('environment', entry.get('env', 'unknown')))

                latency_ms = 0.0
                if 'latency_ms' in entry:
                    latency_ms = pd.to_numeric(entry['latency_ms'], errors='coerce')
                elif 'latency' in entry:
                     latency_ms = pd.to_numeric(entry['latency'], errors='coerce')
                elif 'response_time' in entry:
                    rt_secs = pd.to_numeric(entry['response_time'], errors='coerce')
                    if pd.notna(rt_secs): latency_ms = rt_secs * 1000 # Convert seconds to ms

                latency_ms = float(latency_ms) if pd.notna(latency_ms) else 0.0

                message_base = str(entry.get('message', ''))
                extra_details = []
                # Include more fields potentially, check for None
                for key in ['log_code', 'process_id', 'cpu_usage', 'request_id', 'user_id']:
                     if key in entry and entry[key] is not None and str(entry[key]).strip():
                         extra_details.append(f"{key.replace('_',' ').title()}: {entry[key]}")
                message = f"{message_base} ({', '.join(extra_details)})" if extra_details else message_base
                if not message.strip(): message = f"Log Event ({entry.get('log_code', '')} - {entry.get('request_id', '')})" # Fallback

                data.append({
                    "timestamp": timestamp, "level": level, "module": module, "message": message,
                    "raw": line, "status_code": status_code, "latency": latency_ms, "env": env
                })

            except json.JSONDecodeError as e:
                 data.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "level": "PARSE_ERROR", "module": "parser",
                    "message": f"JSON Decode Error line {i+1}: {e}", "raw": line[:200],
                    "status_code": "N/A", "latency": 0.0, "env": "parser_error", "datetime": pd.NaT })
            except Exception as e:
                 data.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "level": "PARSE_ERROR", "module": "parser",
                    "message": f"General Parsing Error line {i+1}: {e}", "raw": line[:200],
                    "status_code": "N/A", "latency": 0.0, "env": "parser_error", "datetime": pd.NaT })

    # --- 2. Try CSV Format (Only if JSON wasn't detected) ---
    if log_format_detected != "JSON":
        is_csv = is_csv_file(filename=filename, content=logs)
        if is_csv:
            log_format_detected = "CSV"
            try:
                log_data_str = "\n".join(filter(None, [str(l).strip() for l in logs]))
                df_temp = pd.read_csv(io.StringIO(log_data_str), on_bad_lines='warn') # 'warn' instead of 'error' or 'skip'
                df_temp.columns = [col.strip().lower().replace(' ', '_') for col in df_temp.columns]

                for _, row in df_temp.iterrows():
                    entry = {}
                    entry["timestamp"] = str(row.get('timestamp', row.get('time', '')))
                    entry["level"] = str(row.get('level', row.get('log_level', 'INFO'))).upper()
                    entry["module"] = str(row.get('module', row.get('service', row.get('api_id', row.get('components', 'unknown')))))
                    entry["status_code"] = str(row.get('status_code', row.get('status', 'N/A')))
                    entry["env"] = str(row.get('env', row.get('environment', 'unknown')))

                    latency_ms = pd.to_numeric(row.get('latency_ms', row.get('latency', row.get('response_time_ms', 0))), errors='coerce')
                    entry["latency"] = float(latency_ms) if pd.notna(latency_ms) else 0.0

                    entry["message"] = str(row.get('message', ''))
                    if not entry["message"]:
                         details = [f"{k}:{v}" for k, v in row.items() if k not in ['timestamp', 'time', 'level', 'log_level', 'module', 'service', 'api_id', 'components', 'status_code', 'status', 'env', 'environment', 'latency_ms', 'latency', 'response_time_ms', 'message']]
                         entry["message"] = f"Status:{entry['status_code']} Latency:{entry['latency']:.0f}ms Details:[{','.join(details)}]"

                    if entry['level'] == 'INFO' and 'level' not in df_temp.columns and 'log_level' not in df_temp.columns:
                        if pd.to_numeric(row.get('error', 0), errors='coerce') == 1: entry['level'] = 'ERROR'
                        elif str(entry['status_code']).startswith(('4', '5')): entry['level'] = 'WARNING' if str(entry['status_code']).startswith('4') else 'ERROR'

                    entry["raw"] = ','.join(map(str, row.values))
                    data.append(entry)

            except Exception as e:
                print(f"[Warning] CSV parsing failed: {e}. Trying generic.")
                log_format_detected = None
                data = []

    # --- 3. Fallback to Generic Line Parsing ---
    if log_format_detected is None and logs:
        ts_patterns = [
            r'\[?(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]?', # ISOish
            r'\[?(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\]?', # Syslog style
            r'\[?(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})\]?' # Apache common log format
        ]
        level_patterns = [
            (r'\b(ERROR|ERR|CRITICAL|FATAL)\b', "ERROR"),
            (r'\b(WARN|WARNING)\b', "WARNING"),
            (r'\b(DEBUG)\b', "DEBUG"),
            (r'\b(INFO|NOTICE)\b', "INFO")
        ]
        module_patterns = [ r'\[([\w\-.]+)\]', r'\s([\w\-]+):', r'^([\w\-]+)\s' ] # Added pattern for start of line module
        status_patterns = [ r'(?:status|code|status_code)\W+(\d{3})', r'\s(\d{3})\s' ] # Added pattern for space-separated code
        latency_patterns = [ r'(\d+\.?\d*)\s?ms', r'(\d+\.?\d*)\s?milliseconds' ]
        env_patterns = [ r'\[(production|prod|staging|stag|development|dev|test)\]' ]

        for log_line in logs:
            line = str(log_line).strip()
            if not line: continue

            entry = {"raw": line, "level": "INFO", "module": "unknown", "status_code": "N/A", "latency": 0.0, "env": "unknown", "timestamp": "", "message": line}

            # Extract Timestamp
            ts_found = False
            for pattern in ts_patterns:
                match = re.search(pattern, line)
                if match: entry["timestamp"] = match.group(1); ts_found = True; break

            # Extract Level
            level_found = False
            for pattern, lvl in level_patterns:
                if re.search(pattern, line, re.IGNORECASE): entry["level"] = lvl; level_found = True; break

            # Extract Module
            module_found = False
            for pattern in module_patterns:
                matches = re.findall(pattern, line)
                for m in matches:
                    candidate = m.strip()
                    if candidate.upper() not in ["ERROR", "WARN", "INFO", "DEBUG", "WARNING", "CRITICAL", "FATAL", "NOTICE"] and len(candidate) > 1:
                        entry["module"] = candidate
                        module_found = True
                        break
                if module_found: break

            # Extract Status Code
            for pattern in status_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match: entry["status_code"] = match.group(1); break

            # Extract Latency
            for pattern in latency_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try: entry["latency"] = float(match.group(1)); break
                    except ValueError: pass # Ignore if conversion fails

            # Extract Environment
            for pattern in env_patterns:
                 match = re.search(pattern, line, re.IGNORECASE)
                 if match: entry["env"] = match.group(1).lower(); break

            # Basic message cleanup: Remove extracted parts if found precisely
            msg_temp = line
            if ts_found and entry["timestamp"] in msg_temp: msg_temp = msg_temp.replace(entry["timestamp"], '', 1)
            if level_found and f"[{entry['level']}]" in msg_temp: msg_temp = msg_temp.replace(f"[{entry['level']}]", '', 1)
            if module_found and f"[{entry['module']}]" in msg_temp: msg_temp = msg_temp.replace(f"[{entry['module']}]", '', 1)

            entry["message"] = msg_temp.strip(' []-:') # Strip common separators

            data.append(entry)

    # --- Final DataFrame Creation and Cleanup ---
    if data:
        df_extracted = pd.DataFrame(data)
        for col in required_columns:
            if col not in df_extracted.columns:
                 if col == 'latency': df_extracted[col] = 0.0
                 elif col == 'datetime': df_extracted[col] = pd.NaT
                 elif col == 'level': df_extracted[col] = 'INFO' # Default level if missing
                 elif col == 'status_code': df_extracted[col] = 'N/A'
                 else: df_extracted[col] = ''

        df_extracted['latency'] = pd.to_numeric(df_extracted['latency'], errors='coerce').fillna(0.0)
        df_extracted['status_code'] = df_extracted['status_code'].astype(str)
        df_extracted['datetime'] = df_extracted['timestamp'].apply(parse_timestamp)

        # Ensure 'message' exists and is not empty, fallback to raw
        if 'message' not in df_extracted.columns:
            df_extracted['message'] = df_extracted['raw']
        else:
            df_extracted['message'] = df_extracted['message'].fillna(df_extracted['raw'])
            df_extracted.loc[df_extracted['message'].str.strip() == '', 'message'] = df_extracted['raw']

        # Ensure all required columns are present before returning
        for col in required_columns:
            if col not in df_extracted.columns:
                # Add missing columns with appropriate defaults if somehow missed
                if col == 'latency': df_extracted[col] = 0.0
                elif col == 'datetime': df_extracted[col] = pd.NaT
                elif col == 'level': df_extracted[col] = 'INFO'
                elif col == 'status_code': df_extracted[col] = 'N/A'
                elif col == 'raw': df_extracted[col] = '' # Should have raw, but fallback
                else: df_extracted[col] = ''

        return df_extracted[required_columns]
    else:
        print("[Error] Failed to parse any log entries from the provided source.")
        return pd.DataFrame(columns=required_columns)


# --- LLM & Ollama Functions ---
@st.cache_data(ttl=60)
def is_ollama_available(api_url):
    """Checks if Ollama service is available and API endpoint is valid."""
    if not api_url or not isinstance(api_url, str) or not api_url.endswith("/api/generate"): return False, "Invalid URL format (must end with /api/generate)"
    base_url = api_url.replace("/api/generate", "")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5) # Reduced timeout
        if response.status_code == 200:
            return True, "Connected"
        else:
             return False, f"Ollama responded with status {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Connection timed out"
    except requests.exceptions.ConnectionError:
         return False, "Connection refused (is Ollama running?)"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {type(e).__name__}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

@st.cache_data(ttl=300) # Cache model list for 5 minutes
def get_ollama_models(api_url):
    """Gets list of available models from Ollama."""
    models = []
    if not api_url or not isinstance(api_url, str) or not api_url.endswith("/api/generate"): return models
    base_url = api_url.replace("/api/generate", "")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            models = sorted([model["name"] for model in models_data])
        else:
            print(f"Warning: Failed to get Ollama models (Status {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not connect to Ollama to get models: {e}")
    return models

def query_ollama(prompt, model_name, api_url):
    """Queries the local Ollama API."""
    if not api_url or not model_name: return "Error: Ollama API URL or model name not configured."
    is_available, status_msg = is_ollama_available(api_url)
    if not is_available: return f"Error: Cannot connect to Ollama. {status_msg}"

    try:
        response = requests.post(
            api_url,
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=90
        )
        response.raise_for_status()
        resp_json = response.json()
        if "response" in resp_json:
            return resp_json["response"]
        elif "error" in resp_json:
             return f"Error from Ollama: {resp_json['error']}"
        else:
             return "Error: Unexpected response format from Ollama."

    except requests.exceptions.Timeout:
        return f"Error: Ollama request timed out after 90 seconds."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama API ({api_url}): {e}"
    except Exception as e:
        return f"Error processing Ollama response: {e}"

def query_remote_llm(prompt, model, api_key):
    """Queries remote LLM API (OpenRouter)."""
    if not api_key or api_key.startswith("sk-or-v1-...") or not model:
         return "Error: OpenRouter API Key or model not configured."
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=90
        )
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and result['choices']:
            return result['choices'][0]['message']['content']
        elif 'error' in result:
             err_details = result['error']
             return f"Error from OpenRouter: {err_details.get('message', str(err_details))}"
        else:
            return f"Error: Unexpected response format from OpenRouter: {result}"
    except requests.exceptions.Timeout:
         return f"Error: OpenRouter request timed out after 90 seconds."
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 'N/A'
        if status_code == 401: return "Error: Invalid OpenRouter API Key."
        if status_code == 402: return "Error: OpenRouter Quota Exceeded or Payment Required."
        if status_code == 404: return f"Error: OpenRouter Model Not Found ({model})."
        return f"Error querying OpenRouter API ({model}, Status: {status_code}): {e}"
    except Exception as e:
        return f"Error processing OpenRouter response: {e}"


# --- Specific LLM Analysis Task Functions ---
def _get_summary_hash(summary_dict):
    """Helper to create a hashable representation of a dictionary for caching."""
    return hash(json.dumps(summary_dict, sort_keys=True))

@st.cache_data(show_spinner="🧠 Performing Holistic Analysis...")
def perform_holistic_analysis(_log_df_summary_hash, _clusters_summary_str, use_ollama=True, ollama_model=None, ollama_url=None, remote_model=None, api_key=None):
    # Retrieve the actual summary data from session state using the hash (or pass it directly if feasible)
    # For simplicity here, we'll assume the calling code makes the data available or reconstructs it
    # In a real app, you might pass the actual data and hash it inside if caching complex objects is tricky
    log_summary_data_str = json.dumps(st.session_state.get('log_df_summary', {}), indent=2) # Fetch current summary for prompt

    prompt = f"""
# System Log Analysis Report

Analyze the provided log statistics and cluster summaries to assess overall system health.

## Overall Statistics (Summary Hash: {_log_df_summary_hash})
{log_summary_data_str}

## Cluster Summaries
{_clusters_summary_str}

## Analysis Task
Based *only* on the statistics and cluster summaries provided above:
1.  Provide a brief (2-3 sentences) assessment of the overall system health (e.g., stable, issues detected, critical problems).
2.  Identify the top 1-2 most concerning clusters or patterns observed (cite cluster IDs if possible). Explain why they are concerning (e.g., high error rate, specific error types, critical module involvement).
3.  Suggest 1-2 high-level areas to investigate further based *only* on this summary data.
Keep the response concise, actionable, and focused on the provided data. Avoid speculation beyond the data. Use Markdown for formatting.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

@st.cache_data(show_spinner="🧠 Performing Comparative Analysis...")
def perform_comparative_analysis(_error_profiles_str, use_ollama=True, ollama_model=None, ollama_url=None, remote_model=None, api_key=None):
    prompt = f"""
# Comparative Error Analysis

Compare the error patterns across these log clusters based *only* on the provided error profiles:

## Error Profiles by Cluster
{_error_profiles_str}

## Analysis Task
1.  Which cluster appears to have the most critical or impactful errors based on the summaries (count, rate, modules, status codes, latency)? Explain briefly why.
2.  Are there any notable similarities or differences in the types of errors (modules, status codes, sample messages) between clusters?
3.  Based *only* on this information, can you hypothesize if errors in one cluster might be related to or causing errors in another? (e.g., 'Errors in Cluster X (database) might be causing timeouts seen in Cluster Y (api-gateway)'). Be speculative but grounded in the data.
Keep the response concise and focused on comparing the provided profiles. Use Markdown for formatting.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

@st.cache_data(show_spinner="🧠 Performing Temporal Analysis...")
def perform_temporal_analysis(_hourly_stats_summary_str, _anomalous_hours_str, use_ollama=True, ollama_model=None, ollama_url=None, remote_model=None, api_key=None):
    prompt = f"""
# Temporal Log Analysis

Analyze the temporal patterns based *only* on the provided hourly statistics summary and identified anomalous hours:

## Hourly Statistics Summary (Sample or Aggregated Data)
{_hourly_stats_summary_str}

## Anomalous Hours (e.g., High Error Rate or Volume)
{_anomalous_hours_str}

## Analysis Task
1.  Describe the main time-based pattern observed (e.g., errors peak during specific hours, volume increases during business hours, constant load).
2.  Highlight the significance of the anomalous hours identified. What might be happening during these times based on the data?
3.  Suggest 1-2 potential reasons for these temporal patterns (e.g., peak user load, batch jobs, external system dependencies, specific types of errors occurring at certain times).
Keep the response concise and focused on interpreting the time-based data provided. Use Markdown for formatting.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

@st.cache_data(show_spinner="🧠 Analyzing Cluster Summary...")
def analyze_cluster_summary(cluster_id, _cluster_summary_str, use_ollama=True, ollama_model=None, ollama_url=None, remote_model=None, api_key=None):
    prompt = f"""
# Log Cluster Analysis (Cluster {cluster_id})

Analyze the following summary for Log Cluster {cluster_id}:

## Cluster Summary
{_cluster_summary_str}

## Analysis Task
Based *only* on the summary provided above for Cluster {cluster_id}:
1.  What is the primary characteristic or theme of this cluster? (e.g., dominated by specific errors like 'database connection timeout', mostly high latency warnings for 'api-gateway', successful 'auth' operations, mixed INFO logs). Be specific based on top modules, status codes, and sample messages.
2.  What are the 1-2 most likely issues or points of interest indicated by this summary? What makes them interesting (e.g., high error count, critical module, unusual status code)?
3.  Suggest 1-2 specific, actionable next steps for investigating this cluster further (e.g., 'Check database connection pool settings', 'Examine API gateway logs around the timestamps of high latency', 'Filter for specific error message in this cluster').
Keep the response concise and directly related to the provided summary data. Use Markdown for formatting.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

@st.cache_data(show_spinner="🧠 Explaining Log Entry...")
def explain_single_log(log_entry_raw, use_ollama=True, ollama_model=None, ollama_url=None, remote_model=None, api_key=None):
    """Explain a single raw log entry using the LLM."""
    prompt = f"""
# Explain Log Entry

Explain the following log entry in simple terms. Focus on what it likely means, potential causes, and possible impact.

Log Entry:
```
{log_entry_raw}
```

## Explanation Task:
1.  **Meaning:** What event or status is this log message reporting? (1-2 sentences)
2.  **Potential Cause(s):** What are 1-3 common technical reasons why this message might appear?
3.  **Potential Impact:** What could be the consequence of this event for users or the system? (e.g., slow response, failed request, data inconsistency, no impact)
4.  **Suggested Next Step:** What is one simple, concrete action an engineer could take first to investigate this specific log message? (e.g., check service status, look at related logs, verify configuration)

Keep the explanation clear, concise (use bullet points or numbered lists), and actionable. Avoid overly technical jargon where possible. Base the explanation *only* on the content of the log entry provided. Use Markdown for formatting.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

# --- SSE Client Thread Function ---
def sse_client_thread(url, log_q, stop_event):
    """Connects to SSE endpoint, puts raw log strings into the queue."""
    headers = {'Accept': 'text/event-stream'}
    retry_delay = 1
    max_retry_delay = 30
    session_id = random.randint(1000, 9999)
    print(f"[{datetime.now()}] [SSE-{session_id}] Thread: Starting...")
    st.session_state['sse_last_error'] = None # Clear previous errors

    while not stop_event.is_set():
        try:
            st.session_state['sse_connection_status'] = "connecting"
            print(f"[{datetime.now()}] [SSE-{session_id}] Thread: Connecting to {url}...")
            response = requests.get(url, stream=True, headers=headers, timeout=(10, 60)) # (connect_timeout, read_timeout)
            response.raise_for_status() # Check for 4xx/5xx errors immediately
            print(f"[{datetime.now()}] [SSE-{session_id}] Thread: Connection successful (Status: {response.status_code}).")
            st.session_state['sse_connection_status'] = "connected"
            st.session_state['sse_last_error'] = None # Clear error on success
            retry_delay = 1

            for line in response.iter_lines():
                if stop_event.is_set():
                    print(f"[{datetime.now()}] [SSE-{session_id}] Thread: Stop event detected during line iteration.")
                    break

                if line:
                    decoded_line = line.decode('utf-8', errors='replace')
                    if decoded_line.startswith('data:'):
                        log_data = decoded_line[len('data:'):].strip()
                        if log_data:
                            log_q.put(log_data)

            if not stop_event.is_set():
                print(f"[{datetime.now()}] [SSE-{session_id}] Thread: Stream ended by server.")
                st.session_state['sse_connection_status'] = "disconnected"
                st.session_state['sse_last_error'] = "Stream ended by server."
                break # Exit outer loop, requires manual reconnect

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection Error: {e}"
            print(f"[{datetime.now()}] [SSE-{session_id}] Thread: {error_msg}. Retrying in {retry_delay}s...")
            st.session_state['sse_connection_status'] = "error"
            st.session_state['sse_last_error'] = error_msg
        except requests.exceptions.Timeout as e:
            error_msg = f"Connection Timeout: {e}"
            print(f"[{datetime.now()}] [SSE-{session_id}] Thread: {error_msg}. Retrying in {retry_delay}s...")
            st.session_state['sse_connection_status'] = "error"
            st.session_state['sse_last_error'] = error_msg
        except requests.exceptions.HTTPError as e:
             error_msg = f"HTTP Error: {e.response.status_code} {e.response.reason} at {url}"
             print(f"[{datetime.now()}] [SSE-{session_id}] Thread: {error_msg}. Retrying in {retry_delay}s...")
             st.session_state['sse_connection_status'] = "error"
             st.session_state['sse_last_error'] = error_msg
             if e.response.status_code in [404, 401, 403]:
                 print(f"[{datetime.now()}] [SSE-{session_id}] Thread: Unrecoverable HTTP error {e.response.status_code}. Stopping thread.")
                 stop_event.set() # Stop the thread
                 break
        except requests.exceptions.RequestException as e:
            error_msg = f"Request Exception: {e}"
            print(f"[{datetime.now()}] [SSE-{session_id}] Thread: {error_msg}. Retrying in {retry_delay}s...")
            st.session_state['sse_connection_status'] = "error"
            st.session_state['sse_last_error'] = error_msg
        except Exception as e:
            error_msg = f"Unexpected Error: {type(e).__name__} - {e}"
            print(f"[{datetime.now()}] [SSE-{session_id}] Thread: {error_msg}. Retrying in {retry_delay}s...")
            st.session_state['sse_connection_status'] = "error"
            st.session_state['sse_last_error'] = error_msg

        # Wait before retrying, check stop event periodically
        if not stop_event.is_set():
            slept_time = 0
            sleep_interval = 0.5
            while slept_time < retry_delay and not stop_event.is_set():
                 time.sleep(sleep_interval)
                 slept_time += sleep_interval
            retry_delay = min(retry_delay * 1.5, max_retry_delay)

        if stop_event.is_set():
             print(f"[{datetime.now()}] [SSE-{session_id}] Thread: Stop event detected during retry wait.")
             break

    # Cleanup
    final_status = "disconnected" if not st.session_state.get('sse_last_error') else "error"
    st.session_state['sse_connection_status'] = final_status
    st.session_state['sse_thread'] = None
    print(f"[{datetime.now()}] [SSE-{session_id}] Thread: Stopped. Final Status: {final_status}")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Data Source")
    data_source_options = ["Real-time Stream", "Upload Log File", "Sample Logs"] # Stream first
    try:
        current_ds = st.session_state.get('current_data_source', "Real-time Stream")
        default_ds_index = data_source_options.index(current_ds)
    except ValueError:
        default_ds_index = 0

    data_source = st.radio(
        "Choose data source",
        options=data_source_options,
        key="data_source_radio",
        horizontal=True,
        index=default_ds_index
    )

    # --- Handle Data Source Change ---
    if data_source != st.session_state.get('current_data_source'):
        print(f"Switching data source from {st.session_state.get('current_data_source')} to {data_source}")
        # Stop existing stream if switching away
        if st.session_state.get('current_data_source') == "Real-time Stream" and st.session_state.get('sse_thread') is not None:
            print("Stopping SSE thread due to data source change.")
            if st.session_state.get('sse_stop_event'):
                st.session_state['sse_stop_event'].set()
            thread = st.session_state.get('sse_thread')
            if thread and thread.is_alive():
                thread.join(timeout=1.5)
            st.session_state['sse_connection_status'] = "disconnected"
            st.session_state['sse_thread'] = None
            st.session_state['sse_stop_event'] = None
            st.session_state['log_queue'] = queue.Queue()
            st.session_state['sse_last_error'] = None

        # Clear log data and analysis results
        st.session_state['log_df'] = pd.DataFrame(columns=required_columns)
        st.session_state['current_data_source'] = data_source
        # Clear specific analysis results and flags
        st.session_state.pop('clusters_summary', None)
        st.session_state.pop('error_profiles', None)
        st.session_state.pop('log_df_summary', None)
        st.session_state['last_uploaded_file_info'] = None # Use new unique ID logic
        st.session_state['sample_logs_loaded'] = False
        st.session_state['log_page'] = 1 # Reset pagination
        st.session_state['log_explorer_filter_hash'] = None
        # Safely attempt to remove cluster column if it exists
        if 'log_df' in st.session_state and 'cluster' in st.session_state.log_df.columns:
            st.session_state.log_df = st.session_state.log_df.drop(columns=['cluster'], errors='ignore')

        st.rerun()

    # --- Source Specific UI ---
    if data_source == "Upload Log File":
        uploaded_file = st.file_uploader("📤 Upload log file (.log, .txt, .csv, .jsonl)", type=["log", "txt", "csv", "jsonl"])

        current_file_info = None
        if uploaded_file is not None:
            # Create a tuple of properties to identify this specific upload
            current_file_info = (uploaded_file.name, uploaded_file.size)

        # Process ONLY if the file info is new compared to the last processed one
        if current_file_info is not None and current_file_info != st.session_state.get('last_uploaded_file_info'):
            print(f"New file uploaded or selected: {current_file_info[0]} ({current_file_info[1]} bytes)")
            try:
                content_bytes = uploaded_file.read()
                # It's better to decode when needed, but keep original bytes if decoding fails
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    print("UTF-8 decoding failed, trying latin-1.")
                    try:
                         content = content_bytes.decode("latin-1")
                    except UnicodeDecodeError:
                         st.error("Failed to decode file content with UTF-8 or latin-1.")
                         content = None # Indicate decoding failure

                if content:
                    logs = content.splitlines()
                    file_name = uploaded_file.name
                    log_df_temp = extract_components(tuple(logs), filename=file_name) # Pass tuple
                    if not log_df_temp.empty:
                        st.session_state['log_df'] = log_df_temp
                        # Store info of the successfully processed file
                        st.session_state['last_uploaded_file_info'] = current_file_info
                        st.success(f"Loaded and parsed {len(log_df_temp)} entries from {file_name}")
                        # Rerun needed to update the main UI with the new dataframe
                        st.rerun()
                    else:
                        st.error("Parsing failed or file contained no valid log entries.")
                        # Clear last info if parsing failed for this new file
                        st.session_state['last_uploaded_file_info'] = None
                else:
                    # This case might happen if decoding failed completely or file was empty
                    st.warning("Uploaded file appears empty or could not be decoded.")
                    st.session_state['last_uploaded_file_info'] = None
            except Exception as e:
                st.error(f"Error reading or parsing uploaded file: {e}")
                st.session_state['last_uploaded_file_info'] = None
        elif st.session_state.get('log_df', pd.DataFrame()).empty:
             st.info("Upload a log file to begin analysis.")
        elif uploaded_file is None and st.session_state.get('last_uploaded_file_info') is not None:
             print("File uploader cleared.")
             st.session_state['last_uploaded_file_info'] = None
        if uploaded_file is None and st.session_state.get('log_df', pd.DataFrame()).empty:
            st.info("Upload a log file to begin analysis.")


    elif data_source == "Sample Logs":
        # Load sample only if DataFrame is empty for this source AND not already loaded
        if st.session_state.get('log_df', pd.DataFrame()).empty and not st.session_state.get('sample_logs_loaded', False):
            with st.spinner("Loading sample logs..."):
                logs = fetch_static_logs()
            if logs:
                # Infer filename hint for parsing based on first few lines
                file_name_hint = "sample_logs.log" # Default fallback
                if isinstance(logs, list) and len(logs) > 1:
                    if is_csv_file(content=logs[:5]): # Check first 5 lines
                        file_name_hint = "sample_logs.csv"
                    elif str(logs[0]).strip().startswith('{'):
                        file_name_hint = "sample_logs.jsonl"

                log_df_temp = extract_components(tuple(logs), filename=file_name_hint) # Pass tuple and hint
                if not log_df_temp.empty:
                    st.session_state['log_df'] = log_df_temp
                    st.session_state['sample_logs_loaded'] = True # Mark as loaded
                    st.success(f"Loaded and parsed {len(log_df_temp)} sample log entries (format inferred: {file_name_hint.split('.')[-1]}).")
                    st.rerun() # Rerun after load
                else:
                    st.error("Failed to parse sample logs.")
            else:
                 st.info("No sample logs were loaded (check logs_backend.py or api_logs.csv).")
        elif not st.session_state.get('log_df', pd.DataFrame()).empty:
             st.success(f"Using {len(st.session_state.log_df)} previously loaded sample logs.")
        # Reset 'loaded' flag if df becomes empty again for this source
        if st.session_state.get('log_df', pd.DataFrame()).empty:
             st.session_state['sample_logs_loaded'] = False

    elif data_source == "Real-time Stream":
        st.markdown("### 📡 Real-time Settings")
        sse_url = st.text_input("SSE Log Stream URL", value=DEFAULT_SSE_LOG_URL, key="sse_url_input")

        conn_status = st.session_state.get('sse_connection_status', 'disconnected')
        status_class = f"status-{conn_status}"
        status_text = conn_status.capitalize().replace("_", " ")

        connect_button_text = "Disconnect" if conn_status in ["connected", "connecting"] else "Connect"
        connect_button_disabled = (conn_status == "connecting")

        st.markdown(f"Status: **{status_text}** <span class='status-indicator {status_class}'></span>", unsafe_allow_html=True)

        # Display last error if status is 'error'
        if conn_status == 'error' and st.session_state.get('sse_last_error'):
            st.error(f"Connection Error: {st.session_state.get('sse_last_error')}")

        if st.button(connect_button_text, key="sse_connect_button", disabled=connect_button_disabled, use_container_width=True):
            if conn_status in ["connected", "connecting"]: # DISCONNECT Action
                print("Disconnect button pressed.")
                if st.session_state.get('sse_stop_event'):
                    st.session_state['sse_stop_event'].set()
                time.sleep(0.5)
                st.session_state['sse_connection_status'] = "disconnected"
                st.session_state['sse_last_error'] = None # Clear error on manual disconnect
                st.rerun() # Force UI update
            else: # CONNECT Action
                print(f"Connect button pressed. Attempting to connect SSE thread to: {sse_url}")
                if not sse_url or not sse_url.startswith(("http://", "https://")):
                    st.error("Invalid SSE URL provided. Must start with http:// or https://")
                else:
                    # Clear previous logs for a fresh stream
                    st.session_state['log_df'] = pd.DataFrame(columns=required_columns)
                    st.session_state['log_queue'] = queue.Queue()
                    st.session_state['sse_last_error'] = None # Clear previous error
                    stop_event = threading.Event()
                    st.session_state['sse_stop_event'] = stop_event
                    thread = threading.Thread(
                        target=sse_client_thread,
                        args=(sse_url, st.session_state['log_queue'], stop_event),
                        daemon=True
                    )
                    st.session_state['sse_thread'] = thread
                    st.session_state['sse_connection_status'] = "connecting"
                    thread.start()
                    print("SSE thread started.")
                    time.sleep(0.5) # Give thread time to attempt connection
                    st.rerun() # Update UI to show 'connecting' status

        st.caption(f"Received logs: {len(st.session_state.get('log_df', pd.DataFrame()))}")
        max_logs = st.number_input("Max logs to keep in memory", min_value=100, max_value=50000,
                                   value=st.session_state.get('max_stream_logs', 5000), step=100, key="max_stream_logs_input",
                                   help="Limits memory usage during long streams. Oldest logs are discarded.")
        if max_logs != st.session_state.get('max_stream_logs'):
            st.session_state['max_stream_logs'] = max_logs

    # --- Common Settings ---
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    current_log_count = len(st.session_state.get('log_df', pd.DataFrame()))
    min_logs_for_cluster = 10
    if current_log_count >= min_logs_for_cluster :
        max_possible_clusters = max(2, current_log_count // 5) # Heuristic
        max_clusters_allowed = min(15, max_possible_clusters) # Upper limit
        default_clusters = min(st.session_state.get('n_clusters', 4), max_clusters_allowed)
        default_clusters = max(2, default_clusters)

        n_clusters = st.slider(
            "Number of log clusters", min_value=2, max_value=max_clusters_allowed,
            value=default_clusters,
            help=f"Adjust log grouping (requires ≥ {min_logs_for_cluster} logs). Max: {max_clusters_allowed}",
            key="n_clusters_slider",
            disabled=(current_log_count < min_logs_for_cluster)
        )
        if n_clusters != st.session_state.get('n_clusters'):
            st.session_state['n_clusters'] = n_clusters
    else:
        st.markdown(f"_(Clustering requires ≥ {min_logs_for_cluster} logs. Current: {current_log_count})_")

    # --- LLM Settings UI ---
    st.markdown("### 🤖 LLM Settings")
    ollama_api_for_check = st.session_state.get('ollama_url', OLLAMA_API_URL)
    ollama_is_ready, ollama_status_msg = is_ollama_available(ollama_api_for_check)
    default_provider_index = 0 if ollama_is_ready else 1

    llm_provider = st.radio("Choose LLM Provider", ["Local Ollama", "Remote OpenRouter"],
                            index=default_provider_index,
                            horizontal=True, key="llm_provider_radio")
    use_ollama_choice = (llm_provider == "Local Ollama")
    if use_ollama_choice != st.session_state.get('use_ollama'):
        st.session_state['use_ollama'] = use_ollama_choice

    if st.session_state.get('use_ollama'):
        ollama_url_input = st.text_input("Ollama API URL", value=st.session_state.get('ollama_url', OLLAMA_API_URL), key="ollama_url_input",
                                help="URL of your local Ollama /api/generate endpoint")
        if ollama_url_input != st.session_state.get('ollama_url'):
            st.session_state['ollama_url'] = ollama_url_input
            ollama_is_ready, ollama_status_msg = is_ollama_available(ollama_url_input) # Recheck

        if ollama_is_ready:
            st.success(f"✅ Ollama Connected ({ollama_status_msg})")
            available_models = get_ollama_models(st.session_state['ollama_url'])
            if available_models:
                preferred_models = ["llama3", "llama3:8b", "llama3.1", "llama3.1:8b", "mistral", "mistral:7b", "phi3", "phi3:mini"]
                default_model = next((m for m in preferred_models if m in available_models), available_models[0])
                current_model = st.session_state.get('ollama_model', default_model)
                if current_model not in available_models: current_model = default_model
                try: current_model_index = available_models.index(current_model)
                except ValueError: current_model_index = 0

                selected_ollama_model = st.selectbox(
                    "Select Ollama Model", options=available_models, index=current_model_index, key="ollama_model_select"
                )
                if selected_ollama_model != st.session_state.get('ollama_model'):
                    st.session_state['ollama_model'] = selected_ollama_model
            else:
                st.warning("Ollama running, but no models found/listed? Pull one (e.g., `ollama pull llama3`)")
                fallback_ollama_model = st.text_input("Enter Ollama Model Name", value=st.session_state.get('ollama_model', "llama3"), key="ollama_model_input_fallback")
                if fallback_ollama_model != st.session_state.get('ollama_model'):
                     st.session_state['ollama_model'] = fallback_ollama_model
        else:
            st.error(f"❌ Ollama connection failed: {ollama_status_msg}")
            error_ollama_model = st.text_input("Ollama Model Name (if running)", value=st.session_state.get('ollama_model', "llama3"), key="ollama_model_input_error")
            if error_ollama_model != st.session_state.get('ollama_model'):
                 st.session_state['ollama_model'] = error_ollama_model
    else: # Remote OpenRouter
        api_key_input = st.text_input("OpenRouter API Key", value=st.session_state.get('api_key', OPENROUTER_API_KEY), type="password", key="openrouter_api_key_input", help="Get from OpenRouter.ai")
        if api_key_input != st.session_state.get('api_key'):
             st.session_state['api_key'] = api_key_input

        remote_models = [
            "mistralai/mistral-7b-instruct:free", "meta-llama/llama-3.1-8b-instruct:free", "google/gemini-flash-1.5",
            "anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku", "google/gemma-2-9b-it", "microsoft/wizardlm-2-8x22b"
        ]
        default_remote_model = "mistralai/mistral-7b-instruct:free"
        current_remote_model = st.session_state.get('llm_model', default_remote_model)
        if current_remote_model not in remote_models: current_remote_model = default_remote_model
        try: current_remote_index = remote_models.index(current_remote_model)
        except ValueError: current_remote_index = 0

        selected_remote_model = st.selectbox(
            "Select Remote LLM Model", options=remote_models, index=current_remote_index, key="remote_llm_select",
            help="Check model availability and pricing on OpenRouter.ai"
        )
        if selected_remote_model != st.session_state.get('llm_model'):
             st.session_state['llm_model'] = selected_remote_model

    # --- Visualization Settings ---
    st.markdown("---")
    st.markdown("### 🎨 Visualization")
    chart_theme_options = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
    current_theme = st.session_state.get('chart_theme', 'plotly_white')
    try: default_theme_index = chart_theme_options.index(current_theme)
    except ValueError: default_theme_index = 1 # Default to plotly_white

    chart_theme_select = st.selectbox("Chart Theme", chart_theme_options, index=default_theme_index, key="chart_theme_select")
    if chart_theme_select != st.session_state.get('chart_theme'):
        st.session_state['chart_theme'] = chart_theme_select

    # --- About Section ---
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    Analyze logs from static files or real-time streams.
    - **Dashboard:** Key metrics & charts.
    - **Log Explorer:** Filter, search, explain logs with AI.
    - **AI Analysis:** Cluster logs, get AI summaries & comparisons.
    - **Advanced Viz:** Deeper dives into data patterns.
    """)

# --- Process Log Queue ---
UI_UPDATE_INTERVAL_SECONDS = 0.75 # How often the UI should refresh when streaming
MIN_LOGS_FOR_UPDATE = 10 # Minimum number of logs to accumulate before forcing an update

def process_log_queue():
    """Checks queue, parses logs, updates DataFrame, enforces limits."""
    logs_processed_count = 0
    new_log_jsons = []
    if 'log_queue' in st.session_state and not st.session_state['log_queue'].empty():
        start_time = time.time()
        # Process for a short duration or until queue empty
        while not st.session_state['log_queue'].empty() and (time.time() - start_time < 0.1):
            try:
                log_json_str = st.session_state['log_queue'].get_nowait()
                new_log_jsons.append(log_json_str)
                logs_processed_count += 1
            except queue.Empty:
                break

        if new_log_jsons:
            new_df = extract_components(tuple(new_log_jsons), filename=None) # Pass tuple

            if not new_df.empty:
                current_df = st.session_state.get('log_df', pd.DataFrame(columns=required_columns)).copy()
                # Use pd.concat instead of append for efficiency
                # Handle potential FutureWarning about dtype changes
                combined_df = pd.concat([current_df, new_df], ignore_index=True)

                max_logs = st.session_state.get('max_stream_logs', 5000)
                if len(combined_df) > max_logs:
                    combined_df = combined_df.iloc[-max_logs:].reset_index(drop=True)

                st.session_state['log_df'] = combined_df
                return logs_processed_count # Return how many were processed
    return 0 # No logs processed

# --- Trigger Queue Processing and Rerun ---
queue_processed_count = 0
now = time.time()
should_rerun = False
force_rerun_check = False # Flag to force a check even if not connected yet

# Get current status *before* processing queue
previous_status_for_check = st.session_state.get('sse_connection_status')

# Process queue if connected/connecting or items remain
is_streaming_active = previous_status_for_check in ["connected", "connecting"]
queue_has_items = not st.session_state.get('log_queue', queue.Queue()).empty()

if is_streaming_active or queue_has_items:
    queue_processed_count = process_log_queue()

# --- Determine if UI refresh is needed ---
last_update_time = st.session_state.get('last_ui_update_time', 0)
time_since_last_update = now - last_update_time

# Reason 1: Logs were processed and interval passed OR batch size reached
if queue_processed_count > 0:
    force_update_due_to_batch = queue_processed_count >= MIN_LOGS_FOR_UPDATE
    if time_since_last_update > UI_UPDATE_INTERVAL_SECONDS or force_update_due_to_batch:
        should_rerun = True

    # elif is_streaming_active and time_since_last_update > UI_UPDATE_INTERVAL_SECONDS:
    #      should_rerun = True

    # if should_rerun:
    #     st.session_state['last_ui_update_time'] = now # Record the time of this UI update
current_status_after_processing = st.session_state.get('sse_connection_status') # Check status *after* potential update by thread
if not should_rerun and current_status_after_processing in ["connected", "connecting"] and time_since_last_update > UI_UPDATE_INTERVAL_SECONDS:
    should_rerun = True
if previous_status_for_check == "connecting" and current_status_after_processing != "connecting":
     should_rerun = True

if should_rerun:
    st.session_state['last_ui_update_time'] = now # Record the time of this UI update


# --- Main App Area ---
# Retrieve current state potentially updated by queue processing
log_df = st.session_state.get('log_df', pd.DataFrame(columns=required_columns))
n_clusters = st.session_state.get('n_clusters', 4)
use_ollama = st.session_state.get('use_ollama', False)
ollama_model = st.session_state.get('ollama_model', None)
ollama_url = st.session_state.get('ollama_url', None)
llm_model = st.session_state.get('llm_model', None)
api_key = st.session_state.get('api_key', None)
chart_theme = st.session_state.get('chart_theme', 'plotly_white')


# --- Display Content ---
if log_df.empty and st.session_state.get('sse_connection_status') not in ["connected", "connecting"]:
    st.info("⬅️ Welcome! Please load log data or connect to a real-time stream using the sidebar.")
elif log_df.empty and st.session_state.get('sse_connection_status') == "connecting":
    st.info("⏳ Connecting to real-time stream...")
elif log_df.empty and st.session_state.get('sse_connection_status') == "connected":
     st.info("📡 Connected to stream. Waiting for log data...")
else:
    # --- Create Tabs ---
    tab_titles = ["📊 Dashboard", "🔍 Log Explorer"]
    if not log_df.empty:
         tab_titles.extend(["🧠 AI Analysis", "📈 Advanced Visualizations"])
    tabs = st.tabs(tab_titles)
    tab_map = {title: tab for title, tab in zip(tab_titles, tabs)}

    # --- TAB 1: Dashboard ---
    with tab_map["📊 Dashboard"]:
        st.markdown("### 📈 Key Metrics")

        # --- Calculate Metrics ---
        # @st.cache_data # Could cache this based on log_df hash if needed
        def calculate_dashboard_metrics(_df):
            metrics = {
                'total_logs': 0, 'error_count': 0, 'warning_count': 0, 'error_rate': 0,
                'warning_rate': 0, 'unique_modules': 0, 'avg_latency': 0.0,
                'log_start_time': None, 'log_end_time': None, 'log_time_span_str': "N/A"
            }
            if _df.empty:
                return metrics

            metrics['total_logs'] = len(_df)
            metrics['error_count'] = int(_df["level"].eq("ERROR").sum()) if "level" in _df.columns else 0
            metrics['warning_count'] = int(_df["level"].eq("WARNING").sum()) if "level" in _df.columns else 0
            metrics['unique_modules'] = _df["module"].nunique() if "module" in _df.columns else 0

            if metrics['total_logs'] > 0:
                metrics['error_rate'] = round(metrics['error_count'] / metrics['total_logs'] * 100, 1)
                metrics['warning_rate'] = round(metrics['warning_count'] / metrics['total_logs'] * 100, 1)

            if "latency" in _df.columns:
                valid_latency = pd.to_numeric(_df["latency"], errors='coerce').fillna(0)
                positive_latency = valid_latency[valid_latency > 0]
                if not positive_latency.empty:
                    metrics['avg_latency'] = positive_latency.mean()

            # Calculate time span
            if 'datetime' in _df.columns and _df['datetime'].notna().any():
                valid_times = _df['datetime'].dropna()
                if not valid_times.empty:
                    metrics['log_start_time'] = valid_times.min()
                    metrics['log_end_time'] = valid_times.max()
                    time_delta = metrics['log_end_time'] - metrics['log_start_time']

                    days = time_delta.days
                    seconds = time_delta.seconds
                    hours, remainder = divmod(seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)

                    if days > 0:
                        metrics['log_time_span_str'] = f"{days}d {hours}h {minutes}m"
                    elif hours > 0:
                        metrics['log_time_span_str'] = f"{hours}h {minutes}m {seconds}s"
                    elif minutes > 0:
                        metrics['log_time_span_str'] = f"{minutes}m {seconds}s"
                    else:
                        metrics['log_time_span_str'] = f"{seconds}s"
                else:
                    metrics['log_time_span_str'] = "No valid times"
            else:
                metrics['log_time_span_str'] = "No timestamp data"

            return metrics

        log_df_summary = calculate_dashboard_metrics(log_df)
        st.session_state['log_df_summary'] = log_df_summary # Store for potential use in AI Analysis

        # Display Metric Cards - Use 2 rows of 3 columns
        col1, col2, col3 = st.columns(3)
        with col1:
             st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['primary']}">
                <div class="metric-label">Total Logs</div>
                <div class="metric-value" style="color: {COLORS['primary']}">{log_df_summary.get('total_logs', 0):,}</div>
                <div class="metric-sub-label">Entries Processed</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['error']}">
                <div class="metric-label">Errors</div>
                <div class="metric-value" style="color: {COLORS['error']}">{log_df_summary.get('error_count', 0)}</div>
                <div class="metric-sub-label">{log_df_summary.get('error_rate', 0)}% of total logs</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
             st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['warning']}">
                <div class="metric-label">Warnings</div>
                <div class="metric-value" style="color: {COLORS['warning']}">{log_df_summary.get('warning_count', 0)}</div>
                <div class="metric-sub-label">{log_df_summary.get('warning_rate', 0)}% of total logs</div>
            </div>
            """, unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['info']}">
                <div class="metric-label">System Up Time</div>
                <div class="metric-value" style="color: {COLORS['info']}; font-size: 20px;">{log_df_summary.get('log_time_span_str', 'N/A')}</div>
                <div class="metric-sub-label">Duration covered by logs</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
             st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['secondary']}">
                <div class="metric-label">Unique Modules</div>
                <div class="metric-value" style="color: {COLORS['secondary']}">{log_df_summary.get('unique_modules', 0)}</div>
                <div class="metric-sub-label">Sources reporting logs</div>
            </div>
            """, unsafe_allow_html=True)
        with col6:
             st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['success']}">
                <div class="metric-label">Avg Latency</div>
                <div class="metric-value" style="color: {COLORS['success']}">{log_df_summary.get('avg_latency', 0.0):.1f} ms</div>
                <div class="metric-sub-label">Avg duration (if available)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("##### Log Level Distribution")
            if "level" in log_df.columns and not log_df["level"].empty:
                level_counts = log_df["level"].value_counts().reset_index()
                level_counts.columns = ["Level", "Count"]
                level_counts['Count'] = pd.to_numeric(level_counts['Count'], errors='coerce').fillna(0)
                color_map = {"ERROR": COLORS["error"], "WARNING": COLORS["warning"], "INFO": COLORS["success"], "DEBUG": COLORS["secondary"], "PARSE_ERROR": "#FFA500"}
                fig = px.pie(level_counts, values='Count', names='Level', color='Level',
                             color_discrete_map=color_map, hole=0.4, template=chart_theme)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else: st.caption("No log level data available.")

        with col_chart2:
            st.markdown("##### Top Modules by Error Count")
            if log_df_summary.get('error_count', 0) > 0 and "module" in log_df.columns:
                module_errors = log_df[log_df['level']=='ERROR']['module'].value_counts().head(10).reset_index()
                module_errors.columns = ['Module', 'Error Count']
                module_errors['Error Count'] = pd.to_numeric(module_errors['Error Count'], errors='coerce').fillna(0)
                fig = px.bar(module_errors, x='Error Count', y='Module', orientation='h',
                            color='Error Count', color_continuous_scale=px.colors.sequential.Reds,
                            template=chart_theme)
                fig.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=300, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else: st.caption("No errors found or module data missing.")

        # Error timeline
        if 'datetime' in log_df.columns and log_df['datetime'].notna().any():
            st.markdown("##### Log Timeline (by Hour)")
            time_df = log_df.dropna(subset=['datetime']).copy()
            if not time_df.empty:
                time_df['hour'] = time_df['datetime'].dt.floor('h') # Group by hour
                hourly_counts = pd.pivot_table(time_df, index='hour', columns='level', aggfunc='size', fill_value=0)
                fig = go.Figure()
                level_order = ["INFO", "DEBUG", "WARNING", "ERROR", "PARSE_ERROR"] # Stack order
                level_colors = {"INFO": COLORS["success"], "DEBUG": COLORS["secondary"], "WARNING": COLORS["warning"], "ERROR": COLORS["error"], "PARSE_ERROR": "#FFA500"}
                for level in level_order:
                    if level in hourly_counts.columns:
                         fig.add_trace(go.Bar(x=hourly_counts.index, y=hourly_counts[level], name=level, marker_color=level_colors.get(level)))
                fig.update_layout(barmode='stack', template=chart_theme,
                                  margin=dict(t=30, b=20, l=20, r=20), height=350,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                  xaxis_title=None, yaxis_title='Log Count per Hour')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No valid timestamp data for timeline after filtering.")
        else:
             st.caption("Waiting for timestamp data for timeline...")

    # --- TAB 2: Log Explorer ---
    if "🔍 Log Explorer" in tab_map:
        with tab_map["🔍 Log Explorer"]:
            st.markdown("### 🔍 Filter & Explore Logs")
            filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

            # Filtering Widgets
            with filter_col1:
                levels = ["All"] + (sorted(log_df["level"].unique().tolist()) if "level" in log_df.columns and not log_df["level"].empty else [])
                selected_level = st.selectbox("Level", levels, key="level_filter")
            with filter_col2:
                modules = ["All"] + (sorted(log_df["module"].unique().tolist()) if "module" in log_df.columns and not log_df["module"].empty else [])
                selected_module = st.selectbox("Module", modules, key="module_filter")
            with filter_col3:
                keyword = st.text_input("Search Keyword (in raw log)", key="keyword_filter", placeholder="e.g., error, timeout, user_id")

            with st.expander("Advanced Filters"):
                adv_col1, adv_col2 = st.columns(2)
                with adv_col1:
                    valid_status = log_df["status_code"].astype(str).unique() if "status_code" in log_df.columns else []
                    status_codes = ["All"] + sorted([s for s in valid_status if s.strip() and s != 'N/A'])
                    selected_status = st.selectbox("Status Code", status_codes, key="status_filter")
                with adv_col2:
                    valid_envs = log_df["env"].astype(str).unique() if "env" in log_df.columns else []
                    envs = ["All"] + sorted([e for e in valid_envs if e.strip() and e != 'unknown'])
                    selected_env = st.selectbox("Environment", envs, key="env_filter")

            # Apply filters
            filtered_df = log_df.copy()
            if selected_level != "All" and "level" in filtered_df.columns: filtered_df = filtered_df[filtered_df["level"] == selected_level]
            if selected_module != "All" and "module" in filtered_df.columns: filtered_df = filtered_df[filtered_df["module"] == selected_module]
            if selected_status != "All" and "status_code" in filtered_df.columns: filtered_df = filtered_df[filtered_df["status_code"] == selected_status]
            if selected_env != "All" and "env" in filtered_df.columns: filtered_df = filtered_df[filtered_df["env"] == selected_env]
            if keyword and "raw" in filtered_df.columns:
                try:
                    filtered_df = filtered_df[filtered_df["raw"].str.contains(re.escape(keyword), case=False, na=False, regex=True)]
                except re.error:
                    st.warning("Invalid regex in search keyword. Treating as plain text.")
                    filtered_df = filtered_df[filtered_df["raw"].str.contains(keyword, case=False, na=False, regex=False)]

            st.markdown(f"#### 📝 Log Entries ({len(filtered_df)} matching filters)")

            # Pagination --- <<< CORRECTED LOGIC >>> ---
            page_size = 20
            total_rows = len(filtered_df)
            total_pages = max(1, (total_rows + page_size - 1) // page_size) # Calculate pages correctly

            # Reset page number if filters change or data source changes
            filter_tuple = (selected_level, selected_module, keyword, selected_status, selected_env, st.session_state.get('current_data_source'))
            current_filter_hash = hash(filter_tuple)

            if current_filter_hash != st.session_state.get('log_explorer_filter_hash'):
                 st.session_state['log_page'] = 1 # Reset page number in state *before* widget reads it
                 st.session_state['log_explorer_filter_hash'] = current_filter_hash

            # Get the potentially reset page number from state
            page_number_from_state = st.session_state.get('log_page', 1)

            # Validate the page number *before* passing it to the widget
            validated_page_number = min(max(1, page_number_from_state), total_pages)

            # Instantiate the widget with the *validated* number.
            page_number_input = st.number_input(
                'Page',
                min_value=1,
                max_value=total_pages,
                step=1,
                value=validated_page_number, # Pass the validated value here
                key="log_page",               # Widget uses this key to read/write state
                label_visibility="collapsed", # Hide label, show only number input and controls
            )
            st.caption(f"Page {page_number_input} of {total_pages} ({total_rows} total entries)")


            # Use the value returned by the widget for calculations *in this run*.
            start_idx = (page_number_input - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows) # Prevent index out of bounds
            paginated_df = filtered_df.iloc[start_idx:end_idx]
            # --- <<< END OF CORRECTED PAGINATION LOGIC >>> ---


            # Display Logs
            if not paginated_df.empty:
                # Use st.container to group each log entry and its button
                for i, (original_index, row) in enumerate(paginated_df.iterrows()):
                    with st.container():
                        level = row.get("level", "INFO")
                        level_class = "info-card"
                        level_color = COLORS['info']
                        if level == "ERROR": level_class, level_color = "error-card", COLORS['error']
                        elif level == "WARNING": level_class, level_color = "warning-card", COLORS['warning']
                        elif level == "DEBUG": level_class, level_color = "info-card", COLORS['secondary']
                        elif level == "PARSE_ERROR": level_class, level_color = "error-card", "#FFA500" # Orange

                        button_key = f"explain_{original_index}_{page_number_input}" # Use widget value for unique key per page view

                        # Display log entry card
                        st.markdown(f"""
                        <div class="log-entry-card {level_class}" style="border-left-color: {level_color};">
                            <div class="log-text">
                               <span style="color: {COLORS['secondary']}; font-weight: normal;">[{row.get('timestamp', 'No Timestamp')}]</span>
                               <span style="font-weight: bold; color:{level_color};"> [{level}]</span>
                               <span style="font-weight: bold; color:{COLORS['primary']};"> [{row.get('module', '?')}]</span>
                               <span> {row.get('message', 'No Message')}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Explain button and explanation area
                        explain_clicked = st.button("Explain with AI", key=button_key, help="Get an AI explanation for this log entry")
                        if explain_clicked:
                            explanation = explain_single_log(
                                row.get('raw', 'Log entry data missing.'),
                                use_ollama=use_ollama, ollama_model=ollama_model, ollama_url=ollama_url,
                                remote_model=llm_model, api_key=api_key
                            )
                            # Display explanation in an info box directly below the button
                            st.info(f"**AI Explanation for log {original_index}:**\n\n{explanation}")

            elif total_rows > 0:
                st.caption("No logs match your filter criteria on the current page.")
            else:
                 st.caption("No logs match your filter criteria, or waiting for stream data.")

    # --- TAB 3: AI Analysis ---
    if "🧠 AI Analysis" in tab_map:
        with tab_map["🧠 AI Analysis"]:
            st.markdown("### 🧠 AI-Powered Log Analysis")
            st.caption("Cluster similar logs to identify patterns and get targeted AI insights. Analysis runs on the currently loaded/streamed data.")

            analysis_disabled = len(log_df) < min_logs_for_cluster
            analysis_help = f"Requires at least {min_logs_for_cluster} logs to run clustering." if analysis_disabled else "Cluster logs and generate AI summaries."

            run_button_col, _ = st.columns([1, 3])
            with run_button_col:
                 run_analysis = st.button("🔄 Run / Update Clustering & Analysis", key="run_analysis_button", disabled=analysis_disabled, help=analysis_help, use_container_width=True)

            if run_analysis:
                with st.spinner("Clustering logs & preparing summaries..."):
                    try:
                        # --- Clustering ---
                        valid_logs_for_clustering = log_df['raw'].dropna().astype(str)
                        if len(valid_logs_for_clustering) < min_logs_for_cluster:
                             raise ValueError(f"Not enough valid log messages ({len(valid_logs_for_clustering)}) for clustering (minimum {min_logs_for_cluster}).")

                        clean_logs = preprocess(valid_logs_for_clustering.tolist()) # Pass list
                        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2, ngram_range=(1, 2))
                        X = vectorizer.fit_transform(clean_logs)

                        if X.shape[0] < 2 or X.shape[1] == 0: # Need at least 2 samples/features
                            raise ValueError(f"TF-IDF Vectorization resulted in insufficient data for clustering. Shape: {X.shape}. Check log content and diversity.")

                        actual_n_clusters = min(n_clusters, X.shape[0])
                        if actual_n_clusters < 2:
                            raise ValueError("Need at least 2 distinct log patterns to form clusters.")
                        if actual_n_clusters < n_clusters:
                            st.warning(f"Reduced number of clusters to {actual_n_clusters} due to limited distinct log patterns.")

                        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
                        labels = kmeans.fit_predict(X)

                        # Add cluster labels back safely using the index from valid_logs_for_clustering
                        temp_cluster_df = pd.DataFrame({'cluster': labels}, index=valid_logs_for_clustering.index)
                        # Use combine_first or update to merge cluster labels, handle existing column
                        if 'cluster' in st.session_state['log_df'].columns:
                             st.session_state['log_df'].update(temp_cluster_df)
                             st.session_state['log_df']['cluster'] = st.session_state['log_df']['cluster'].fillna(-1).astype(int)
                        else:
                             st.session_state['log_df'] = st.session_state['log_df'].join(temp_cluster_df)
                             st.session_state['log_df']['cluster'] = st.session_state['log_df']['cluster'].fillna(-1).astype(int)

                        # --- Prepare Summaries ---
                        clusters_summary_list = []
                        error_profiles = []
                        valid_clusters = sorted([int(c) for c in st.session_state['log_df']['cluster'].unique() if c >= 0])

                        for i in valid_clusters:
                            cluster_logs = st.session_state['log_df'][st.session_state['log_df']["cluster"] == i]
                            total = len(cluster_logs)
                            if total == 0: continue

                            errors = int(cluster_logs["level"].eq("ERROR").sum())
                            warnings = int(cluster_logs["level"].eq("WARNING").sum())
                            error_rate = round(errors / total * 100, 1) if total > 0 else 0.0
                            warning_rate = round(warnings / total * 100, 1) if total > 0 else 0.0

                            top_modules_series = cluster_logs["module"].value_counts().head(3)
                            top_modules = {str(k): int(v) for k, v in top_modules_series.items()}

                            top_status_series = cluster_logs["status_code"][cluster_logs["status_code"] != 'N/A'].value_counts().head(3)
                            top_status = {str(k): int(v) for k, v in top_status_series.items()}

                            cluster_latency = pd.to_numeric(cluster_logs["latency"], errors='coerce').fillna(0)
                            positive_latency = cluster_latency[cluster_latency > 0]
                            avg_latency_py = float(round(positive_latency.mean(), 2)) if not positive_latency.empty else 0.0

                            sample_logs_list = []
                            for lvl in ['ERROR', 'WARNING', 'INFO', 'DEBUG']:
                                level_samples = cluster_logs[cluster_logs['level'] == lvl]['raw'].head(2).tolist()
                                sample_logs_list.extend(level_samples)
                                if len(sample_logs_list) >= 5: break
                            samples = sample_logs_list[:5]

                            cluster_summary = {
                                "cluster_id": i, "total_logs": total, "error_count": errors,
                                "warning_count": warnings, "error_rate": error_rate, "warning_rate": warning_rate,
                                "top_modules": top_modules, "top_status_codes": top_status,
                                "avg_latency": avg_latency_py, "sample_logs": samples
                            }
                            clusters_summary_list.append(cluster_summary)

                            if errors > 0:
                                error_logs = cluster_logs[cluster_logs["level"] == "ERROR"]
                                error_modules_series = error_logs["module"].value_counts().head(3)
                                error_modules = {str(k): int(v) for k, v in error_modules_series.items()}

                                error_status_series = error_logs["status_code"][error_logs["status_code"] != 'N/A'].value_counts().head(3)
                                error_status = {str(k): int(v) for k, v in error_status_series.items()}

                                err_latency = pd.to_numeric(error_logs["latency"], errors='coerce').fillna(0)
                                positive_err_latency = err_latency[err_latency > 0]
                                avg_err_latency_py = float(round(positive_err_latency.mean(), 2)) if not positive_err_latency.empty else 0.0

                                sample_errors = error_logs["raw"].head(3).tolist()
                                error_profiles.append({
                                    "cluster_id": i, "error_count": errors, "error_rate": error_rate,
                                    "error_modules": error_modules, "error_status_codes": error_status,
                                    "avg_error_latency": avg_err_latency_py, "sample_errors": sample_errors
                                })

                        # Store results in session state
                        st.session_state['clusters_summary'] = clusters_summary_list
                        st.session_state['error_profiles'] = error_profiles
                        st.success(f"Clustering complete. Found {len(valid_clusters)} clusters.")
                        st.rerun() # Force rerun to show results

                    except ValueError as ve:
                        st.error(f"Clustering Error: {ve}")
                        st.session_state['clusters_summary'] = None
                        st.session_state['error_profiles'] = None
                    except Exception as e:
                        st.error(f"An unexpected error occurred during analysis: {e}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                        st.session_state['clusters_summary'] = None
                        st.session_state['error_profiles'] = None

            # --- Display Analysis Tabs (if clustering results exist) ---
            clusters_summary = st.session_state.get('clusters_summary')
            if clusters_summary is not None:
                analysis_tab_titles = ["Cluster Explorer"]
                error_profiles = st.session_state.get('error_profiles', [])
                if clusters_summary: analysis_tab_titles.append("Holistic Analysis")
                if error_profiles: analysis_tab_titles.append("Comparative Analysis")
                if 'datetime' in log_df.columns and log_df['datetime'].notna().any():
                    analysis_tab_titles.append("Temporal Analysis")

                analysis_tabs = st.tabs(analysis_tab_titles)
                analysis_tab_map = {title: tab for title, tab in zip(analysis_tab_titles, analysis_tabs)}

                # --- Cluster Explorer Tab ---
                with analysis_tab_map["Cluster Explorer"]:
                    if isinstance(clusters_summary, list) and clusters_summary:
                        valid_clusters = sorted([cs['cluster_id'] for cs in clusters_summary])
                        if not valid_clusters:
                            st.info("No valid clusters found in the summary data.")
                        else:
                            selected_cluster_id = st.selectbox(
                                "Select cluster to explore", options=valid_clusters,
                                format_func=lambda x: f"Cluster {x} ({next((cs['total_logs'] for cs in clusters_summary if cs['cluster_id'] == x), 0)} logs)",
                                key="cluster_selector"
                            )
                            current_cluster_summary = next((cs for cs in clusters_summary if cs['cluster_id'] == selected_cluster_id), None)

                            if current_cluster_summary:
                                exp_col1, exp_col2 = st.columns([3, 2])
                                with exp_col1:
                                    st.markdown(f"##### Sample Logs from Cluster {selected_cluster_id}")
                                    st.caption("Showing representative log entries.")
                                    for sample in current_cluster_summary.get('sample_logs', []):
                                        st.code(sample, language='log')

                                with exp_col2:
                                    st.markdown(f"##### Cluster {selected_cluster_id} Stats")
                                    st.metric("Total Logs", current_cluster_summary['total_logs'])
                                    error_rate = current_cluster_summary.get('error_rate', 0)
                                    delta_color = "inverse" if error_rate > 10 else ("off" if error_rate == 0 else "normal")
                                    st.metric("Error Rate", f"{error_rate}%", delta=f"{current_cluster_summary.get('error_count', 0)} errors", delta_color=delta_color)
                                    st.metric("Warning Rate", f"{current_cluster_summary.get('warning_rate', 0)}%", delta=f"{current_cluster_summary.get('warning_count', 0)} warnings", delta_color="off")
                                    st.metric("Avg. Latency (ms)", f"{current_cluster_summary.get('avg_latency', 0):.1f}")
                                    st.markdown("**Top Modules:**")
                                    st.json(current_cluster_summary.get('top_modules', {}), expanded=False)
                                    st.markdown("**Top Status Codes:**")
                                    st.json(current_cluster_summary.get('top_status_codes', {}), expanded=False)

                                    st.markdown("---")
                                    st.markdown("##### 🧠 AI Analysis of Cluster Summary")
                                    analyze_cluster_button = st.button(f"Analyze Summary for Cluster {selected_cluster_id}", key=f"analyze_summary_{selected_cluster_id}", use_container_width=True)
                                    if analyze_cluster_button:
                                        summary_str = json.dumps(current_cluster_summary, indent=2)
                                        with st.spinner("Getting AI insights..."):
                                            cluster_analysis_text = analyze_cluster_summary(
                                                selected_cluster_id, summary_str,
                                                use_ollama=use_ollama, ollama_model=ollama_model, ollama_url=ollama_url,
                                                remote_model=llm_model, api_key=api_key
                                            )
                                        st.markdown(f"""<div class="card" style="border-left: 5px solid {COLORS['primary']};"><small>{cluster_analysis_text}</small></div>""", unsafe_allow_html=True)
                            else:
                                 st.warning(f"Could not load summary for Cluster {selected_cluster_id}.")
                    else:
                        st.info("Run clustering analysis to explore clusters.")

                # --- Holistic Analysis Tab ---
                if "Holistic Analysis" in analysis_tab_map and isinstance(clusters_summary, list) and clusters_summary:
                    with analysis_tab_map["Holistic Analysis"]:
                        st.markdown("##### System-Wide Analysis")
                        st.caption("High-level assessment based on overall stats and cluster summaries.")
                        holistic_col1, holistic_col2 = st.columns(2)
                        with holistic_col1:
                            cluster_dist_df = pd.DataFrame([{'Cluster': str(cs['cluster_id']), 'Count': cs['total_logs']} for cs in clusters_summary])
                            fig = px.bar(cluster_dist_df, x="Cluster", y="Count", title="Log Distribution by Cluster", template=chart_theme, color="Cluster", color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig.update_layout(xaxis={'type': 'category'}, height=300, margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        with holistic_col2:
                            error_df = pd.DataFrame([{'Cluster': str(cs['cluster_id']), 'Error Rate': cs['error_rate']} for cs in clusters_summary])
                            fig = px.bar(error_df, x="Cluster", y="Error Rate", title="Error Rate (%) by Cluster", template=chart_theme, color="Error Rate", color_continuous_scale=px.colors.sequential.Reds)
                            fig.update_layout(xaxis={'type': 'category'}, height=300, margin=dict(t=30, b=0, l=0, r=0))
                            st.plotly_chart(fig, use_container_width=True)

                        st.markdown("##### 🧠 AI Holistic Analysis")
                        holistic_button = st.button("🔍 Generate Holistic System Analysis", key="holistic_analysis", use_container_width=True)
                        if holistic_button:
                            log_summary_data = st.session_state.get('log_df_summary', {})
                            if log_summary_data and clusters_summary:
                                summary_hash = _get_summary_hash(log_summary_data)
                                clusters_str = json.dumps(clusters_summary, indent=2)
                                with st.spinner("Generating system-wide insights..."):
                                    holistic_analysis_text = perform_holistic_analysis(
                                        summary_hash, clusters_str, # Pass hash and string
                                        use_ollama=use_ollama, ollama_model=ollama_model, ollama_url=ollama_url,
                                        remote_model=llm_model, api_key=api_key
                                    )
                                st.markdown(f"""<div class="card" style="border-left: 5px solid {COLORS['primary']};"><small>{holistic_analysis_text}</small></div>""", unsafe_allow_html=True)
                            else: st.warning("Log summary data or cluster summaries missing for holistic analysis.")

                # --- Comparative Analysis Tab ---
                if "Comparative Analysis" in analysis_tab_map and error_profiles: # Check error_profiles directly
                     with analysis_tab_map["Comparative Analysis"]:
                        st.markdown("##### Comparative Error Analysis")
                        st.caption("Compares error characteristics across clusters containing errors.")
                        comp_err_df = pd.DataFrame([{'Cluster': str(ep['cluster_id']), 'Error Count': ep['error_count']} for ep in error_profiles])
                        fig = px.bar(comp_err_df, x="Cluster", y="Error Count", title="Error Count per Cluster (with Errors)", template=chart_theme, color="Error Count", color_continuous_scale=px.colors.sequential.Reds)
                        fig.update_layout(xaxis={'type': 'category'}, height=300, margin=dict(t=30, b=0, l=0, r=0))
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("###### Error Profile Summaries")
                        for profile in error_profiles:
                            with st.expander(f"Cluster {profile['cluster_id']} Error Profile ({profile['error_count']} errors)"):
                                st.json(profile, expanded=False)

                        st.markdown("##### 🧠 AI Comparative Analysis")
                        comp_button = st.button("🔍 Generate Comparative Error Analysis", key="comparative_analysis", use_container_width=True)
                        if comp_button:
                            profiles_str = json.dumps(error_profiles, indent=2)
                            with st.spinner("Comparing error patterns..."):
                                comp_analysis_text = perform_comparative_analysis(
                                    profiles_str, # Pass string
                                    use_ollama=use_ollama, ollama_model=ollama_model, ollama_url=ollama_url,
                                    remote_model=llm_model, api_key=api_key
                                )
                            st.markdown(f"""<div class="card" style="border-left: 5px solid {COLORS['primary']};"><small>{comp_analysis_text}</small></div>""", unsafe_allow_html=True)

                # --- Temporal Analysis Tab ---
                if "Temporal Analysis" in analysis_tab_map:
                    with analysis_tab_map["Temporal Analysis"]:
                        st.markdown("##### Temporal Pattern Analysis")
                        st.caption("Examines log distribution and error rates over time.")
                        time_df_temporal = log_df.dropna(subset=['datetime']).copy()
                        if not time_df_temporal.empty and pd.api.types.is_datetime64_any_dtype(time_df_temporal['datetime']):
                            time_df_temporal['hour'] = time_df_temporal['datetime'].dt.hour # Use hour of day (0-23)
                            if pd.api.types.is_numeric_dtype(time_df_temporal['hour']):
                                try:
                                    hourly_agg = time_df_temporal.groupby('hour').agg(
                                        total_logs=('raw', 'count'),
                                        error_count=('level', lambda x: (x == 'ERROR').sum()),
                                        avg_latency=('latency', lambda x: pd.to_numeric(x, errors='coerce').fillna(0).mean())
                                    ).reset_index()
                                    hourly_agg['error_rate'] = (hourly_agg['error_count'] / hourly_agg['total_logs'] * 100).fillna(0).round(1)

                                    if 'hour' in hourly_agg.columns:
                                        # Plot Volume & Error Rate by Hour
                                        fig_vol = px.bar(hourly_agg, x='hour', y='total_logs', title='Log Volume by Hour of Day', template=chart_theme)
                                        fig_vol.update_layout(xaxis = dict(tickmode = 'linear'), height=300, margin=dict(t=30, b=0, l=0, r=0)) # Ensure x-axis treats hours as linear
                                        st.plotly_chart(fig_vol, use_container_width=True)

                                        fig_err = px.line(hourly_agg, x='hour', y='error_rate', title='Error Rate (%) by Hour of Day', template=chart_theme, markers=True)
                                        fig_err.update_traces(line=dict(color=COLORS["error"], width=2))
                                        fig_err.update_layout(xaxis = dict(tickmode = 'linear'), height=300, margin=dict(t=30, b=0, l=0, r=0), yaxis_range=[0,100]) # Fix y-axis range
                                        st.plotly_chart(fig_err, use_container_width=True)

                                        # Temporal AI Analysis Button
                                        st.markdown("##### 🧠 AI Temporal Analysis")
                                        temporal_button = st.button("🔍 Generate Temporal Pattern Analysis", key="temporal_analysis", use_container_width=True)
                                        if temporal_button:
                                            hourly_stats_str = hourly_agg[['hour', 'total_logs', 'error_rate', 'avg_latency']].round(2).to_string(index=False)
                                            avg_error_rate = hourly_agg['error_rate'].mean()
                                            anomalous_hours = hourly_agg[hourly_agg['error_rate'] > max(avg_error_rate * 1.5, 5)]['hour'].tolist()
                                            anomalous_hours_str = f"Hours with notably high error rates (> {max(avg_error_rate * 1.5, 5):.1f}%): {anomalous_hours}" if anomalous_hours else "No hours with significantly high error rates identified."

                                            with st.spinner("Analyzing temporal patterns..."):
                                                temporal_analysis_text = perform_temporal_analysis(
                                                    hourly_stats_str, anomalous_hours_str, # Pass strings
                                                    use_ollama=use_ollama, ollama_model=ollama_model, ollama_url=ollama_url,
                                                    remote_model=llm_model, api_key=api_key
                                                )
                                            st.markdown(f"""<div class="card" style="border-left: 5px solid {COLORS['primary']};"><small>{temporal_analysis_text}</small></div>""", unsafe_allow_html=True)
                                    else: st.warning("Could not prepare hourly aggregated data for plotting.")
                                except Exception as e:
                                    st.error(f"Error during temporal aggregation: {e}")
                            else: st.warning("Could not extract valid hour data for temporal analysis.")
                        else: st.info("Insufficient valid timestamp data for temporal analysis.")

            # Message if clustering hasn't been run yet or not enough data
            elif not log_df.empty and not analysis_disabled:
                 st.info("Click '🔄 Run / Update Clustering & Analysis' button above to perform analysis on the current logs.")
            elif analysis_disabled:
                st.info(f"Load more data (at least {min_logs_for_cluster} logs needed) to enable clustering and AI analysis.")
            else: # If clusters_summary is None after attempting analysis (error occurred)
                st.warning("Analysis could not be completed. Check logs or try again.")


    # --- TAB 4: Advanced Visualizations ---
    if "📈 Advanced Visualizations" in tab_map:
        with tab_map["📈 Advanced Visualizations"]:
            st.markdown("### 📈 Advanced Log Data Visualizations")

            if log_df.empty:
                st.info("Load data to see advanced visualizations.")
            else:
                viz_options = []
                if "latency" in log_df.columns and pd.to_numeric(log_df["latency"], errors='coerce').gt(0).any():
                    viz_options.append("Latency Analysis")
                if "status_code" in log_df.columns and log_df["status_code"].astype(str).str.match(r'^\d{3}$').any():
                    viz_options.append("Status Code Analysis")
                if "module" in log_df.columns and log_df["module"].nunique() > 1:
                    viz_options.append("Module Analysis")
                if 'datetime' in log_df.columns and log_df['datetime'].notna().any():
                    viz_options.append("Time-based Analysis")

                if not viz_options:
                    st.warning("Insufficient or unsuitable data columns (latency > 0, 3-digit status_code, multiple modules, datetime) found for advanced visualizations.")
                else:
                    viz_type = st.selectbox("Select visualization type", viz_options, key="viz_type_selector")

                    # --- Latency Viz ---
                    if viz_type == "Latency Analysis":
                        st.markdown("#### Latency Distribution & Outliers")
                        latency_df_viz = log_df.copy()
                        latency_df_viz['latency'] = pd.to_numeric(latency_df_viz["latency"], errors='coerce')
                        latency_df_viz = latency_df_viz.dropna(subset=['latency'])
                        latency_df_viz = latency_df_viz[latency_df_viz['latency'] > 0]

                        if not latency_df_viz.empty:
                            viz_col1, viz_col2 = st.columns(2)
                            with viz_col1:
                                fig = px.histogram(latency_df_viz, x="latency", nbins=50, title="Latency Distribution (ms, Log Scale Y-axis)", template=chart_theme, log_y=True)
                                fig.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0), xaxis_title="Latency (ms)")
                                st.plotly_chart(fig, use_container_width=True)
                            with viz_col2:
                                if 'module' in latency_df_viz.columns and latency_df_viz['module'].nunique() > 0:
                                    top_modules = latency_df_viz['module'].value_counts().head(15).index.tolist()
                                    fig = px.box(latency_df_viz[latency_df_viz['module'].isin(top_modules)],
                                                 x="latency", y="module", color="level", points=False,
                                                 title="Latency by Module (Top 15, Log Scale X-axis)", template=chart_theme, log_x=True,
                                                 category_orders={"module": top_modules[::-1]},
                                                 color_discrete_map={"ERROR": COLORS["error"], "WARNING": COLORS["warning"], "INFO": COLORS["success"], "DEBUG": COLORS["secondary"], "PARSE_ERROR": "#FFA500"})
                                    fig.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0), xaxis_title="Latency (ms, Log Scale)")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.caption("Module information missing for Latency by Module chart.")
                        else: st.info("No valid latency data (> 0 ms) found.")

                    # --- Status Code Viz ---
                    elif viz_type == "Status Code Analysis":
                        st.markdown("#### HTTP Status Code Analysis")
                        status_df_viz = log_df.copy()
                        status_df_viz["status_code_str"] = status_df_viz["status_code"].astype(str)
                        status_df_viz = status_df_viz[status_df_viz["status_code_str"].str.match(r'^\d{3}$')].copy()

                        if not status_df_viz.empty:
                            status_df_viz["status_code"] = status_df_viz["status_code_str"].astype(int)
                            status_df_viz['status_category'] = status_df_viz['status_code'].apply(
                                lambda x: 'Success (2xx)' if 200 <= x < 300 else
                                          'Redirect (3xx)' if 300 <= x < 400 else
                                          'Client Error (4xx)' if 400 <= x < 500 else
                                          'Server Error (5xx)' if 500 <= x < 600 else 'Other')

                            viz_col1, viz_col2 = st.columns(2)
                            with viz_col1:
                                status_counts = status_df_viz["status_code_str"].value_counts().reset_index().head(20)
                                status_counts.columns = ["Status Code", "Count"]
                                fig = px.bar(status_counts, x="Status Code", y="Count", title="Top 20 Status Codes", template=chart_theme)
                                fig.update_layout(xaxis={'type': 'category'}, height=350, margin=dict(t=30, b=0, l=0, r=0))
                                st.plotly_chart(fig, use_container_width=True)
                            with viz_col2:
                                category_counts = status_df_viz["status_category"].value_counts().reset_index()
                                category_counts.columns = ["Category", "Count"]
                                color_map = {"Success (2xx)": COLORS["success"], "Redirect (3xx)": COLORS["info"], "Client Error (4xx)": COLORS["warning"], "Server Error (5xx)": COLORS["error"], "Other": COLORS["secondary"]}
                                fig = px.pie(category_counts, values="Count", names="Category", title="Status Code Categories", color="Category", color_discrete_map=color_map, template=chart_theme)
                                fig.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig, use_container_width=True)

                            if 'module' in status_df_viz.columns and status_df_viz['module'].nunique() > 0:
                                st.markdown("###### Status Code Categories per Module")
                                top_modules = status_df_viz['module'].value_counts().head(15).index.tolist()
                                status_by_module = pd.crosstab(status_df_viz[status_df_viz['module'].isin(top_modules)]["module"], status_df_viz["status_category"])
                                cat_order = ['Success (2xx)', 'Redirect (3xx)', 'Client Error (4xx)', 'Server Error (5xx)', 'Other']
                                status_by_module = status_by_module.reindex(columns=cat_order, fill_value=0)

                                if not status_by_module.empty:
                                    fig = px.imshow(status_by_module, title="Status Code Categories by Module (Top 15)",
                                                    color_continuous_scale=px.colors.sequential.Blues, template=chart_theme, text_auto=True, aspect="auto")
                                    fig.update_layout(height=max(400, len(top_modules)*25))
                                    st.plotly_chart(fig, use_container_width=True)
                                else: st.info("Not enough data for Status vs Module heatmap.")
                            else: st.caption("Module information missing for Status Code vs Module heatmap.")
                        else: st.info("No valid 3-digit status codes found.")

                    # --- Module Viz ---
                    elif viz_type == "Module Analysis":
                         st.markdown("#### Module Log Activity")
                         if 'module' in log_df.columns and not log_df['module'].empty:
                             module_counts_viz = log_df['module'].value_counts().reset_index()
                             module_counts_viz.columns = ['Module', 'Count']

                             viz_col1, viz_col2 = st.columns(2)
                             with viz_col1:
                                 fig = px.bar(module_counts_viz.head(15), x='Count', y='Module', orientation='h', title="Top 15 Modules by Log Volume", template=chart_theme)
                                 fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(t=30, b=0, l=0, r=0))
                                 st.plotly_chart(fig, use_container_width=True)
                             with viz_col2:
                                if 'level' in log_df.columns and not log_df['level'].empty:
                                    df_for_treemap = log_df.dropna(subset=['module', 'level']).copy()
                                    df_for_treemap['module'] = df_for_treemap['module'].astype(str)
                                    df_for_treemap['level'] = df_for_treemap['level'].astype(str)

                                    if not df_for_treemap.empty:
                                        try:
                                            treemap_data = df_for_treemap.groupby(['module', 'level']).size().reset_index(name='count')
                                            fig = px.treemap(treemap_data,
                                                             path=[px.Constant("All Modules"), 'module', 'level'],
                                                             values='count',
                                                             title='Module Activity by Log Level (Count)',
                                                             template=chart_theme, height=400, color='level',
                                                             color_discrete_map={
                                                                 "ERROR": COLORS["error"], "WARNING": COLORS["warning"],
                                                                 "INFO": COLORS["success"], "DEBUG": COLORS["secondary"],
                                                                 "PARSE_ERROR": "#FFA500", "(?)": COLORS['secondary']
                                                             })
                                            fig.update_layout(margin=dict(t=50, b=0, l=0, r=0))
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Failed to generate Treemap: {e}")
                                    else:
                                        st.info("No valid data (module, level) found for Treemap after filtering.")
                                else:
                                    st.info("Missing 'level' column required for Treemap.")
                         else:
                             st.info("Missing 'module' column required for Module Analysis.")


                    # --- Time Viz ---
                    elif viz_type == "Time-based Analysis":
                        st.markdown("#### Log Activity Over Time")
                        time_df_viz = log_df.dropna(subset=['datetime']).copy()
                        if not time_df_viz.empty and pd.api.types.is_datetime64_any_dtype(time_df_viz['datetime']):
                            time_df_viz = time_df_viz.sort_values('datetime')
                            time_span_hours = (time_df_viz['datetime'].max() - time_df_viz['datetime'].min()).total_seconds() / 3600

                            resample_options = {'Auto': None}
                            if time_span_hours > 0.5: resample_options['5 Minutes'] = '5min'
                            if time_span_hours > 2: resample_options['15 Minutes'] = '15min'
                            if time_span_hours > 6: resample_options['Hour'] = 'h'
                            if time_span_hours > 48: resample_options['Day'] = 'D'

                            selected_freq_label = st.selectbox("Time Aggregation", list(resample_options.keys()), index=0) # Default to Auto

                            resample_freq_code = resample_options[selected_freq_label]
                            if resample_freq_code is None:
                                if time_span_hours <= 2: resample_freq_code = '1min'
                                elif time_span_hours <= 12: resample_freq_code = '5min'
                                elif time_span_hours <= 72: resample_freq_code = 'h'
                                else: resample_freq_code = 'D'
                                st.caption(f"Auto-selected aggregation: {resample_freq_code}")

                            try:
                                time_agg = time_df_viz.set_index('datetime').resample(resample_freq_code).agg(
                                    total_logs=('raw', 'count'),
                                    error_count=('level', lambda x: (x == 'ERROR').sum())
                                ).reset_index()
                                time_agg['error_rate'] = (time_agg['error_count'] / time_agg['total_logs'] * 100).fillna(0).round(1)

                                if not time_agg.empty:
                                    fig_time = go.Figure()
                                    fig_time.add_trace(go.Bar(x=time_agg['datetime'], y=time_agg['total_logs'], name='Total Logs', marker_color=COLORS['primary'], opacity=0.6))
                                    fig_time.add_trace(go.Scatter(x=time_agg['datetime'], y=time_agg['error_rate'], name='Error Rate (%)', yaxis='y2', mode='lines+markers', line=dict(color=COLORS['error'])))
                                    fig_time.update_layout(
                                        title=f'Log Volume & Error Rate (Aggregated by {resample_freq_code})', template=chart_theme, height=400,
                                        yaxis=dict(title='Total Logs'),
                                        yaxis2=dict(title='Error Rate (%)', overlaying='y', side='right', range=[0, 100]),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                        margin=dict(t=40, b=0, l=0, r=0)
                                    )
                                    st.plotly_chart(fig_time, use_container_width=True)
                                else: st.info("No data after time aggregation.")
                            except Exception as e:
                                st.error(f"Error during time aggregation/plotting: {e}")

                            # Daily patterns if data spans multiple days
                            if time_df_viz['datetime'].dt.date.nunique() > 1:
                                 st.markdown("###### Activity by Day of Week")
                                 time_df_viz['day_of_week'] = time_df_viz['datetime'].dt.day_name()
                                 day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                                 daily_counts = pd.pivot_table(time_df_viz, index='day_of_week', columns='level', aggfunc='size', fill_value=0).reindex(day_order, fill_value=0)

                                 if not daily_counts.empty:
                                     fig_daily = go.Figure()
                                     level_order = ["INFO", "DEBUG", "WARNING", "ERROR", "PARSE_ERROR"]
                                     level_colors = {"INFO": COLORS["success"], "DEBUG": COLORS["secondary"], "WARNING": COLORS["warning"], "ERROR": COLORS["error"], "PARSE_ERROR": "#FFA500"}
                                     for level in level_order:
                                        if level in daily_counts.columns:
                                            fig_daily.add_trace(go.Bar(x=daily_counts.index, y=daily_counts[level], name=level, marker_color=level_colors.get(level)))
                                     fig_daily.update_layout(barmode='stack', title="Log Volume by Day of Week", template=chart_theme, height=350, margin=dict(t=30, b=0, l=0, r=0))
                                     st.plotly_chart(fig_daily, use_container_width=True)
                                 else: st.info("No data to show activity by day of week.")
                        else: st.info("No valid timestamp data found for time analysis.")


# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 20px; padding-top: 15px; border-top: 1px solid #e6e6e6;">
    <p style="color: #6c757d; font-size: 0.9em;">AI-Powered Log Analyzer v1.4 (Dashboard Update)</p>
</div>
""", unsafe_allow_html=True)


# --- Final Rerun Logic for Stream ---
if should_rerun:
     time.sleep(0.05) # Short delay allows state updates to settle
     st.rerun()
# --- END OF FILE app.py ---
