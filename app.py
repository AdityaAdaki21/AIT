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
from datetime import datetime
import random
# Ensure logs_backend is in the same directory or accessible via PYTHONPATH
try:
    from logs_backend import fetch_logs
except ImportError:
    st.error("Could not import logs_backend.py. Make sure it's in the same directory.")
    st.stop() # Stop execution if backend cannot be imported

import os
# Set page configuration
st.set_page_config(
    page_title="AI-Powered Log Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme
COLORS = {
    "primary": "#4169E1",  # Royal Blue
    "secondary": "#6C757D",
    "success": "#28A745",
    "error": "#DC3545",
    "warning": "#FFC107",
    "info": "#17A2B8",
    "background": "#F8F9FA",
    "card": "#FFFFFF",
    "text": "#212529"
}

# Custom CSS for better styling
st.markdown(f"""
<style>
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    h1, h2, h3 {{
        color: {COLORS["primary"]};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 20px;
        border-radius: 4px 4px 0px 0px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS["primary"]};
        color: white;
    }}
    .error-card {{
        color: {COLORS["error"]};
        border-left: 5px solid {COLORS["error"]};
        padding-left: 10px;
    }}
    .warning-card {{
        color: {COLORS["warning"]};
        border-left: 5px solid {COLORS["warning"]};
        padding-left: 10px;
    }}
    .info-card {{
        color: {COLORS["info"]};
        border-left: 5px solid {COLORS["info"]};
        padding-left: 10px;
    }}
    /* Style for log entries in explorer */
    .log-entry-card {{
        margin-bottom: 10px;
        background-color: {COLORS['background']};
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9em;
        display: flex; /* Use flexbox */
        justify-content: space-between; /* Space between log and button */
        align-items: center; /* Vertically align items */
    }}
    .log-text {{
        flex-grow: 1; /* Allow text to take available space */
        margin-right: 15px; /* Add space before the button */
    }}
    .css-1544g2n.e1fqkh3o4 {{
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .card {{
        background-color: {COLORS["card"]};
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }}
    .metric-card {{
        background-color: {COLORS["card"]};
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }}
    .metric-value {{
        font-size: 24px;
        font-weight: bold;
        margin: 5px 0;
    }}
    .metric-label {{
        font-size: 14px;
        color: {COLORS["secondary"]};
    }}
</style>
""", unsafe_allow_html=True)

# App title with icon and gradient
st.markdown(f"""
<div style="background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['info']});
            padding: 20px; border-radius: 10px; margin-bottom: 25px;">
    <h1 style="color: white; margin: 0; display: flex; align-items: center;">
        <span style="font-size: 2.5rem; margin-right: 10px;">🧠</span>
        AI-Powered Log Analyzer
    </h1>
    <p style="color: white; opacity: 0.9; margin-top: 5px;">
        Intelligent log analysis with clustering, visualizations, and targeted LLM insights
    </p>
</div>
""", unsafe_allow_html=True)

# Constants - IMPORTANT: Replace with your actual keys or use environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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") # Use environment variable or placeholder
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.160.210:11434/api/generate") # Default Ollama API URL

# --- 1. Data Loading and Preparation ---
def is_csv_file(filename=None, content=None):
    if filename and filename.lower().endswith('.csv'):
        return True
    if content and len(content) > 0:
        # Simple check for CSV header-like structure
        header = content[0].lower()
        if header.count(',') > 2 and ('timestamp' in header or 'time' in header):
             return True
    return False

def preprocess(logs):
    """Basic preprocessing: lowercase and remove special chars except spaces, commas, colons, brackets"""
    return [re.sub(r'[^\w\s,:\[\]\-]', '', log.lower()) for log in logs]

@st.cache_data(show_spinner="Parsing logs...")
def extract_components(logs, filename=None):
    """Extract components from logs, detecting CSV or common formats"""
    is_csv = is_csv_file(filename=filename, content=logs)
    
    if is_csv:
        try:
            # Use StringIO to treat the list of strings as a file
            log_data = "\n".join(logs)
            df = pd.read_csv(io.StringIO(log_data))
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns] # Normalize columns

            # --- Try to map common CSV columns ---
            data = []
            for index, row in df.iterrows():
                entry = {
                    "timestamp": str(row.get('timestamp', row.get('time', ''))),
                    "level": str(row.get('level', row.get('log_level', 'INFO'))).upper(),
                    "module": str(row.get('module', row.get('service', row.get('api_id', 'unknown')))),
                    "status_code": str(row.get('status_code', row.get('status', 'N/A'))),
                    "latency": pd.to_numeric(row.get('latency_ms', row.get('latency', 0)), errors='coerce'),
                    "env": str(row.get('env', row.get('environment', 'unknown'))),
                    "message": str(row.get('message', '')),
                    "raw": ','.join(row.astype(str).values) # Reconstruct raw log
                }

                # If message is empty, construct one
                if not entry["message"]:
                     details = [f"{k}: {v}" for k, v in row.items() if k not in ['timestamp', 'level', 'module', 'status_code', 'latency', 'env', 'raw', 'message']]
                     entry["message"] = f"Status: {entry['status_code']}, Latency: {entry['latency']}ms, Details: {', '.join(details)}"

                # Infer level from status/error column if not present
                if 'level' not in df.columns and entry['level'] == 'INFO':
                     if pd.to_numeric(row.get('error', 0), errors='coerce') == 1:
                         entry['level'] = 'ERROR'
                     elif str(entry['status_code']).startswith(('4', '5')):
                         entry['level'] = 'WARNING' if str(entry['status_code']).startswith('4') else 'ERROR'

                data.append(entry)

            df_extracted = pd.DataFrame(data)
            df_extracted['latency'] = df_extracted['latency'].fillna(0.0).astype(float)
            return df_extracted[['timestamp', 'level', 'module', 'message', 'raw', 'status_code', 'latency', 'env']]

        except Exception as e:
            st.warning(f"CSV parsing failed: {e}. Falling back to line-by-line parsing.")
            # Fall through to generic parsing if CSV fails

    # --- Generic line-by-line parsing (Fallback or for non-CSV) ---
    data = []
    # Common timestamp patterns
    ts_patterns = [
        r'\[?(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]?', # ISOish
        r'\[?(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\]?', # Syslog style (e.g., Oct 9 13:54:00) - Year missing often
        r'\[?(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})\]?' # Apache common log format
    ]
    # More robust module extraction
    module_patterns = [
        r'\[\w+\]\s+\[?([\w\-]+)\]?', # [<level>] [<module>]
        r':\s+([\w\-]+):',           # : module:
        r'\[([\w\-]+)\]\s+\[\w+\]',   # [<module>] [<env>] (like original backend)
        r'\]\s+([\w\-]+)\s+\['        # ] module [
    ]

    for log in logs:
        timestamp, level, module, status_code, latency, env, message = "", "INFO", "unknown", "N/A", 0.0, "unknown", log

        # Extract Timestamp
        for pattern in ts_patterns:
            match = re.search(pattern, log)
            if match:
                timestamp = match.group(1)
                break # Use first found timestamp

        # Determine Log Level
        if re.search(r'\b(ERROR|ERR|CRITICAL|FATAL)\b', log, re.IGNORECASE): level = "ERROR"
        elif re.search(r'\b(WARN|WARNING)\b', log, re.IGNORECASE): level = "WARNING"
        elif re.search(r'\b(DEBUG)\b', log, re.IGNORECASE): level = "DEBUG"
        elif re.search(r'\b(INFO|NOTICE)\b', log, re.IGNORECASE): level = "INFO"

        # Extract Module
        for pattern in module_patterns:
             match = re.search(pattern, log)
             if match:
                 potential_module = match.group(1).strip()
                 # Avoid extracting log levels as modules
                 if potential_module.upper() not in ["INFO", "WARN", "WARNING", "ERROR", "DEBUG", "CRITICAL", "FATAL"]:
                     module = potential_module
                     break

        # Extract Status Code
        status_match = re.search(r'(?:status|code|status_code)\W+(\d{3})', log, re.IGNORECASE)
        if status_match: status_code = status_match.group(1)

        # Extract Latency
        latency_match = re.search(r'(\d+\.?\d*)\s*(?:ms|milliseconds)', log, re.IGNORECASE)
        if latency_match: latency = float(latency_match.group(1))
        else: # check for seconds
             latency_match_s = re.search(r'(\d+\.?\d*)\s*s(?:ec|econds)?', log, re.IGNORECASE)
             if latency_match_s: latency = float(latency_match_s.group(1)) * 1000 # convert s to ms

        # Extract Environment
        env_match = re.search(r'\[(production|prod|staging|stag|development|dev|test)\]', log, re.IGNORECASE)
        if env_match: env = env_match.group(1).lower()
        elif 'prod' in log.lower(): env = "production"
        elif 'dev' in log.lower(): env = "development"


        data.append({
            "timestamp": timestamp, "level": level, "module": module, "message": message,
            "raw": log, "status_code": status_code, "latency": latency, "env": env
        })

    df_extracted = pd.DataFrame(data)
    df_extracted['latency'] = df_extracted['latency'].fillna(0.0).astype(float)
    return df_extracted


def parse_timestamp(ts_str):
    if not isinstance(ts_str, str) or not ts_str.strip():
        return None
    # Try different timestamp formats
    formats = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%dT%H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%d/%b/%Y:%H:%M:%S", # Apache CLF needs timezone handling, omitted here for simplicity
        "%b %d %H:%M:%S",   # Syslog style (no year) - assumes current year
        # Add more formats as needed
    ]
    now = datetime.now()
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str.strip(), fmt)
            if fmt == "%b %d %H:%M:%S" and dt.year == 1900: # Handle missing year in syslog
                 dt = dt.replace(year=now.year)
                 # Handle year rollover if log date is in Dec but current date is Jan
                 if dt.month == 12 and now.month == 1:
                     dt = dt.replace(year=now.year - 1)
            return dt
        except ValueError:
            continue
    # Fallback: try pandas to_datetime (slower but flexible)
    try:
        return pd.to_datetime(ts_str.strip(), errors='coerce')
    except Exception:
        return None

# --- LLM Query Functions ---

def query_ollama(prompt, model_name="llama3.2:3b", api_url=OLLAMA_API_URL):
    """Query the local Ollama API."""
    if not api_url or not model_name:
        return "Error: Ollama API URL or model name not configured."
    try:
        response = requests.post(
            api_url,
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=60 # Add a timeout
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json().get("response", "Error: No response field in Ollama output.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Ollama ({api_url}): {e}")
        return f"Error connecting to Ollama API: {e}"
    except Exception as e:
        st.error(f"Error processing Ollama response: {e}")
        return f"Error processing Ollama response: {e}"


def query_remote_llm(prompt, model="mistralai/mistral-7b-instruct:free", api_key=OPENROUTER_API_KEY):
    """Query remote LLM API (OpenRouter)."""
    if not api_key or api_key == "sk-or-..." or not model: # Basic check
         return "Error: OpenRouter API Key or model not configured."
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60 # Add a timeout
        )
        response.raise_for_status() # Raise HTTPError for bad responses
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return f"Error: Unexpected response format from OpenRouter: {result}"
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying OpenRouter API ({model}): {e}")
        return f"Error querying remote LLM: {e}"
    except Exception as e:
        st.error(f"Error processing OpenRouter response: {e}")
        return f"Error processing remote LLM response: {e}"

# --- Specific LLM Analysis Tasks ---

@st.cache_data(show_spinner="🧠 Performing Holistic Analysis...")
def perform_holistic_analysis(_log_df_summary, clusters_summary, use_ollama=True, ollama_model="llama3.1:8b", ollama_url=None, remote_model=None, api_key=None):
    """Perform holistic analysis using summaries."""
    # Prepare cluster summaries for context (already done mostly by caller)
    # Create a prompt for holistic analysis
    system_prompt = """You are an expert log analysis AI assistant specializing in system diagnostics.
    Analyze the provided log cluster summaries and overall log statistics. Provide a concise assessment of the system's health."""

    prompt = f"""
# System Log Analysis Report

## Overall Statistics
Total Logs: {_log_df_summary['total_logs']}
Error Count: {_log_df_summary['error_count']} ({_log_df_summary['error_rate']}%)
Warning Count: {_log_df_summary['warning_count']} ({_log_df_summary['warning_rate']}%)
Unique Modules: {_log_df_summary['unique_modules']}
Avg. Latency: {_log_df_summary['avg_latency']:.2f}ms

## Cluster Summaries
{clusters_summary}

## Analysis Task
Based *only* on the statistics and cluster summaries provided above:
1.  Provide a brief (2-3 sentences) assessment of the overall system health.
2.  Identify the top 1-2 most concerning clusters or patterns observed (e.g., high error rate, specific module issues).
3.  Suggest 1-2 high-level areas to investigate further based on this summary data.
Keep the response concise and focused on the provided data.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

@st.cache_data(show_spinner="🧠 Performing Comparative Analysis...")
def perform_comparative_analysis(error_profiles, use_ollama=True, ollama_model="llama3.1:8b", ollama_url=None, remote_model=None, api_key=None):
    """Compare error patterns between clusters using summaries."""
    prompt = f"""
# Comparative Error Analysis

Compare the error patterns across these log clusters based *only* on the provided error profiles:

## Error Profiles by Cluster
{error_profiles}

## Analysis Task
1.  Which cluster appears to have the most critical or impactful errors based on the summaries (count, modules, status codes)? Explain briefly why.
2.  Are there any notable similarities or differences in the types of errors (modules, status codes) between clusters?
3.  Based *only* on this information, can you hypothesize if errors in one cluster might be related to another? (e.g., 'Errors in Cluster X (database) might be causing issues seen in Cluster Y (api-gateway)'). Be speculative.
Keep the response concise and focused on comparing the provided profiles.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

@st.cache_data(show_spinner="🧠 Performing Temporal Analysis...")
def perform_temporal_analysis(hourly_stats, anomalous_hours, use_ollama=True, ollama_model="llama3.1:8b", ollama_url=None, remote_model=None, api_key=None):
    """Analyze temporal patterns using summaries."""
    prompt = f"""
# Temporal Log Analysis

Analyze the temporal patterns based *only* on the provided hourly statistics and identified anomalous hours:

## Hourly Statistics Summary
(Describes general trends across hours - e.g., highest error counts between 2-4 PM, high latency during morning peak)
{hourly_stats}

## Anomalous Hours (High Error Rate)
{anomalous_hours}

## Analysis Task
1.  Describe the main time-based pattern observed (e.g., errors peak during specific hours, constant load).
2.  Highlight the significance of the anomalous hours identified. What might be happening during these times?
3.  Suggest 1-2 potential reasons for these temporal patterns (e.g., business hours, batch jobs, external factors).
Keep the response concise and focused on interpreting the time-based data provided.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

@st.cache_data(show_spinner="🧠 Analyzing Cluster Summary...")
def analyze_cluster_summary(cluster_id, cluster_summary, use_ollama=True, ollama_model="llama3.1:8b", ollama_url=None, remote_model=None, api_key=None):
    """Analyze a *summary* of a single cluster, not raw logs."""
    prompt = f"""
# Log Cluster Analysis (Cluster {cluster_id})

Analyze the following summary for Log Cluster {cluster_id}:

## Cluster Summary
Total Logs: {cluster_summary.get('total_logs', 'N/A')}
Error Count: {cluster_summary.get('error_count', 'N/A')} ({cluster_summary.get('error_rate', 'N/A')}%)
Warning Count: {cluster_summary.get('warning_count', 'N/A')} ({cluster_summary.get('warning_rate', 'N/A')}%)
Top Modules: {cluster_summary.get('top_modules', 'N/A')}
Top Status Codes: {cluster_summary.get('top_status_codes', 'N/A')}
Average Latency: {cluster_summary.get('avg_latency', 'N/A'):.2f}ms
Sample Log Messages (if available):
{chr(10).join(cluster_summary.get('sample_logs', ['N/A']))}

## Analysis Task
Based *only* on the summary provided above for Cluster {cluster_id}:
1.  What is the primary characteristic or theme of this cluster? (e.g., dominated by errors from 'database', mostly high latency warnings for 'api-gateway', successful 'auth' operations).
2.  What are the 1-2 most likely issues or points of interest indicated by this summary?
3.  Suggest 1-2 specific next steps for investigating this cluster further.
Keep the response concise and directly related to the provided summary data.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

@st.cache_data(show_spinner="🧠 Explaining Log Entry...")
def explain_single_log(log_entry, use_ollama=True, ollama_model="llama3.1:8b", ollama_url=None, remote_model=None, api_key=None):
    """Explain a single log entry using the LLM."""
    prompt = f"""
# Explain Log Entry

Explain the following log entry in simple terms. Focus on what it likely means, potential causes, and possible impact.

Log Entry:
## Explanation Task:
1.  **Meaning:** What is this log message telling us?
2.  **Potential Cause(s):** What are common reasons for this type of message?
3.  **Potential Impact:** What could be the consequence of this event (e.g., user impact, system stability)?
4.  **Suggested Next Step:** What is one simple thing someone could check first to investigate?

Keep the explanation clear, concise, and actionable. Avoid jargon where possible.
"""
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)


# --- Utility functions for Ollama status ---
@st.cache_data(ttl=60) # Cache for 60 seconds
def is_ollama_available(api_url):
    """Check if Ollama service is available"""
    if not api_url: return False
    try:
        # Check the base URL, not the /api/generate endpoint
        base_url = api_url.split('/api/')[0]
        response = requests.get(f"{base_url}/api/tags", timeout=5) # Short timeout
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

@st.cache_data(ttl=300) # Cache for 5 minutes
def get_ollama_models(api_url):
    """Get list of available models from Ollama"""
    models = []
    if not api_url: return models
    try:
        base_url = api_url.split('/api/')[0]
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            models = sorted([model["name"] for model in models_data])
        return models
    except requests.exceptions.RequestException:
        return models # Return empty list on connection error

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Data Source")
    data_source = st.radio(
        "Choose data source",
        ["Sample Logs", "Upload Log File"],
        key="data_source_radio",
        horizontal=True
    )

    uploaded_file = None
    logs = []
    file_name = None

    if data_source == "Upload Log File":
        uploaded_file = st.file_uploader("📤 Upload log file (.log, .txt, .csv)", type=["log", "txt", "csv"])
        if uploaded_file:
            try:
                content_bytes = uploaded_file.read()
                # Attempt to decode with common encodings
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        content = content_bytes.decode("latin-1") # Common fallback
                    except UnicodeDecodeError:
                         st.error("Could not decode file. Please ensure it's UTF-8 or Latin-1 encoded.")
                         content = "" # Prevent further processing

                if content:
                    logs = content.splitlines()
                    file_name = uploaded_file.name
                else:
                    st.warning("Uploaded file appears empty or could not be decoded.")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
        elif 'log_df' not in st.session_state: # Show info only if no data is loaded yet
             st.info("Upload a log file or switch to sample logs.")

    # Fetch sample logs only if upload wasn't successful and no data is loaded
    if not logs and data_source == "Sample Logs":
        with st.spinner("Loading sample logs..."):
            logs = fetch_logs()
            file_name = "api_logs.csv" # Assume sample is the CSV
        if not logs:
             st.error("Failed to load sample logs.")

    # Process logs only if we have some
    if logs:
        # Use filename hint for CSV detection
        log_df = extract_components(logs, filename=file_name)
        log_df['datetime'] = log_df['timestamp'].apply(parse_timestamp)
        st.session_state['log_df'] = log_df # Store in session state
        st.success(f"Loaded {len(log_df)} log entries.")
    elif 'log_df' in st.session_state:
        # Use previously loaded data if available (e.g., user switched source after upload)
        log_df = st.session_state['log_df']
        st.info("Using previously loaded data.")
    else:
        st.warning("No log data loaded. Please upload a file or select sample logs.")
        # Create an empty DataFrame to prevent errors down the line
        log_df = pd.DataFrame(columns=['timestamp', 'level', 'module', 'message', 'raw', 'status_code', 'latency', 'env', 'datetime'])
        st.session_state['log_df'] = log_df


    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")

    # Only allow clustering if data is loaded
    if not log_df.empty:
        max_clusters = min(8, len(log_df) // 10) # Heuristic: at least 10 logs per cluster
        max_clusters = max(2, max_clusters) # Ensure at least 2 clusters if possible
        default_clusters = min(4, max_clusters)

        n_clusters = st.slider(
            "Number of log clusters",
            min_value=2,
            max_value=max_clusters,
            value=default_clusters,
            help="Adjust the granularity of log grouping (requires > 10 logs)."
        )
        st.session_state['n_clusters'] = n_clusters
    else:
        st.markdown("_(Clustering disabled: Load data first)_")
        st.session_state['n_clusters'] = 4 # Default even if disabled

    # LLM settings
    st.markdown("### 🤖 LLM Settings")
    llm_provider = st.radio("Choose LLM Provider", ["Local Ollama", "Remote OpenRouter"],
                            index=0 if is_ollama_available(OLLAMA_API_URL) else 1, # Default to Ollama if available
                            horizontal=True)
    st.session_state['use_ollama'] = (llm_provider == "Local Ollama")

    if st.session_state['use_ollama']:
        st.session_state['ollama_url'] = st.text_input("Ollama API URL", value=OLLAMA_API_URL,
                                help="URL of your local Ollama /api/generate endpoint")
        ollama_is_ready = is_ollama_available(st.session_state['ollama_url'])
        if ollama_is_ready:
            st.success("✅ Connected to Ollama")
            available_models = get_ollama_models(st.session_state['ollama_url'])
            if available_models:
                # Try common defaults first
                preferred_models = ["llama3.1:8b", "llama3:8b", "mistral:7b", "phi3:mini"]
                default_model = next((m for m in preferred_models if m in available_models), available_models[0])

                st.session_state['ollama_model'] = st.selectbox(
                    "Select Ollama Model",
                    options=available_models,
                    index=available_models.index(default_model) if default_model in available_models else 0
                )
            else:
                st.warning("Ollama running, but no models found? Pull one (e.g., `ollama pull llama3.1:8b`)")
                st.session_state['ollama_model'] = st.text_input("Enter Model Name", value="llama3.1:8b")
        else:
            st.error(f"❌ Cannot connect to Ollama at {st.session_state['ollama_url']}. Is it running?")
            st.session_state['ollama_model'] = st.text_input("Model Name (if Ollama runs)", value="llama3.1:8b")
        # Clear remote settings
        st.session_state['llm_model'] = None
        st.session_state['api_key'] = None
    else: # Remote OpenRouter
        st.session_state['api_key'] = st.text_input("OpenRouter API Key", value=OPENROUTER_API_KEY, type="password",
                                     help="Get yours from OpenRouter.ai")
        st.session_state['llm_model'] = st.selectbox(
            "Remote LLM Model",
             # Add more models as needed, check OpenRouter for availability
            ["mistralai/mistral-7b-instruct:free", "meta-llama/llama-3.1-8b-instruct:free", "anthropic/claude-3.5-sonnet", "google/gemini-flash-1.5"],
            help="Select the remote AI model (check costs/limits on OpenRouter)",
            index=0
        )
         # Clear Ollama settings
        st.session_state['ollama_url'] = None
        st.session_state['ollama_model'] = None


    st.markdown("---")
    st.markdown("### 🎨 Visualization")
    chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"], index=1)
    st.session_state['chart_theme'] = chart_theme

    # Add about section
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    Analyze logs using clustering and targeted AI insights.
    - **Dashboard:** Overview metrics & charts.
    - **Log Explorer:** Filter, search, and explain individual logs.
    - **AI Analysis:** Cluster logs and get AI summaries.
    - **Advanced Viz:** Deeper dives into latency, status codes, etc.
    """)


# --- Main App Area ---

# Retrieve from session state if possible, otherwise use defaults/empty
log_df = st.session_state.get('log_df', pd.DataFrame(columns=['timestamp', 'level', 'module', 'message', 'raw', 'status_code', 'latency', 'env', 'datetime']))
n_clusters = st.session_state.get('n_clusters', 4)
use_ollama = st.session_state.get('use_ollama', False)
ollama_model = st.session_state.get('ollama_model', 'llama3.1:8b')
ollama_url = st.session_state.get('ollama_url', OLLAMA_API_URL)
llm_model = st.session_state.get('llm_model', 'mistralai/mistral-7b-instruct:free')
api_key = st.session_state.get('api_key', OPENROUTER_API_KEY)
chart_theme = st.session_state.get('chart_theme', 'plotly_white')


# Check if data is loaded before creating tabs
if log_df.empty:
    st.warning("⬅️ Please load log data using the sidebar.")
else:
    # --- Create Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔍 Log Explorer",
        "🧠 AI Analysis",
        "📈 Advanced Visualizations"
    ])

    # --- TAB 1: Dashboard ---
    with tab1:
        st.markdown("### 📈 Key Metrics")
        total_logs = len(log_df)
        error_count = len(log_df[log_df["level"] == "ERROR"])
        warning_count = len(log_df[log_df["level"] == "WARNING"])
        modules_count = log_df["module"].nunique()
        # Calculate latency only if numeric and > 0
        valid_latency = pd.to_numeric(log_df["latency"], errors='coerce').fillna(0)
        avg_latency = valid_latency[valid_latency > 0].mean() if (valid_latency > 0).any() else 0.0

        # Store summary for potential use in holistic analysis
        st.session_state['log_df_summary'] = {
            'total_logs': total_logs,
            'error_count': error_count,
            'warning_count': warning_count,
            'error_rate': round(error_count / total_logs * 100, 1) if total_logs else 0,
            'warning_rate': round(warning_count / total_logs * 100, 1) if total_logs else 0,
            'unique_modules': modules_count,
            'avg_latency': avg_latency
        }

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['error']}">
                <div class="metric-label">Errors</div>
                <div class="metric-value" style="color: {COLORS['error']}">{error_count}</div>
                <div class="metric-label">{st.session_state['log_df_summary']['error_rate']}% of logs</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
             st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['warning']}">
                <div class="metric-label">Warnings</div>
                <div class="metric-value" style="color: {COLORS['warning']}">{warning_count}</div>
                <div class="metric-label">{st.session_state['log_df_summary']['warning_rate']}% of logs</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
             st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['primary']}">
                <div class="metric-label">Unique Modules</div>
                <div class="metric-value" style="color: {COLORS['primary']}">{modules_count}</div>
                <div class="metric-label">Involved</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
             st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {COLORS['info']}">
                <div class="metric-label">Avg Latency</div>
                <div class="metric-value" style="color: {COLORS['info']}">{avg_latency:.1f} ms</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("### Log Level Distribution")
            level_counts = log_df["level"].value_counts().reset_index()
            level_counts.columns = ["Level", "Count"]
            color_map = {"ERROR": COLORS["error"], "WARNING": COLORS["warning"], "INFO": COLORS["success"], "DEBUG": COLORS["secondary"]}
            fig = px.pie(level_counts, values='Count', names='Level', color='Level',
                         color_discrete_map=color_map, hole=0.4, template=chart_theme)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=30, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            st.markdown("### Top Modules by Error Count")
            module_errors = log_df[log_df['level']=='ERROR']['module'].value_counts().head(10).reset_index()
            module_errors.columns = ['Module', 'Error Count']
            if not module_errors.empty:
                fig = px.bar(module_errors, x='Error Count', y='Module', orientation='h',
                            color='Error Count', color_continuous_scale=px.colors.sequential.Reds,
                            template=chart_theme)
                fig.update_layout(margin=dict(t=30, b=20, l=20, r=20), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No errors found to display module breakdown.")

        # Error timeline if timestamps are available
        if log_df['datetime'].notna().any():
            st.markdown("### Log Timeline (by Hour)")
            time_df = log_df[log_df['datetime'].notna()].copy()
            time_df['hour'] = time_df['datetime'].dt.floor('h') # Group by hour
            hourly_counts = time_df.groupby(['hour', 'level']).size().unstack(fill_value=0)

            fig = go.Figure()
            for level, color in [("ERROR", COLORS["error"]), ("WARNING", COLORS["warning"]), ("INFO", COLORS["success"])]:
                if level in hourly_counts.columns:
                    fig.add_trace(go.Bar(x=hourly_counts.index, y=hourly_counts[level], name=level, marker_color=color))

            fig.update_layout(barmode='stack', template=chart_theme,
                              margin=dict(t=30, b=20, l=20, r=20),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              xaxis_title='Time', yaxis_title='Log Count')
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("No valid timestamp data found for timeline visualization.")

    # --- TAB 2: Log Explorer ---
    with tab2:
        st.markdown("### 🔍 Filter & Explore Logs")
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

        with filter_col1:
            levels = ["All"] + sorted(log_df["level"].unique().tolist())
            selected_level = st.selectbox("Level", levels, key="level_filter")
        with filter_col2:
            modules = ["All"] + sorted(log_df["module"].unique().tolist())
            selected_module = st.selectbox("Module", modules, key="module_filter")
        with filter_col3:
            keyword = st.text_input("Search Keyword", key="keyword_filter", placeholder="Search in message content...")

        # Advanced Filters Expander
        with st.expander("Advanced Filters"):
            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                status_codes = ["All"] + sorted([str(code) for code in log_df["status_code"].unique() if str(code).strip() and str(code) != 'N/A'])
                selected_status = st.selectbox("Status Code", status_codes, key="status_filter")
            with adv_col2:
                envs = ["All"] + sorted([env for env in log_df["env"].unique() if str(env).strip() and str(env) != 'unknown'])
                selected_env = st.selectbox("Environment", envs, key="env_filter")


        # Apply filters
        filtered_df = log_df.copy()
        if selected_level != "All": filtered_df = filtered_df[filtered_df["level"] == selected_level]
        if selected_module != "All": filtered_df = filtered_df[filtered_df["module"] == selected_module]
        if selected_status != "All": filtered_df = filtered_df[filtered_df["status_code"] == selected_status]
        if selected_env != "All": filtered_df = filtered_df[filtered_df["env"] == selected_env]
        if keyword: filtered_df = filtered_df[filtered_df["raw"].str.contains(keyword, case=False, na=False)] # Search in raw log

        st.markdown(f"### 📝 Log Entries ({len(filtered_df)} found)")

        # Pagination
        page_size = 25
        total_pages = (len(filtered_df) // page_size) + (1 if len(filtered_df) % page_size > 0 else 0)
        page_number = st.number_input('Page', min_value=1, max_value=max(1, total_pages), step=1, value=1, key="log_page")

        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        paginated_df = filtered_df.iloc[start_idx:end_idx]

        # Display logs with "Explain" button
        if not paginated_df.empty:
            for i, (_, row) in enumerate(paginated_df.iterrows()):
                level_class = "info-card"
                level_color = COLORS['info']
                if row["level"] == "ERROR":
                    level_class = "error-card"
                    level_color = COLORS['error']
                elif row["level"] == "WARNING":
                    level_class = "warning-card"
                    level_color = COLORS['warning']

                # Use index from the original filtered_df for a stable key
                original_index = filtered_df.index[start_idx + i]
                button_key = f"explain_{original_index}"

                with st.container():
                    st.markdown(f"""
                    <div class="log-entry-card {level_class}">
                        <div class="log-text">
                           <span style="color: {COLORS['secondary']}; font-weight: normal;">[{row['timestamp']}]</span>
                           <span style="font-weight: bold; color:{level_color};"> [{row['level']}]</span>
                           <span style="font-weight: bold; color:{COLORS['primary']};"> [{row['module']}]</span>
                           <span> {row['message']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Place button separately to control layout better if needed
                    explain_button_clicked = st.button("Explain with AI", key=button_key, help="Get an AI explanation for this log entry")

                    if explain_button_clicked:
                        explanation = explain_single_log(
                            row['raw'],
                            use_ollama=use_ollama,
                            ollama_model=ollama_model, ollama_url=ollama_url,
                            remote_model=llm_model, api_key=api_key
                        )
                        st.info(f"**AI Explanation for Log:**\n{explanation}")
        else:
            st.info("No logs match your filter criteria.")


    # --- TAB 3: AI Analysis ---
    with tab3:
        st.markdown("### 🧠 AI-Powered Log Analysis")

        if len(log_df) < n_clusters or len(log_df) < 10:
            st.warning(f"Not enough log entries ({len(log_df)}) for meaningful clustering with {n_clusters} clusters. Need at least 10.")
        else:
            with st.spinner("Clustering logs... This may take a moment."):
                try:
                    # Use 'raw' logs for TF-IDF as they contain more context
                    clean_logs = preprocess(log_df['raw'].tolist())
                    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
                    X = vectorizer.fit_transform(clean_logs)

                    # Check if vocabulary is empty (can happen with very uniform logs)
                    if X.shape[1] == 0:
                         st.error("Could not generate features for clustering. Logs might be too uniform or contain only stop words.")
                         # Assign all logs to cluster 0 as a fallback
                         log_df["cluster"] = 0
                         n_clusters = 1 # Override n_clusters
                    else:
                        # Ensure n_clusters is not more than samples
                        actual_n_clusters = min(n_clusters, X.shape[0])
                        if actual_n_clusters < n_clusters:
                            st.warning(f"Reduced number of clusters to {actual_n_clusters} due to limited unique log patterns.")
                            n_clusters = actual_n_clusters

                        if n_clusters >= 2:
                             # Try KMeans with different initializations
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5) # Try 5 different centroids seeds
                            labels = kmeans.fit_predict(X)
                            log_df["cluster"] = labels
                            st.session_state['log_df'] = log_df # Update session state with clusters
                        else:
                             st.warning("Clustering requires at least 2 clusters. Skipping.")
                             log_df["cluster"] = 0 # Assign default cluster


                except Exception as e:
                    st.error(f"Error during clustering: {e}")
                    log_df["cluster"] = -1 # Indicate clustering failed

            # Proceed only if clustering was successful (or handled)
            if "cluster" in log_df.columns and log_df["cluster"].max() >= 0:
                # Prepare summaries for holistic/comparative analysis
                clusters_summary_list = []
                error_profiles = []
                valid_clusters = sorted(log_df['cluster'].unique())
                actual_n_clusters = len(valid_clusters)

                for i in valid_clusters:
                    cluster_logs = log_df[log_df["cluster"] == i]
                    total = len(cluster_logs)
                    errors = len(cluster_logs[cluster_logs["level"] == "ERROR"])
                    warnings = len(cluster_logs[cluster_logs["level"] == "WARNING"])
                    error_rate = round(errors / total * 100, 1) if total else 0
                    warning_rate = round(warnings / total * 100, 1) if total else 0
                    top_modules = cluster_logs["module"].value_counts().head(3).to_dict()
                    top_status = cluster_logs["status_code"][cluster_logs["status_code"] != 'N/A'].value_counts().head(3).to_dict()
                    avg_latency = pd.to_numeric(cluster_logs["latency"], errors='coerce').fillna(0)
                    avg_latency = avg_latency[avg_latency > 0].mean() if (avg_latency > 0).any() else 0.0
                    # Get a few *representative* samples (e.g., errors first)
                    sample_logs_errors = cluster_logs[cluster_logs["level"] == "ERROR"]["raw"].head(3).tolist()
                    sample_logs_warnings = cluster_logs[cluster_logs["level"] == "WARNING"]["raw"].head(2).tolist()
                    sample_logs_info = cluster_logs[cluster_logs["level"] == "INFO"]["raw"].head(2).tolist()
                    sample_logs = sample_logs_errors + sample_logs_warnings + sample_logs_info
                    sample_logs = sample_logs[:5] # Limit total samples


                    cluster_summary = {
                        "cluster_id": i, "total_logs": total, "error_count": errors, "warning_count": warnings,
                        "error_rate": error_rate, "warning_rate": warning_rate,
                        "top_modules": top_modules, "top_status_codes": top_status,
                        "avg_latency": avg_latency, "sample_logs": sample_logs
                    }
                    clusters_summary_list.append(cluster_summary)

                    # Create error profile if errors exist
                    if errors > 0:
                        error_logs = cluster_logs[cluster_logs["level"] == "ERROR"]
                        error_modules = error_logs["module"].value_counts().head(3).to_dict()
                        error_status = error_logs["status_code"][error_logs["status_code"] != 'N/A'].value_counts().head(3).to_dict()
                        avg_err_latency = pd.to_numeric(error_logs["latency"], errors='coerce').fillna(0)
                        avg_err_latency = avg_err_latency[avg_err_latency > 0].mean() if (avg_err_latency > 0).any() else 0.0
                        sample_errors = error_logs["raw"].head(3).tolist()
                        error_profiles.append({
                            "cluster_id": i, "error_count": errors, "error_rate": error_rate,
                            "error_modules": error_modules, "error_status_codes": error_status,
                            "avg_error_latency": avg_err_latency, "sample_errors": sample_errors
                        })

                st.session_state['clusters_summary'] = clusters_summary_list
                st.session_state['error_profiles'] = error_profiles

                # --- Create Analysis Tabs ---
                analysis_tab_titles = ["Cluster Explorer"]
                # Add other tabs only if summaries were generated
                if clusters_summary_list: analysis_tab_titles.append("Holistic Analysis")
                if error_profiles: analysis_tab_titles.append("Comparative Analysis")
                if log_df['datetime'].notna().any(): analysis_tab_titles.append("Temporal Analysis")

                analysis_tabs = st.tabs(analysis_tab_titles)
                tab_idx = 0

                # --- Cluster Explorer Tab ---
                with analysis_tabs[tab_idx]:
                    st.markdown("### Explore Individual Clusters")
                    selected_cluster = st.selectbox(
                        "Select cluster to explore",
                        options=valid_clusters,
                        format_func=lambda x: f"Cluster {x} ({len(log_df[log_df['cluster'] == x])} logs)",
                        key="cluster_selector"
                    )

                    cluster_logs = log_df[log_df["cluster"] == selected_cluster]
                    # Find the pre-calculated summary for this cluster
                    current_cluster_summary = next((cs for cs in clusters_summary_list if cs['cluster_id'] == selected_cluster), {})


                    exp_col1, exp_col2 = st.columns([2, 1]) # Log samples on left, stats/analysis on right

                    with exp_col1:
                        st.markdown(f"#### Sample Logs from Cluster {selected_cluster}")
                        st.caption(f"Showing up to 10 representative log entries.")
                        # Display samples from summary first, then add more if needed
                        displayed_count = 0
                        for sample in current_cluster_summary.get('sample_logs', []):
                             # Find level for styling (approximate based on content)
                             level_class = "info-card"
                             if "ERROR" in sample.upper(): level_class = "error-card"
                             elif "WARN" in sample.upper(): level_class = "warning-card"
                             st.markdown(f'<div class="{level_class}" style="margin-bottom: 5px; background-color: {COLORS["background"]}; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 0.85em;">{sample}</div>', unsafe_allow_html=True)
                             displayed_count += 1

                        # Add more samples if needed, up to 10
                        if displayed_count < 10:
                            more_samples = cluster_logs[~cluster_logs['raw'].isin(current_cluster_summary.get('sample_logs', []))]['raw'].head(10 - displayed_count).tolist()
                            for sample in more_samples:
                                level_class = "info-card"
                                if "ERROR" in sample.upper(): level_class = "error-card"
                                elif "WARN" in sample.upper(): level_class = "warning-card"
                                st.markdown(f'<div class="{level_class}" style="margin-bottom: 5px; background-color: {COLORS["background"]}; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 0.85em;">{sample}</div>', unsafe_allow_html=True)


                    with exp_col2:
                        st.markdown(f"#### Cluster {selected_cluster} Statistics")
                        if current_cluster_summary:
                            st.metric("Total Logs", current_cluster_summary['total_logs'])
                            st.metric("Error Rate", f"{current_cluster_summary['error_rate']}%", delta=f"{current_cluster_summary['error_count']} errors", delta_color="inverse" if current_cluster_summary['error_rate'] > 10 else "normal")
                            st.metric("Warning Rate", f"{current_cluster_summary['warning_rate']}%", delta=f"{current_cluster_summary['warning_count']} warnings", delta_color="off")
                            st.metric("Avg. Latency", f"{current_cluster_summary['avg_latency']:.1f} ms")
                            st.markdown("**Top Modules:**")
                            st.json(current_cluster_summary['top_modules'], expanded=False)
                            st.markdown("**Top Status Codes:**")
                            st.json(current_cluster_summary['top_status_codes'], expanded=False)
                        else:
                            st.info("Summary not available for this cluster.")

                        st.markdown("---")
                        st.markdown("#### 🧠 AI Analysis of Cluster Summary")
                        analyze_cluster_button = st.button(f"Analyze Summary for Cluster {selected_cluster}", key=f"analyze_summary_{selected_cluster}")
                        if analyze_cluster_button and current_cluster_summary:
                             cluster_analysis_summary = analyze_cluster_summary(
                                 selected_cluster,
                                 current_cluster_summary,
                                 use_ollama=use_ollama,
                                 ollama_model=ollama_model, ollama_url=ollama_url,
                                 remote_model=llm_model, api_key=api_key
                             )
                             st.markdown(f"""
                             <div class="card" style="background-color: {COLORS['background']}; border-left: 5px solid {COLORS['primary']};">
                                 <h5>✅ AI Insights for Cluster {selected_cluster}</h5>
                                 <small>{cluster_analysis_summary}</small>
                             </div>
                             """, unsafe_allow_html=True)
                        elif analyze_cluster_button:
                             st.warning("Cluster summary data not available for analysis.")

                tab_idx += 1

                # --- Holistic Analysis Tab ---
                if "Holistic Analysis" in analysis_tab_titles:
                    with analysis_tabs[tab_idx]:
                        st.markdown("### System-Wide Log Analysis")
                        st.markdown("Provides a high-level assessment based on overall stats and cluster summaries.")

                        holistic_col1, holistic_col2 = st.columns([1, 1])
                        with holistic_col1:
                            # Show cluster distribution (bar chart)
                            cluster_dist_df = log_df["cluster"].value_counts().reset_index()
                            cluster_dist_df.columns = ["Cluster", "Count"]
                            cluster_dist_df = cluster_dist_df.sort_values("Cluster")
                            fig = px.bar(cluster_dist_df, x="Cluster", y="Count", title="Log Distribution Across Clusters",
                                        template=chart_theme, color="Cluster", color_continuous_scale=px.colors.sequential.Blues_r)
                            fig.update_layout(xaxis={'type': 'category'}) # Treat cluster ID as category
                            st.plotly_chart(fig, use_container_width=True)
                        with holistic_col2:
                             # Show error rates by cluster (bar chart)
                            error_df = pd.DataFrame([{'Cluster': cs['cluster_id'], 'Error Rate': cs['error_rate']} for cs in clusters_summary_list])
                            error_df = error_df.sort_values("Cluster")
                            fig = px.bar(error_df, x="Cluster", y="Error Rate", title="Error Rate (%) by Cluster",
                                         template=chart_theme, color="Error Rate", color_continuous_scale=px.colors.sequential.Reds_r)
                            fig.update_layout(xaxis={'type': 'category'})
                            st.plotly_chart(fig, use_container_width=True)

                        st.markdown("#### 🧠 AI Holistic Analysis")
                        holistic_button = st.button("🔍 Generate Holistic System Analysis", key="holistic_analysis")
                        if holistic_button:
                            # Ensure we have the necessary summaries
                            log_summary_data = st.session_state.get('log_df_summary', {})
                            clusters_summary_data = st.session_state.get('clusters_summary', [])
                            if log_summary_data and clusters_summary_data:
                                holistic_analysis = perform_holistic_analysis(
                                    log_summary_data,
                                    clusters_summary_data,
                                    use_ollama=use_ollama, ollama_model=ollama_model, ollama_url=ollama_url,
                                    remote_model=llm_model, api_key=api_key
                                )
                                st.markdown(f"""
                                <div class="card" style="background-color: {COLORS['background']}; border-left: 5px solid {COLORS['primary']};">
                                    <h5>✅ System-Wide Analysis</h5>
                                    <small>{holistic_analysis}</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning("Log summary data is missing, cannot perform holistic analysis.")
                    tab_idx += 1

                # --- Comparative Analysis Tab ---
                if "Comparative Analysis" in analysis_tab_titles:
                    with analysis_tabs[tab_idx]:
                        st.markdown("### Comparative Error Analysis")
                        st.markdown("Compares error characteristics across clusters that contain errors.")

                        if error_profiles:
                             # Bar chart of error counts per cluster
                            comp_err_df = pd.DataFrame([{'Cluster': ep['cluster_id'], 'Error Count': ep['error_count']} for ep in error_profiles])
                            comp_err_df = comp_err_df.sort_values("Cluster")
                            fig = px.bar(comp_err_df, x="Cluster", y="Error Count", title="Error Count per Cluster",
                                         template=chart_theme, color="Error Count", color_continuous_scale=px.colors.sequential.Reds_r)
                            fig.update_layout(xaxis={'type': 'category'})
                            st.plotly_chart(fig, use_container_width=True)

                             # Display error profiles (optional: use expander if long)
                            st.markdown("#### Error Profile Summaries")
                            for profile in error_profiles:
                                with st.expander(f"Cluster {profile['cluster_id']} Error Profile ({profile['error_count']} errors)"):
                                    st.json(profile)

                            st.markdown("#### 🧠 AI Comparative Analysis")
                            comparative_button = st.button("🔍 Generate Comparative Error Analysis", key="comparative_analysis")
                            if comparative_button:
                                comparative_analysis = perform_comparative_analysis(
                                    error_profiles,
                                    use_ollama=use_ollama, ollama_model=ollama_model, ollama_url=ollama_url,
                                    remote_model=llm_model, api_key=api_key
                                )
                                st.markdown(f"""
                                <div class="card" style="background-color: {COLORS['background']}; border-left: 5px solid {COLORS['primary']};">
                                    <h5>✅ Comparative Error Insights</h5>
                                    <small>{comparative_analysis}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No clusters with errors found to perform comparative analysis.")
                    tab_idx += 1

                # --- Temporal Analysis Tab ---
                if "Temporal Analysis" in analysis_tab_titles:
                    with analysis_tabs[tab_idx]:
                        st.markdown("### Temporal Pattern Analysis")
                        st.markdown("Examines log distribution and error rates over time.")

                        time_df = log_df[log_df['datetime'].notna()].copy()
                        if not time_df.empty:
                            time_df['hour'] = time_df['datetime'].dt.hour
                            hourly_agg = time_df.groupby('hour').agg(
                                total_logs=('raw', 'count'),
                                error_count=('level', lambda x: (x == 'ERROR').sum()),
                                warning_count=('level', lambda x: (x == 'WARNING').sum()),
                                avg_latency=('latency', lambda x: pd.to_numeric(x, errors='coerce').fillna(0).mean())
                            ).reset_index()
                            hourly_agg['error_rate'] = (hourly_agg['error_count'] / hourly_agg['total_logs'] * 100).fillna(0)

                            # Plot Log Volume by Hour
                            fig_vol = px.bar(hourly_agg, x='hour', y='total_logs', title='Log Volume by Hour of Day', template=chart_theme)
                            st.plotly_chart(fig_vol, use_container_width=True)

                             # Plot Error Rate by Hour
                            fig_err = px.line(hourly_agg, x='hour', y='error_rate', title='Error Rate (%) by Hour of Day',
                                              template=chart_theme, markers=True)
                            fig_err.update_traces(line=dict(color=COLORS["error"], width=2))
                            st.plotly_chart(fig_err, use_container_width=True)

                            # Prepare data for LLM
                            hourly_stats_summary = hourly_agg[['hour', 'total_logs', 'error_rate', 'avg_latency']].to_string(index=False)
                            avg_error_rate = hourly_agg['error_rate'].mean()
                            anomalous_hours = hourly_agg[hourly_agg['error_rate'] > avg_error_rate * 1.5]['hour'].tolist()

                            st.markdown("#### 🧠 AI Temporal Analysis")
                            temporal_button = st.button("🔍 Generate Temporal Pattern Analysis", key="temporal_analysis")
                            if temporal_button:
                                temporal_analysis = perform_temporal_analysis(
                                    hourly_stats_summary,
                                    anomalous_hours,
                                    use_ollama=use_ollama, ollama_model=ollama_model, ollama_url=ollama_url,
                                    remote_model=llm_model, api_key=api_key
                                )
                                st.markdown(f"""
                                <div class="card" style="background-color: {COLORS['background']}; border-left: 5px solid {COLORS['primary']};">
                                    <h5>✅ Temporal Pattern Insights</h5>
                                     <small>{temporal_analysis}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("Insufficient timestamp data for temporal analysis.")
                    tab_idx += 1

            else: # Clustering failed or skipped
                 st.error("Log clustering could not be performed. AI analysis features requiring clusters are disabled.")

    # --- TAB 4: Advanced Visualizations ---
    with tab4:
        st.markdown("### 📈 Advanced Log Data Visualizations")

        viz_options = []
        if (pd.to_numeric(log_df["latency"], errors='coerce').fillna(0) > 0).any():
             viz_options.append("Latency Analysis")
        if log_df["status_code"].nunique() > 1 and (log_df["status_code"] != 'N/A').any():
            viz_options.append("Status Code Distribution")
        if log_df["module"].nunique() > 1:
            viz_options.append("Module Analysis")
        if log_df['datetime'].notna().any():
            viz_options.append("Time Analysis")

        if not viz_options:
            st.info("No suitable data found for advanced visualizations (check latency, status codes, modules, timestamps).")
        else:
            viz_type = st.selectbox(
                "Select visualization type",
                viz_options,
                key="viz_type_selector"
            )

            if viz_type == "Latency Analysis":
                st.markdown("#### Latency Distribution & Outliers")
                latency_df = log_df[pd.to_numeric(log_df["latency"], errors='coerce').fillna(0) > 0].copy()
                latency_df['latency'] = pd.to_numeric(latency_df['latency']) # Ensure numeric

                if not latency_df.empty:
                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        fig = px.histogram(latency_df, x="latency", color="level", nbins=30,
                                           title="Latency Distribution by Log Level", template=chart_theme,
                                           color_discrete_map={"ERROR": COLORS["error"], "WARNING": COLORS["warning"], "INFO": COLORS["success"], "DEBUG": COLORS["secondary"]})
                        st.plotly_chart(fig, use_container_width=True)
                    with viz_col2:
                        fig = px.box(latency_df, x="latency", y="module", color="level", points="outliers",
                                     title="Latency by Module (Log Scale)", template=chart_theme, log_x=True,
                                     color_discrete_map={"ERROR": COLORS["error"], "WARNING": COLORS["warning"], "INFO": COLORS["success"], "DEBUG": COLORS["secondary"]})
                        st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("No valid latency data (> 0) found.")


            elif viz_type == "Status Code Distribution":
                st.markdown("#### HTTP Status Code Analysis")
                status_df = log_df[log_df["status_code"].apply(lambda x: str(x).isdigit() and len(str(x)) == 3)].copy() # Ensure 3 digits
                status_df["status_code"] = status_df["status_code"].astype(int)

                if not status_df.empty:
                    status_df['status_category'] = status_df['status_code'].apply(
                        lambda x: 'Success (2xx)' if 200 <= x < 300 else
                                  'Redirect (3xx)' if 300 <= x < 400 else
                                  'Client Error (4xx)' if 400 <= x < 500 else
                                  'Server Error (5xx)' if 500 <= x < 600 else 'Other')

                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                         status_counts = status_df["status_code"].value_counts().reset_index()
                         status_counts.columns = ["Status Code", "Count"]
                         fig = px.bar(status_counts.head(15), x="Status Code", y="Count", title="Top 15 Status Codes",
                                      template=chart_theme, color="Count", color_continuous_scale=px.colors.sequential.Viridis)
                         fig.update_layout(xaxis={'type': 'category'})
                         st.plotly_chart(fig, use_container_width=True)
                    with viz_col2:
                         category_counts = status_df["status_category"].value_counts().reset_index()
                         category_counts.columns = ["Category", "Count"]
                         color_map = {"Success (2xx)": COLORS["success"], "Redirect (3xx)": COLORS["info"],
                                      "Client Error (4xx)": COLORS["warning"], "Server Error (5xx)": COLORS["error"], "Other": COLORS["secondary"]}
                         fig = px.pie(category_counts, values="Count", names="Category", title="Status Code Categories",
                                      color="Category", color_discrete_map=color_map, template=chart_theme)
                         st.plotly_chart(fig, use_container_width=True)

                    # Heatmap: Status vs Module
                    st.markdown("#### Status Codes per Module")
                    status_by_module = pd.crosstab(status_df["module"], status_df["status_category"])
                    if not status_by_module.empty:
                        fig = px.imshow(status_by_module, title="Status Code Categories by Module",
                                        color_continuous_scale=px.colors.sequential.Blues, template=chart_theme)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data for Status vs Module heatmap.")
                else:
                    st.info("No valid 3-digit HTTP status codes found.")


            elif viz_type == "Module Analysis":
                 st.markdown("#### Module Log Activity")
                 module_counts = log_df['module'].value_counts().reset_index()
                 module_counts.columns = ['Module', 'Count']

                 viz_col1, viz_col2 = st.columns(2)
                 with viz_col1:
                     fig = px.bar(module_counts.head(15), x='Count', y='Module', orientation='h',
                                 title="Top 15 Modules by Log Volume", template=chart_theme,
                                 color='Count', color_continuous_scale=px.colors.sequential.Blues_r)
                     fig.update_layout(yaxis={'categoryorder':'total ascending'})
                     st.plotly_chart(fig, use_container_width=True)

                 with viz_col2:
                     # Treemap of module activity
                    module_level_counts = log_df.groupby(['module', 'level']).size().reset_index(name='count')
                    if not module_level_counts.empty:
                        fig = px.treemap(module_level_counts, path=[px.Constant("All Modules"), 'module', 'level'], values='count',
                                        title='Module Activity by Log Level', template=chart_theme,
                                        color='level', color_discrete_map={"ERROR": COLORS["error"], "WARNING": COLORS["warning"], "INFO": COLORS["success"], "DEBUG": COLORS["secondary"], "(?)": COLORS['secondary']})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                         st.info("Not enough data for treemap.")


            elif viz_type == "Time Analysis":
                st.markdown("#### Log Activity Over Time")
                time_df = log_df[log_df['datetime'].notna()].copy()

                if not time_df.empty:
                    time_df = time_df.sort_values('datetime')
                    # Resample options
                    resample_freq = st.selectbox("Time Aggregation", ['Hour (H)', '30 Minutes (30T)', 'Day (D)'], index=0)
                    freq_map = {'Hour (H)': 'h', '30 Minutes (30T)': '30min', 'Day (D)': 'D'}
                    time_agg = time_df.set_index('datetime').resample(freq_map[resample_freq]).agg(
                        total_logs=('raw', 'count'),
                        error_count=('level', lambda x: (x == 'ERROR').sum())
                    ).reset_index()
                    time_agg['error_rate'] = (time_agg['error_count'] / time_agg['total_logs'] * 100).fillna(0)

                    fig = px.line(time_agg, x='datetime', y=['total_logs', 'error_count'], title=f'Log Volume & Errors ({resample_freq})',
                                 template=chart_theme, markers=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Daily patterns if data spans multiple days
                    if time_df['datetime'].dt.date.nunique() > 1:
                         st.markdown("#### Activity by Day of Week")
                         time_df['day_of_week'] = time_df['datetime'].dt.day_name()
                         day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                         daily_counts = time_df.groupby(['day_of_week', 'level']).size().unstack(fill_value=0).reindex(day_order)

                         fig_daily = go.Figure()
                         for level, color in [("ERROR", COLORS["error"]), ("WARNING", COLORS["warning"]), ("INFO", COLORS["success"])]:
                             if level in daily_counts.columns:
                                 fig_daily.add_trace(go.Bar(x=daily_counts.index, y=daily_counts[level], name=level, marker_color=color))
                         fig_daily.update_layout(barmode='stack', title="Log Volume by Day of Week", template=chart_theme)
                         st.plotly_chart(fig_daily, use_container_width=True)
                else:
                     st.info("No valid timestamp data for time analysis.")


# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 20px; padding-top: 15px; border-top: 1px solid #e6e6e6;">
    <p style="color: #6c757d; font-size: 0.9em;">AI-Powered Log Analyzer v1.1</p>
</div>
""", unsafe_allow_html=True)