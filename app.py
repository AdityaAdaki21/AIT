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
from logs_backend import fetch_logs
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
        Intelligent log analysis with machine learning clustering and local LLM insights
    </p>
</div>
""", unsafe_allow_html=True)

# Constants
OPENROUTER_API_KEY = "sk"
# OPENROUTER_API_KEY = "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"
OLLAMA_API_URL = "http://192.168.160.210:11434/api/generate"  # Default Ollama API URL

# --- 1. Data Loading and Preparation ---
def is_csv_file(filename=None, content=None):
    if filename and filename.endswith('.csv'):
        return True
    if content and len(content) > 0:
        return "timestamp" in content[0].lower() and "," in content[0]
    return False

def preprocess(logs):
    return [re.sub(r'[^a-zA-Z0-9 ]', ' ', log.lower()) for log in logs]

@st.cache_data
def extract_components(logs, is_csv=False):
    """Extract components from logs with improved performance"""
    if is_csv:
        try:
            df = pd.read_csv(io.StringIO('\n'.join(logs)))
            
            required_columns = ['timestamp', 'error', 'api_id', 'status_code', 'latency_ms', 'env']
            if all(col in df.columns for col in required_columns):
                df['level'] = df['error'].apply(lambda x: "ERROR" if int(x) == 1 else "INFO")
                df['module'] = df['api_id']
                df['message'] = df['api_id'] + " - Status: " + df['status_code'].astype(str) + " - Latency: " + df['latency_ms'].astype(str) + "ms"
                df['raw'] = df.apply(lambda row: ','.join(row.astype(str)), axis=1)
                df['latency'] = df['latency_ms'].astype(float)
                
                return df[['timestamp', 'level', 'module', 'message', 'raw', 'status_code', 'latency', 'env']]
        except Exception as e:
            st.warning(f"CSV parsing error: {str(e)}. Falling back to basic log format.")
    
    data = []
    for log in logs:
        # Extract timestamp with more flexible pattern matching
        timestamp_match = re.search(r'\[(.*?)\]', log)
        timestamp = timestamp_match.group(1) if timestamp_match else ""
        
        # Determine log level
        level = "INFO"  # Default
        if re.search(r'ERROR|FAIL|CRITICAL', log, re.IGNORECASE):
            level = "ERROR"
        elif re.search(r'WARN|WARNING', log, re.IGNORECASE):
            level = "WARNING"
        elif re.search(r'DEBUG', log, re.IGNORECASE):
            level = "DEBUG"
        
        # Extract module information more robustly
        module = "unknown"
        module_match = re.search(r'\](.*?):', log)
        if module_match:
            module = module_match.group(1).strip()
        
        # Extract status code if present
        status_match = re.search(r'status.*?(\d{3})', log, re.IGNORECASE)
        status_code = status_match.group(1) if status_match else "N/A"
        
        # Extract latency if present
        latency_match = re.search(r'(\d+\.?\d*)(?:\s*ms|\s*s)', log, re.IGNORECASE)
        latency = float(latency_match.group(1)) if latency_match else 0.0
        
        data.append({
            "timestamp": timestamp,
            "level": level,
            "module": module,
            "message": log,
            "raw": log,
            "status_code": status_code,
            "latency": latency,
            "env": "production" if "prod" in log.lower() else "development"
        })
    
    return pd.DataFrame(data)

def parse_timestamp(ts_str):
    # Try different timestamp formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S,%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%d/%b/%Y:%H:%M:%S",
        "%a %b %d %H:%M:%S %Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    
    return None

# --- Function to query Ollama API ---
# --- Enhanced LLM Analysis Functions ---

def perform_holistic_analysis(log_df, clusters, use_ollama=True, ollama_model="llama3.2.2:3b", ollama_url=None, remote_model=None, api_key=None):
    """
    Perform holistic analysis across all log clusters
    """
    # Prepare cluster summaries for context
    cluster_summaries = []
    for i in range(len(clusters.unique())):
        cluster_logs = log_df[log_df["cluster"] == i]
        
        # Get cluster stats
        total = len(cluster_logs)
        errors = len(cluster_logs[cluster_logs["level"] == "ERROR"])
        warnings = len(cluster_logs[cluster_logs["level"] == "WARNING"])
        
        # Get top modules and status codes
        top_modules = cluster_logs["module"].value_counts().head(3).to_dict()
        top_status = cluster_logs["status_code"].value_counts().head(3).to_dict()
        
        # Sample a few logs
        sample_logs = cluster_logs["raw"].sample(min(5, len(cluster_logs))).tolist()
        
        cluster_summaries.append({
            "cluster_id": i,
            "total_logs": total,
            "error_count": errors,
            "warning_count": warnings,
            "error_rate": round(errors/total*100 if total else 0, 2),
            "top_modules": top_modules,
            "top_status_codes": top_status,
            "sample_logs": sample_logs
        })
    
    # Create a prompt for holistic analysis
    system_prompt = """You are an expert log analysis AI assistant specializing in system diagnostics. 
    Analyze the provided log cluster summaries and provide a comprehensive assessment of the system's health."""
    
    prompt = f"""
    # System Log Analysis

    Analyze the following log clusters from a system and provide insights on:
    
    1. Overall system health
    2. Key issues detected across clusters
    3. Relationships between clusters
    4. Root causes of major errors
    5. Actionable recommendations
    
    ## Cluster Data
    
    {cluster_summaries}
    
    ## Analysis Instructions
    
    - Identify patterns across clusters
    - Highlight significant anomalies
    - Explain likely causal relationships between errors
    - Suggest monitoring priorities
    - Provide specific remediation steps
    """
    
    # Get analysis from LLM
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

def perform_comparative_analysis(log_df, clusters, use_ollama=True, ollama_model="llama3.2:3b", ollama_url=None, remote_model=None, api_key=None):
    """
    Compare error patterns between clusters
    """
    # Create detailed error profiles for each cluster
    error_profiles = []
    
    for i in range(len(clusters.unique())):
        cluster_logs = log_df[log_df["cluster"] == i]
        error_logs = cluster_logs[cluster_logs["level"] == "ERROR"]
        
        # Skip if no errors
        if len(error_logs) == 0:
            continue
            
        # Get error patterns
        error_modules = error_logs["module"].value_counts().head(3).to_dict()
        error_status = error_logs["status_code"].value_counts().head(3).to_dict()
        
        # Get latency stats if available
        avg_latency = error_logs["latency"].mean() if "latency" in error_logs.columns else 0
        
        # Sample error messages
        sample_errors = error_logs["raw"].sample(min(3, len(error_logs))).tolist()
        
        error_profiles.append({
            "cluster_id": i,
            "error_count": len(error_logs),
            "error_modules": error_modules,
            "error_status_codes": error_status,
            "avg_error_latency": avg_latency,
            "sample_errors": sample_errors
        })
    
    # Create a prompt for comparative analysis
    prompt = f"""
    # Comparative Error Analysis
    
    Compare the error patterns across these log clusters and provide insights on:
    
    1. Which cluster contains the most severe errors
    2. Similarities and differences in error patterns
    3. How errors might propagate across components
    4. Which errors are likely symptoms vs. root causes
    
    ## Error Profiles
    
    {error_profiles}
    
    ## Analysis Instructions
    
    - Identify the most critical error cluster
    - Highlight error propagation patterns
    - Distinguish between primary and secondary failures
    - Suggest error correlation and causation relationships
    """
    
    # Get analysis from LLM
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

def perform_temporal_analysis(log_df, use_ollama=True, ollama_model="llama3.2:3b", ollama_url=None, remote_model=None, api_key=None):
    """
    Analyze temporal patterns in logs
    """
    # Only proceed if we have datetime information
    if not log_df['datetime'].notna().any():
        return "Insufficient timestamp data for temporal analysis"
    
    # Create time-based summaries
    time_df = log_df[log_df['datetime'].notna()].copy()
    
    # Group by hour
    time_df['hour'] = time_df['datetime'].dt.hour
    hourly_stats = time_df.groupby('hour').agg({
        'level': lambda x: x.value_counts().to_dict(),
        'latency': 'mean',
        'module': lambda x: x.value_counts().head(2).to_dict()
    }).to_dict()
    
    # Find anomalous periods
    error_by_hour = time_df[time_df['level'] == 'ERROR'].groupby('hour').size()
    avg_errors = error_by_hour.mean()
    anomalous_hours = error_by_hour[error_by_hour > avg_errors * 1.5].to_dict()
    
    # Create a prompt for temporal analysis
    prompt = f"""
    # Temporal Log Analysis
    
    Analyze the temporal patterns in these system logs and provide insights on:
    
    1. Time-based patterns in errors and system behavior
    2. Anomalous time periods
    3. Potential scheduled jobs or system patterns
    4. Recommendations for monitoring windows
    
    ## Hourly Statistics
    
    {hourly_stats}
    
    ## Anomalous Hours (higher than average error rate)
    
    {anomalous_hours}
    
    ## Analysis Instructions
    
    - Identify peak error periods
    - Suggest correlation with business hours or system jobs
    - Recommend optimal monitoring windows
    - Highlight cyclical patterns if present
    """
    
    # Get analysis from LLM
    if use_ollama:
        return query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
    else:
        return query_remote_llm(prompt, model=remote_model, api_key=api_key)

def query_remote_llm(prompt, model, api_key):
    """Query remote LLM API"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            }
        )
        
        if response.status_code != 200:
            return f"API request failed with status code {response.status_code}: {response.text}"
        
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error querying remote LLM: {str(e)}"

# Enhanced query_ollama function
def query_ollama(prompt, model_name="llama3.2:3b", api_url="http://192.168.160.210:11434/api/generate"):
    """Query the local Ollama API with a prompt and get the response"""
    try:
        response = requests.post(
            api_url,
            json={"model": model_name, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: Received status code {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error connecting to Ollama API: {str(e)}"

# --- Function to check if Ollama is available ---
def is_ollama_available(api_url):
    """Check if Ollama service is available"""
    try:
        response = requests.get(f"{api_url.split('/api/')[0]}/api/tags")
        return response.status_code == 200
    except:
        return False

# --- Function to get available Ollama models ---
def get_ollama_models(api_url):
    """Get list of available models from Ollama"""
    try:
        response = requests.get(f"{api_url.split('/api/')[0]}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        return []
    except:
        return []

# --- 2. Data Loading UI ---
with st.sidebar:
    st.markdown("### 📁 Data Source")
    data_source = st.radio(
        "Choose data source",
        ["Sample Logs", "Upload Log File"],
        key="data_source_radio"
    )
    
    uploaded_file = None
    if data_source == "Upload Log File":
        uploaded_file = st.file_uploader("📤 Upload log file", type=["log", "txt", "csv"])
        
        if not uploaded_file:
            st.info("Please upload a log file or switch to sample logs")
    
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    
    n_clusters = st.slider(
        "Number of log clusters", 
        min_value=2, 
        max_value=8, 
        value=4,
        help="Higher values create more specific clusters"
    )
    
    # Ollama settings
    st.markdown("### 🤖 LLM Settings")
    use_ollama = st.checkbox("Use Local Ollama", value=True, 
                            help="Use locally running Ollama instead of remote API")
    
    if use_ollama:
        ollama_url = st.text_input("Ollama API URL", value="http://192.168.160.210:11434/api/generate",
                                help="URL of your local Ollama API")
        
        # Check if Ollama is running and get available models
        ollama_running = is_ollama_available(ollama_url)
        
        if ollama_running:
            st.success("✅ Connected to Ollama")
            available_models = get_ollama_models(ollama_url)
            
            if available_models:
                default_model = "llama3.2:3b" if "llama3.2:3b" in available_models else available_models[0]
                ollama_model = st.selectbox(
                    "Select Ollama Model",
                    options=available_models,
                    index=available_models.index(default_model) if default_model in available_models else 0
                )
            else:
                st.warning("No models found. Please pull a model in Ollama.")
                ollama_model = st.text_input("Model Name", value="llama3.2:3b")
        else:
            st.error("❌ Could not connect to Ollama. Is it running?")
            ollama_model = st.text_input("Model Name", value="llama3.2:3b")
    else:
        llm_model = st.selectbox(
            "Remote LLM Model",
            ["mistralai/mistral-small-3.1-24b-instruct:free", "anthropic/claude-3-haiku-3"],
            help="Select the remote AI model for log analysis"
        )
    
    st.markdown("---")
    st.markdown("### 🎨 Visualization")
    chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark"])
    
    # Add a pleasant image to the sidebar
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    This AI-powered log analyzer helps you:
    - Cluster similar log entries
    - Identify patterns and anomalies
    - Get AI-generated insights
    - Visualize log data trends
    """)

# Load logs based on selection
if uploaded_file:
    content = uploaded_file.read().decode("utf-8").splitlines()
    is_csv = is_csv_file(filename=uploaded_file.name, content=content)
    logs = content
    log_df = extract_components(logs, is_csv=is_csv)
else:
    logs = fetch_logs()
    log_df = extract_components(logs)

# Convert timestamps to datetime objects where possible
log_df['datetime'] = log_df['timestamp'].apply(parse_timestamp)

# --- 3. Create Tabs for Organization ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", 
    "🔍 Log Explorer", 
    "🧠 AI Analysis",
    "📈 Advanced Visualizations"
])

# --- TAB 1: Dashboard ---
with tab1:
    # Top metrics row
    st.markdown("### 📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        error_count = len(log_df[log_df["level"] == "ERROR"])
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {COLORS['error']}">
            <div class="metric-label">Errors</div>
            <div class="metric-value" style="color: {COLORS['error']}">{error_count}</div>
            <div class="metric-label">{round(error_count/len(log_df)*100)}% of logs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        warning_count = len(log_df[log_df["level"] == "WARNING"])
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {COLORS['warning']}">
            <div class="metric-label">Warnings</div>
            <div class="metric-value" style="color: {COLORS['warning']}">{warning_count}</div>
            <div class="metric-label">{round(warning_count/len(log_df)*100)}% of logs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        modules_count = log_df["module"].nunique()
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {COLORS['primary']}">
            <div class="metric-label">Unique Modules</div>
            <div class="metric-value" style="color: {COLORS['primary']}">{modules_count}</div>
            <div class="metric-label">Affected components</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        avg_latency = log_df["latency"].mean()
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {COLORS['info']}">
            <div class="metric-label">Avg Latency</div>
            <div class="metric-value" style="color: {COLORS['info']}">{avg_latency:.2f}ms</div>
            <div class="metric-label">Response time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Log Level Distribution")
        level_counts = log_df["level"].value_counts().reset_index()
        level_counts.columns = ["Level", "Count"]
        
        color_map = {
            "ERROR": COLORS["error"],
            "WARNING": COLORS["warning"],
            "INFO": COLORS["info"],
            "DEBUG": COLORS["secondary"]
        }
        
        colors = [color_map.get(level, COLORS["secondary"]) for level in level_counts["Level"]]
        
        fig = px.pie(
            level_counts, 
            values='Count', 
            names='Level',
            color='Level',
            color_discrete_map=color_map,
            hole=0.4,
            template=chart_theme
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### Module Breakdown")
        # Get top modules by error count
        module_counts = log_df.groupby("module")["level"].value_counts().unstack().fillna(0)
        
        if "ERROR" not in module_counts.columns:
            module_counts["ERROR"] = 0
        if "WARNING" not in module_counts.columns:
            module_counts["WARNING"] = 0
        if "INFO" not in module_counts.columns:
            module_counts["INFO"] = 0
            
        # Sort by error count and take top 5
        module_counts = module_counts.sort_values("ERROR", ascending=False).head(5)
        
        fig = go.Figure()
        for level, color in [("ERROR", COLORS["error"]), ("WARNING", COLORS["warning"]), ("INFO", COLORS["info"])]:
            if level in module_counts.columns:
                fig.add_trace(go.Bar(
                    y=module_counts.index,
                    x=module_counts[level],
                    name=level,
                    orientation='h',
                    marker_color=color
                ))
                
        fig.update_layout(
            barmode='stack',
            template=chart_theme,
            margin=dict(t=30, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title=''),
            xaxis=dict(title='Count')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Error timeline if we have datetime information
    if log_df['datetime'].notna().any():
        st.markdown("### Error Timeline")
        
        # Create a datetime indexed dataframe for time series analysis
        time_df = log_df[log_df['datetime'].notna()].copy()
        time_df['hour'] = time_df['datetime'].dt.hour
        
        # Count errors by hour
        error_by_hour = time_df.groupby(['hour', 'level']).size().unstack().fillna(0)
        
        # Ensure all levels are represented
        for level in ['ERROR', 'WARNING', 'INFO', 'DEBUG']:
            if level not in error_by_hour.columns:
                error_by_hour[level] = 0
        
        # Create a line chart
        fig = go.Figure()
        for level, color in [("ERROR", COLORS["error"]), ("WARNING", COLORS["warning"]), ("INFO", COLORS["info"])]:
            if level in error_by_hour.columns:
                fig.add_trace(go.Scatter(
                    x=error_by_hour.index,
                    y=error_by_hour[level],
                    mode='lines+markers',
                    name=level,
                    line=dict(color=color, width=2),
                    marker=dict(size=8)
                ))
                
        fig.update_layout(
            template=chart_theme,
            margin=dict(t=30, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title='Hour of Day'),
            yaxis=dict(title='Number of Logs')
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Log Explorer ---
with tab2:
    st.markdown("### 🔍 Filter Logs")
    
    # Filters in more attractive layout
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        selected_level = st.selectbox(
            "Log Level",
            ["All"] + sorted(log_df["level"].unique().tolist()),
            key="level_filter"
        )
        
    with filter_col2:
        module_options = ["All"] + sorted(log_df["module"].unique().tolist())
        selected_module = st.selectbox("Module", module_options, key="module_filter")
        
    with filter_col3:
        keyword = st.text_input("Search by keyword", key="keyword_filter")
    
    # More filters in collapsible section
    with st.expander("Advanced Filters"):
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            status_codes = ["All"] + sorted([str(code) for code in log_df["status_code"].unique().tolist()])
            selected_status = st.selectbox("Status Code", status_codes, key="status_filter")
            
        with adv_col2:
            env_options = ["All"] + sorted(log_df["env"].unique().tolist())
            selected_env = st.selectbox("Environment", env_options, key="env_filter")
    
    # Update the filtering logic
    filtered_df = log_df.copy()
    if selected_level != "All":
        filtered_df = filtered_df[filtered_df["level"] == selected_level]
    if selected_module != "All":
        filtered_df = filtered_df[filtered_df["module"] == selected_module]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["status_code"] == selected_status]
    if selected_env != "All":
        filtered_df = filtered_df[filtered_df["env"] == selected_env]
    if keyword:
        filtered_df = filtered_df[filtered_df["message"].str.contains(keyword, case=False)]
    
    # Display logs with colored styling
    st.markdown("### 📝 Log Entries")
    
    # Create a stylish display
    for _, row in filtered_df.iterrows():
        level_class = "info-card"
        if row["level"] == "ERROR":
            level_class = "error-card"
        elif row["level"] == "WARNING":
            level_class = "warning-card"
            
        st.markdown(f"""
        <div class="{level_class}" style="margin-bottom: 10px; background-color: {COLORS['background']}; padding: 10px; border-radius: 5px;">
            <div style="font-family: monospace; font-size: 0.9em;">
                <span style="color: {COLORS['secondary']}; font-weight: bold;">[{row['timestamp']}]</span> 
                <span style="font-weight: bold;">{row['module']}</span>: {row['message']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if len(filtered_df) == 0:
        st.info("No logs match your filter criteria.")

# --- TAB 3: AI Analysis ---
# --- TAB 3: AI Analysis (Enhanced) ---
with tab3:
    st.markdown("### 🧠 AI-Powered Log Analysis")
    
    # Cluster logs if we have enough data
    clean_logs = preprocess(log_df['raw'].tolist())
    
    if len(clean_logs) > 1:
        with st.spinner("Clustering logs..."):
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(clean_logs)
            
            # KMeans clustering
            n_clusters = min(n_clusters, len(clean_logs))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Add cluster labels to dataframe
            log_df["cluster"] = labels
        
        # Create tabs for different analysis approaches
        analysis_tabs = st.tabs([
            "Holistic Analysis", 
            "Cluster Explorer", 
            "Comparative Analysis",
            "Temporal Analysis"
        ])
        
        # TAB: Holistic Analysis
        with analysis_tabs[0]:
            st.markdown("### System-Wide Log Analysis")
            st.markdown("""
            This analysis examines patterns across all log clusters to provide a comprehensive view of system health.
            """)
            
            holistic_col1, holistic_col2 = st.columns([1, 1])
            
            with holistic_col1:
                # Show cluster distribution
                cluster_counts = log_df["cluster"].value_counts().reset_index()
                cluster_counts.columns = ["Cluster", "Count"]
                
                fig = px.bar(
                    cluster_counts, 
                    x="Cluster", 
                    y="Count",
                    title="Log Distribution Across Clusters",
                    template=chart_theme,
                    color="Cluster",
                    color_continuous_scale=px.colors.sequential.Blues
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with holistic_col2:
                # Show error rates by cluster
                error_rates = []
                for i in range(n_clusters):
                    cluster_logs = log_df[log_df["cluster"] == i]
                    total = len(cluster_logs)
                    errors = len(cluster_logs[cluster_logs["level"] == "ERROR"])
                    error_rates.append({
                        "Cluster": i,
                        "Error Rate": round(errors/total*100 if total else 0, 2)
                    })
                
                error_df = pd.DataFrame(error_rates)
                
                fig = px.bar(
                    error_df,
                    x="Cluster",
                    y="Error Rate",
                    title="Error Rate (%) by Cluster",
                    template=chart_theme,
                    color="Error Rate",
                    color_continuous_scale=px.colors.sequential.Reds
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Holistic LLM Analysis
            st.markdown("### 🧠 AI Holistic Analysis")
            
            holistic_button = st.button("🔍 Generate Holistic System Analysis")
            
            if holistic_button:
                with st.spinner("Analyzing system-wide patterns..."):
                    holistic_analysis = perform_holistic_analysis(
                        log_df,
                        log_df["cluster"],
                        use_ollama=use_ollama,
                        ollama_model=ollama_model if use_ollama else None,
                        ollama_url=ollama_url if use_ollama else None,
                        remote_model=llm_model if not use_ollama else None,
                        api_key=OPENROUTER_API_KEY if not use_ollama else None
                    )
                    
                    st.markdown(f"""
                    <div class="card" style="background-color: {COLORS['background']}; border-left: 5px solid {COLORS['primary']};">
                        <h4>✅ System-Wide Analysis</h4>
                        {holistic_analysis}
                    </div>
                    """, unsafe_allow_html=True)
        
        # TAB: Cluster Explorer
        with analysis_tabs[1]:
            st.markdown("### Explore Individual Clusters")
            
            selected_cluster = st.selectbox(
                "Select a cluster to explore",
                range(n_clusters),
                format_func=lambda x: f"Cluster {x}"
            )
            
            cluster_logs = log_df[log_df["cluster"] == selected_cluster]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### Cluster {selected_cluster} Sample Logs")
                
                # Display sample logs with better styling
                for _, row in cluster_logs.head(5).iterrows():
                    level_class = "info-card"
                    if row["level"] == "ERROR":
                        level_class = "error-card"
                    elif row["level"] == "WARNING":
                        level_class = "warning-card"
                    
                    st.markdown(f"""
                    <div class="{level_class}" style="margin-bottom: 10px; background-color: {COLORS['background']}; padding: 10px; border-radius: 5px;">
                        <div style="font-family: monospace; font-size: 0.9em;">
                            <span style="color: {COLORS['secondary']}; font-weight: bold;">[{row['timestamp']}]</span> 
                            <span style="font-weight: bold;">{row['module']}</span>: {row['message']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Show cluster stats
                total = len(cluster_logs)
                errors = len(cluster_logs[cluster_logs["level"] == "ERROR"])
                warnings = len(cluster_logs[cluster_logs["level"] == "WARNING"])
                
                # Calculate top modules in cluster
                top_modules = cluster_logs["module"].value_counts().head(3)
                
                st.markdown(f"""
                <div class="card">
                    <h4>Cluster Stats</h4>
                    <p><strong>Total logs:</strong> {total}</p>
                    <p><strong>Errors:</strong> {errors} ({round(errors/total*100 if total else 0)}%)</p>
                    <p><strong>Warnings:</strong> {warnings} ({round(warnings/total*100 if total else 0)}%)</p>
                    <h5>Top Modules:</h5>
                    <ul>
                        {"".join([f"<li>{module}: {count}</li>" for module, count in top_modules.items()])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Show distribution of log levels in this cluster
            level_counts = cluster_logs["level"].value_counts().reset_index()
            level_counts.columns = ["Level", "Count"]
            
            color_map = {
                "ERROR": COLORS["error"],
                "WARNING": COLORS["warning"],
                "INFO": COLORS["info"],
                "DEBUG": COLORS["secondary"]
            }
            
            fig = px.pie(
                level_counts, 
                values='Count', 
                names='Level',
                color='Level',
                color_discrete_map=color_map,
                title=f"Log Level Distribution in Cluster {selected_cluster}",
                template=chart_theme
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Clustered logs word cloud
            st.markdown("### Key Words in Cluster")
            
            # Display word frequency in this cluster
            from collections import Counter
            import re
            
            # Tokenize the logs
            all_words = []
            for log in cluster_logs["raw"]:
                # Remove special chars and split into words
                words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]+\b', log.lower())
                all_words.extend(words)
            
            # Count word frequency
            word_counts = Counter(all_words)
            
            # Remove common words
            for word in ['the', 'and', 'for', 'with', 'from', 'this', 'that']:
                if word in word_counts:
                    del word_counts[word]
            
            # Get top 20 words
            top_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
            
            fig = px.bar(
                top_words,
                x='Count',
                y='Word',
                title=f"Most Common Words in Cluster {selected_cluster}",
                template=chart_theme,
                orientation='h',
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual cluster analysis button
            st.markdown("### 🧠 Cluster Analysis")
            
            cluster_analyze_button = st.button(f"🔍 Analyze Cluster {selected_cluster}")
            
            if cluster_analyze_button:
                with st.spinner("Analyzing cluster..."):
                    # Prepare logs for LLM analysis
                    cluster_logs_list = cluster_logs["raw"].tolist()
                    if len(cluster_logs_list) > 200:
                        cluster_logs_list = random.sample(cluster_logs_list, 200)
                        st.info(f"Sampling 200 logs from {len(cluster_logs)} total logs in this cluster")
                    
                    cluster_text = "\n".join(cluster_logs_list)
                    
                    # Create prompt for cluster analysis
                    prompt = f"""You are an expert log analysis assistant. Given a set of application logs (structured or unstructured), extract actionable insights tailored to the following user personas:

1. Developers
2. DevOps Engineers / Site Reliability Engineers (SREs)
3. QA Testers
4. Security Analysts

For each persona:
- Identify the key log events relevant to them.
- Summarize potential issues or anomalies.
- Provide recommendations or next steps to address those issues.
- Mention specific components (e.g., auth, cache, DB, API) if applicable.

Also, highlight:
- Error patterns
- Latency or performance bottlenecks
- Security-related failures (invalid token, failed login, etc.)
- Repeated issues that may warrant further investigation

Here are the logs for Cluster {selected_cluster}:
---
{cluster_text}
---
"""
                    
                    if use_ollama:
                        summary = query_ollama(prompt, model_name=ollama_model, api_url=ollama_url)
                    else:
                        response = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                            json={
                                "model": llm_model,
                                "messages": [{
                                    "role": "user",
                                    "content": prompt
                                }]
                            }
                        )
                        
                        if response.status_code != 200:
                            st.error(f"API request failed with status code {response.status_code}: {response.text}")
                            summary = "Error processing request"
                        else:
                            result = response.json()
                            summary = result['choices'][0]['message']['content']
                    
                    # Display the analysis with styling
                    st.markdown(f"""
                    <div class="card" style="background-color: {COLORS['background']}; border-left: 5px solid {COLORS['primary']};">
                        <h4>✅ AI Analysis for Cluster {selected_cluster}</h4>
                        {summary}
                    </div>
                    """, unsafe_allow_html=True)
        
        # TAB: Comparative Analysis
        with analysis_tabs[2]:
            st.markdown("### Comparative Error Analysis")
            st.markdown("""
            This analysis compares error patterns across clusters to identify relationships between different types of errors.
            """)
            
            # Show error distributions across clusters
            error_by_cluster = log_df[log_df["level"] == "ERROR"].groupby("cluster").size().reset_index()
            error_by_cluster.columns = ["Cluster", "Error Count"]
            
            fig = px.bar(
                error_by_cluster,
                x="Cluster",
                y="Error Count",
                title="Error Distribution Across Clusters",
                template=chart_theme,
                color="Error Count",
                color_continuous_scale=px.colors.sequential.Reds
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show module distribution by cluster
            module_by_cluster = pd.crosstab(log_df["cluster"], log_df["module"]).T
            
            # Keep only top modules for readability
            top_modules = log_df["module"].value_counts().head(8).index
            filtered_modules = module_by_cluster.loc[module_by_cluster.index.isin(top_modules)]
            
            fig = px.imshow(
                filtered_modules,
                title="Module Distribution by Cluster",
                template=chart_theme,
                labels=dict(x="Cluster", y="Module", color="Count"),
                color_continuous_scale=px.colors.sequential.Blues
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparative LLM Analysis
            st.markdown("### 🧠 AI Comparative Analysis")
            
            comparative_button = st.button("🔍 Generate Comparative Error Analysis")
            
            if comparative_button:
                with st.spinner("Analyzing error patterns across clusters..."):
                    comparative_analysis = perform_comparative_analysis(
                        log_df,
                        log_df["cluster"],
                        use_ollama=use_ollama,
                        ollama_model=ollama_model if use_ollama else None,
                        ollama_url=ollama_url if use_ollama else None,
                        remote_model=llm_model if not use_ollama else None,
                        api_key=OPENROUTER_API_KEY if not use_ollama else None
                    )
                    
                    st.markdown(f"""
                    <div class="card" style="background-color: {COLORS['background']}; border-left: 5px solid {COLORS['primary']};">
                        <h4>✅ Comparative Error Analysis</h4>
                        {comparative_analysis}
                    </div>
                    """, unsafe_allow_html=True)
        
        # TAB: Temporal Analysis
        with analysis_tabs[3]:
            st.markdown("### Temporal Pattern Analysis")
            st.markdown("""
            This analysis examines how logs and errors are distributed over time to identify cyclical patterns.
            """)
            
            if log_df['datetime'].notna().any():
                # Create a datetime indexed dataframe for time series analysis
                time_df = log_df[log_df['datetime'].notna()].copy()
                time_df['hour'] = time_df['datetime'].dt.hour
                
                # Count errors by hour
                logs_by_hour = time_df.groupby(['hour', 'level']).size().unstack().fillna(0)
                
                # Ensure all levels are represented
                for level in ['ERROR', 'WARNING', 'INFO', 'DEBUG']:
                    if level not in logs_by_hour.columns:
                        logs_by_hour[level] = 0
                
                # Create a line chart
                fig = go.Figure()
                for level, color in [("ERROR", COLORS["error"]), ("WARNING", COLORS["warning"]), ("INFO", COLORS["info"])]:
                    if level in logs_by_hour.columns:
                        fig.add_trace(go.Scatter(
                            x=logs_by_hour.index,
                            y=logs_by_hour[level],
                            mode='lines+markers',
                            name=level,
                            line=dict(color=color, width=2),
                            marker=dict(size=8)
                        ))
                        
                fig.update_layout(
                    title="Log Volume by Hour of Day",
                    template=chart_theme,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title='Hour of Day'),
                    yaxis=dict(title='Number of Logs')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show error rates by hour
                error_rates = []
                for hour in time_df['hour'].unique():
                    hour_logs = time_df[time_df['hour'] == hour]
                    total = len(hour_logs)
                    errors = len(hour_logs[hour_logs["level"] == "ERROR"])
                    error_rates.append({
                        "Hour": hour,
                        "Error Rate": round(errors/total*100 if total else 0, 2)
                    })
                
                error_df = pd.DataFrame(error_rates).sort_values("Hour")
                
                fig = px.line(
                    error_df,
                    x="Hour",
                    y="Error Rate",
                    title="Error Rate (%) by Hour of Day",
                    template=chart_theme,
                    markers=True
                )
                fig.update_traces(line=dict(color=COLORS["error"], width=3))
                st.plotly_chart(fig, use_container_width=True)
                
                # Temporal LLM Analysis
                st.markdown("### 🧠 AI Temporal Analysis")
                
                temporal_button = st.button("🔍 Generate Temporal Pattern Analysis")
                
                if temporal_button:
                    with st.spinner("Analyzing temporal patterns..."):
                        temporal_analysis = perform_temporal_analysis(
                            log_df,
                            use_ollama=use_ollama,
                            ollama_model=ollama_model if use_ollama else None,
                            ollama_url=ollama_url if use_ollama else None,
                            remote_model=llm_model if not use_ollama else None,
                            api_key=OPENROUTER_API_KEY if not use_ollama else None
                        )
                        
                        st.markdown(f"""
                        <div class="card" style="background-color: {COLORS['background']}; border-left: 5px solid {COLORS['primary']};">
                            <h4>✅ Temporal Pattern Analysis</h4>
                            {temporal_analysis}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Insufficient timestamp data for temporal analysis. Please use logs with valid timestamps.")
    else:
        st.info("Not enough log data for clustering and analysis. Please upload more logs.")

# --- TAB 4: Advanced Visualizations ---
with tab4:
    st.markdown("### 📈 Advanced Log Analysis")
    
    viz_type = st.radio(
        "Select visualization type",
        ["Latency Analysis", "Status Code Distribution", "Module Correlation", "Time Analysis"],
        horizontal=True
    )
    
    if viz_type == "Latency Analysis":
        st.markdown("### API Latency Analysis")
        
        # Filter out entries with zero latency
        latency_df = log_df[log_df["latency"] > 0].copy()
        
        if len(latency_df) > 0:
            # Create visualization columns
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Create histogram of latencies
                fig = px.histogram(
                    latency_df,
                    x="latency",
                    color="level",
                    color_discrete_map={
                        "ERROR": COLORS["error"],
                        "WARNING": COLORS["warning"], 
                        "INFO": COLORS["info"],
                        "DEBUG": COLORS["secondary"]
                    },
                    nbins=20,
                    title="Response Time Distribution",
                    template=chart_theme
                )
                fig.update_layout(
                    xaxis_title="Latency (ms)",
                    yaxis_title="Count",
                    legend_title="Log Level"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                # Box plot by module
                fig = px.box(
                    latency_df,
                    y="module",
                    x="latency",
                    color="level",
                    color_discrete_map={
                        "ERROR": COLORS["error"],
                        "WARNING": COLORS["warning"],
                        "INFO": COLORS["info"],
                        "DEBUG": COLORS["secondary"]
                    },
                    title="Latency by Module",
                    template=chart_theme
                )
                fig.update_layout(
                    xaxis_title="Latency (ms)",
                    yaxis_title="Module"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Add histogram of latencies by module
            module_latencies = latency_df.groupby("module")["latency"].mean().sort_values(ascending=False)
            
            fig = px.bar(
                module_latencies,
                title="Average Latency by Module",
                template=chart_theme
            )
            fig.update_layout(
                xaxis_title="Module",
                yaxis_title="Avg Latency (ms)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data available in the logs.")
            
    elif viz_type == "Status Code Distribution":
        st.markdown("### HTTP Status Code Analysis")
        
        # Filter for valid status codes (numeric)
        status_df = log_df[log_df["status_code"].str.isnumeric()]
        
        if len(status_df) > 0:
            # Categorize status codes
            def categorize_status(code):
                code = str(code)
                if code.startswith('2'):
                    return 'Success (2xx)'
                elif code.startswith('3'):
                    return 'Redirect (3xx)'
                elif code.startswith('4'):
                    return 'Client Error (4xx)'
                elif code.startswith('5'):
                    return 'Server Error (5xx)'
                else:
                    return 'Other'
            
            status_df['status_category'] = status_df['status_code'].apply(categorize_status)
            
            # Create visualization columns
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Create count of status codes
                status_counts = status_df["status_code"].value_counts().reset_index()
                status_counts.columns = ["Status Code", "Count"]
                
                # Color map based on status code
                color_list = []
                for code in status_counts["Status Code"]:
                    if str(code).startswith('2'):
                        color_list.append(COLORS["success"])
                    elif str(code).startswith('3'):
                        color_list.append(COLORS["info"])
                    elif str(code).startswith('4'):
                        color_list.append(COLORS["warning"])
                    elif str(code).startswith('5'):
                        color_list.append(COLORS["error"])
                    else:
                        color_list.append(COLORS["secondary"])
                
                fig = px.bar(
                    status_counts,
                    x="Status Code",
                    y="Count",
                    title="Status Code Distribution",
                    template=chart_theme
                )
                fig.update_traces(marker_color=color_list)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                # Create pie chart for status code categories
                category_counts = status_df["status_category"].value_counts().reset_index()
                category_counts.columns = ["Category", "Count"]
                
                color_map = {
                    "Success (2xx)": COLORS["success"],
                    "Redirect (3xx)": COLORS["info"],
                    "Client Error (4xx)": COLORS["warning"],
                    "Server Error (5xx)": COLORS["error"],
                    "Other": COLORS["secondary"]
                }
                
                fig = px.pie(
                    category_counts,
                    values="Count",
                    names="Category",
                    title="Status Code Categories",
                    color="Category",
                    color_discrete_map=color_map,
                    template=chart_theme
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            # Add heatmap of status codes by module
            status_by_module = pd.crosstab(status_df["module"], status_df["status_code"])
            
            fig = px.imshow(
                status_by_module,
                title="Status Codes by Module",
                color_continuous_scale=px.colors.sequential.Blues,
                template=chart_theme
            )
            fig.update_layout(
                xaxis_title="Status Code",
                yaxis_title="Module"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No HTTP status code data available in the logs.")
            
    elif viz_type == "Module Correlation":
        st.markdown("### Module Correlation Analysis")
        
        if len(log_df) > 0:
            # Create a correlation matrix between modules
            module_dummies = pd.get_dummies(log_df["module"])
            module_corr = module_dummies.corr()
            
            # Filter out low correlations for clarity
            threshold = 0.3
            module_corr_filtered = module_corr.copy()
            for i in range(len(module_corr_filtered.columns)):
                for j in range(len(module_corr_filtered.columns)):
                    if i != j and abs(module_corr_filtered.iloc[i, j]) < threshold:
                        module_corr_filtered.iloc[i, j] = 0
            
            # Create heatmap
            fig = px.imshow(
                module_corr_filtered,
                title="Module Correlation (Shows which modules tend to log together)",
                color_continuous_scale=px.colors.diverging.RdBu_r,
                zmin=-1,
                zmax=1,
                template=chart_theme
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Network graph of modules
            if len(module_corr.columns) > 1:
                st.markdown("### Module Network Graph")
                st.info("This visualization shows connections between modules based on correlation strength.")
                
                # Create network edges
                edges = []
                for i in range(len(module_corr.columns)):
                    for j in range(i+1, len(module_corr.columns)):
                        if abs(module_corr.iloc[i, j]) > threshold:
                            edges.append((
                                module_corr.columns[i],
                                module_corr.columns[j],
                                abs(module_corr.iloc[i, j])
                            ))
                
                if edges:
                    # Create network graph
                    import networkx as nx
                    
                    G = nx.Graph()
                    for source, target, weight in edges:
                        G.add_edge(source, target, weight=weight)
                    
                    pos = nx.spring_layout(G, seed=42)
                    
                    edge_x = []
                    edge_y = []
                    edge_weights = []
                    
                    for edge in G.edges(data="weight"):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_weights.append(edge[2])
                    
                    node_x = []
                    node_y = []
                    node_sizes = []
                    node_texts = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_sizes.append(len(log_df[log_df["module"] == node]) * 0.5)
                        node_texts.append(node)
                    
                    # Create plot
                    fig = go.Figure()
                    
                    # Add edges
                    fig.add_trace(go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1, color=COLORS["secondary"]),
                        hoverinfo='none',
                        mode='lines',
                        showlegend=False
                    ))
                    
                    # Add nodes
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        marker=dict(
                            showscale=True,
                            color=node_sizes,
                            size=node_sizes,
                            colorscale='Blues',
                            line_width=2,
                            colorbar=dict(
                                thickness=15,
                                title='Log Frequency',
                                xanchor='left',
                                titleside='right'
                            )
                        ),
                        text=node_texts,
                        textposition="top center",
                        hoverinfo='text',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title="Module Network Graph (size indicates log frequency)",
                        template=chart_theme,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough correlation between modules to create a network graph.")
        else:
            st.info("Not enough data to generate module correlation analysis.")
            
    elif viz_type == "Time Analysis":
        st.markdown("### Log Time Analysis")
        
        # Filter for logs with valid datetime
        time_df = log_df[log_df['datetime'].notna()].copy()
        
        if len(time_df) > 0:
            time_df['day'] = time_df['datetime'].dt.day_name()
            time_df['hour'] = time_df['datetime'].dt.hour
            
            # Create plots
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Logs by hour of day
                hourly_logs = time_df.groupby(['hour', 'level']).size().unstack().fillna(0)
                
                fig = go.Figure()
                for level, color in [("ERROR", COLORS["error"]), ("WARNING", COLORS["warning"]), ("INFO", COLORS["info"])]:
                    if level in hourly_logs.columns:
                        fig.add_trace(go.Bar(
                            x=hourly_logs.index,
                            y=hourly_logs[level],
                            name=level,
                            marker_color=color
                        ))
                
                fig.update_layout(
                    title="Log Volume by Hour of Day",
                    barmode='stack',
                    xaxis_title="Hour (24h)",
                    yaxis_title="Number of Logs",
                    template=chart_theme
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                # Error rate by hour
                hourly_error_rate = (time_df[time_df['level'] == 'ERROR'].groupby('hour').size() / 
                                    time_df.groupby('hour').size() * 100).fillna(0)
                
                fig = px.line(
                    x=hourly_error_rate.index,
                    y=hourly_error_rate.values,
                    markers=True,
                    title="Error Rate by Hour of Day",
                    template=chart_theme
                )
                fig.update_traces(line=dict(color=COLORS["error"], width=3))
                fig.update_layout(
                    xaxis_title="Hour (24h)",
                    yaxis_title="Error Rate (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show daily patterns if we have data across multiple days
            if time_df['day'].nunique() > 1:
                st.markdown("### Daily Patterns")
                
                # Logs by day of week
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_logs = time_df.groupby(['day', 'level']).size().unstack().fillna(0)
                daily_logs = daily_logs.reindex(day_order)
                
                fig = go.Figure()
                for level, color in [("ERROR", COLORS["error"]), ("WARNING", COLORS["warning"]), ("INFO", COLORS["info"])]:
                    if level in daily_logs.columns:
                        fig.add_trace(go.Bar(
                            x=daily_logs.index,
                            y=daily_logs[level],
                            name=level,
                            marker_color=color
                        ))
                
                fig.update_layout(
                    title="Log Volume by Day of Week",
                    barmode='stack',
                    xaxis_title="Day",
                    yaxis_title="Number of Logs",
                    template=chart_theme
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timestamp data available for time analysis.")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e6e6e6;">
    <p style="color: #6c757d;">AI-Powered Log Analyzer v1.0</p>
</div>
""", unsafe_allow_html=True)