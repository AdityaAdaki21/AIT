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
        Intelligent log analysis with machine learning clustering and LLM insights
    </p>
</div>
""", unsafe_allow_html=True)

# Constants
OPENROUTER_API_KEY = "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"
API_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

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
    
    llm_model = st.selectbox(
        "LLM Model",
        ["mistralai/mistral-small-3.1-24b-instruct:free", "anthropic/claude-3-haiku-3"],
        help="Select the AI model for log analysis"
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
        
        # Display clusters in a tabbed view
        cluster_tabs = st.tabs([f"Cluster {i+1}" for i in range(n_clusters)])
        
        for i, cluster_tab in enumerate(cluster_tabs):
            with cluster_tab:
                cluster_logs = log_df[log_df["cluster"] == i]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### Cluster {i+1} Sample Logs")
                    
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
                
                # LLM Analysis
                st.markdown("### LLM Root Cause Analysis")
                
                # Prepare logs for LLM analysis
                cluster_logs_list = cluster_logs["raw"].tolist()
                if len(cluster_logs_list) > 200:
                    cluster_logs_list = random.sample(cluster_logs_list, 200)
                    st.info(f"Sampling 200 logs from {len(cluster_logs)} total logs in this cluster")
                
                cluster_text = "\n".join(cluster_logs_list)
                
                analyze_button = st.button(f"🔍 Analyze Cluster {i+1} with LLM", key=f"analyze_button_{i}")
                
                if analyze_button:
                    with st.spinner("Analyzing logs with AI..."):
                        try:
                            response = requests.post(
                                "https://openrouter.ai/api/v1/chat/completions",
                                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                                json={
                                    "model": API_MODEL,
                                    "messages": [{
                                        "role": "user",
                                        "content": f"""You are an expert log analysis assistant. Given a set of application logs (structured or unstructured), extract actionable insights tailored to the following user personas:

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

Here are the logs:
---
{cluster_text}
---
"""
                                    }]
                                }
                            )
                            
                            if response.status_code != 200:
                                st.error(f"API request failed with status code {response.status_code}: {response.text}")
                            else:
                                result = response.json()
                                summary = result['choices'][0]['message']['content']
                                
                                # Display the analysis with styling
                                st.markdown(f"""
                                <div class="card" style="background-color: {COLORS['background']}; border-left: 5px solid {COLORS['primary']};">
                                    <h4>✅ AI Analysis Results</h4>
                                    <div style="padding: 10px;">{summary}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"Error analyzing logs: {str(e)}")
    else:
        st.warning("Not enough logs for clustering. Please upload more logs.")

# --- TAB 4: Advanced Visualizations ---
with tab4:
    st.markdown("### 📊 Advanced Visualizations")
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["Log Patterns", "Error Analysis", "Performance Metrics"])
    
    with viz_tabs[0]:  # Log Patterns
        st.markdown("### Log Patterns Over Time")
        
        if log_df['datetime'].notna().any():
            # Create time-based visualizations
            time_df = log_df[log_df['datetime'].notna()].copy()
            time_df['date'] = time_df['datetime'].dt.date
            time_df['hour'] = time_df['datetime'].dt.hour
            
            # Group by date and level
            daily_logs = time_df.groupby(['date', 'level']).size().unstack().fillna(0)
            
            # Ensure all levels are present
            for level in ['ERROR', 'WARNING', 'INFO', 'DEBUG']:
                if level not in daily_logs.columns:
                    daily_logs[level] = 0
            
            fig = px.line(
                daily_logs.reset_index().melt(id_vars=['date'], value_vars=['ERROR', 'WARNING', 'INFO', 'DEBUG']),
                x='date',
                y='value',
                color='variable',
                color_discrete_map={
                    'ERROR': COLORS['error'],
                    'WARNING': COLORS['warning'],
                    'INFO': COLORS['info'],
                    'DEBUG': COLORS['secondary']
                },
                template=chart_theme,
                markers=True,
                labels={'date': 'Date', 'value': 'Count', 'variable': 'Log Level'}
            )
            
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=30, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add heatmap visualization
            st.markdown("### Hourly Log Activity Heatmap")
            
            # Create a pivot table of hours vs levels
            hourly_logs = time_df.groupby(['hour', 'level']).size().unstack().fillna(0)
            
            heatmap_data = []
            for hour in range(24):
                if hour in hourly_logs.index:
                    for level in ['ERROR', 'WARNING', 'INFO', 'DEBUG']:
                        if level in hourly_logs.columns:
                            heatmap_data.append({
                                'Hour': hour,
                                'Level': level,
                                'Count': hourly_logs.loc[hour, level]
                            })
                        else:
                            heatmap_data.append({
                                'Hour': hour,
                                'Level': level,
                                'Count': 0
                            })
                else:
                    for level in ['ERROR', 'WARNING', 'INFO', 'DEBUG']:
                        heatmap_data.append({
                            'Hour': hour,
                            'Level': level,
                            'Count': 0
                        })
            
            heatmap_df = pd.DataFrame(heatmap_data)
            
            fig = px.density_heatmap(
                heatmap_df,
                x='Hour',
                y='Level',
                z='Count',
                color_continuous_scale=[COLORS['background'], COLORS['primary']],
                template=chart_theme
            )
            
            fig.update_layout(
                xaxis=dict(title='Hour of Day', tickmode='linear', tick0=0, dtick=1),
                yaxis=dict(title='Log Level', categoryorder='array', categoryarray=['DEBUG', 'INFO', 'WARNING', 'ERROR']),
                margin=dict(t=30, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Timestamp data is not available in the correct format for time-based visualizations.")
    
    with viz_tabs[1]:  # Error Analysis
        st.markdown("### Error Analysis")
        
        # Create a status code breakdown
        if 'status_code' in log_df.columns:
            error_df = log_df[log_df['level'] == 'ERROR'].copy()
            
            if not error_df.empty:
                st.markdown("#### Status Code Distribution for Errors")
                
                status_counts = error_df['status_code'].value_counts().reset_index()
                status_counts.columns = ['Status Code', 'Count']
                
                # Assign colors based on status code ranges
                def get_status_color(code):
                    try:
                        code_num = int(code)
                        if code_num < 400:
                            return COLORS['success']
                        elif code_num < 500:
                            return COLORS['warning']
                        else:
                            return COLORS['error']
                    except:
                        return COLORS['secondary']
                
                status_counts['Color'] = status_counts['Status Code'].apply(get_status_color)
                
                fig = px.bar(
                    status_counts,
                    x='Status Code',
                    y='Count',
                    color='Status Code',
                    color_discrete_map={code: color for code, color in zip(status_counts['Status Code'], status_counts['Color'])},
                    template=chart_theme
                )
                
                fig.update_layout(
                    margin=dict(t=30, b=20, l=20, r=20),
                    xaxis=dict(title='Status Code'),
                    yaxis=dict(title='Count')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Module error frequency
                st.markdown("#### Modules with Most Errors")
                
                module_errors = error_df['module'].value_counts().reset_index()
                module_errors.columns = ['Module', 'Error Count']
                
                fig = px.bar(
                    module_errors.head(10),
                    x='Module',
                    y='Error Count',
                    color='Error Count',
                    color_continuous_scale=[[0, COLORS['info']], [1, COLORS['error']]],
                    template=chart_theme
                )
                
                fig.update_layout(
                    margin=dict(t=30, b=20, l=20, r=20),
                    xaxis=dict(title='Module'),
                    yaxis=dict(title='Error Count')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No errors found in the logs.")
        else:
            st.warning("Status code information is not available in the logs.")
    
    with viz_tabs[2]:  # Performance Metrics
        st.markdown("### Performance Metrics")
        
        # Create latency visualizations if available
        if 'latency' in log_df.columns and log_df['latency'].notna().any():
            latency_df = log_df[log_df['latency'] > 0].copy()
            
            if not latency_df.empty:
                # Latency by module
                st.markdown("#### Average Latency by Module")
                
                module_latency = latency_df.groupby('module')['latency'].mean().reset_index()
                module_latency.columns = ['Module', 'Avg Latency (ms)']
                module_latency = module_latency.sort_values('Avg Latency (ms)', ascending=False)
                
                fig = px.bar(
                    module_latency.head(10),
                    x='Module',
                    y='Avg Latency (ms)',
                    color='Avg Latency (ms)',
                    color_continuous_scale=[[0, COLORS['success']], [0.5, COLORS['warning']], [1, COLORS['error']]],
                    template=chart_theme
                )
                
                fig.update_layout(
                    margin=dict(t=30, b=20, l=20, r=20),
                    xaxis=dict(title='Module'),
                    yaxis=dict(title='Average Latency (ms)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Latency distribution
                st.markdown("#### Latency Distribution")
                
                fig = px.histogram(
                    latency_df,
                    x='latency',
                    nbins=20,
                    color_discrete_sequence=[COLORS['primary']],
                    template=chart_theme
                )
                
                fig.update_layout(
                    margin=dict(t=30, b=20, l=20, r=20),
                    xaxis=dict(title='Latency (ms)'),
                    yaxis=dict(title='Count')
                )
                
                fig.add_vline(
                    x=latency_df['latency'].mean(),
                    line_dash="dash",
                    line_color=COLORS['error'],
                    annotation_text="Mean",
                    annotation_position="top right"
                )
                
                fig.add_vline(
                    x=latency_df['latency'].median(),
                    line_dash="dash",
                    line_color=COLORS['info'],
                    annotation_text="Median",
                    annotation_position="top left"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Latency vs Error scatter plot
                st.markdown("#### Latency vs Log Level")
                
                fig = px.scatter(
                    latency_df,
                    x='latency',
                    y='level',
                    color='level',
                    size='latency',
                    color_discrete_map={
                        'ERROR': COLORS['error'],
                        'WARNING': COLORS['warning'],
                        'INFO': COLORS['info'],
                        'DEBUG': COLORS['secondary']
                    },
                    template=chart_theme,
                    opacity=0.7
                )
                
                fig.update_layout(
                    margin=dict(t=30, b=20, l=20, r=20),
                    xaxis=dict(title='Latency (ms)'),
                    yaxis=dict(title='Log Level')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No latency information found in the logs.")
        else:
            st.warning("Latency information is not available in the logs.")

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #eee;">
    <p style="color: #666;">AI-Powered Log Analyzer • Built with Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Add help tooltip
with st.sidebar:
    st.markdown("---")
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        **Tips for using the Log Analyzer:**
        
        1. **Data Loading:** Upload a log file or use the sample logs
        2. **Dashboard:** View key metrics and distributions
        3. **Log Explorer:** Filter logs by various criteria
        4. **AI Analysis:** Get ML clustering and LLM insights
        5. **Advanced Visualizations:** Explore patterns and performance metrics
        
        **Supported Log Formats:**
        - Standard text logs with timestamps
        - CSV logs with specific columns
        """)