# app.py
import streamlit as st
import pandas as pd
import time
import queue # Needed for state init

# --- Project Modules ---
from config import (
    MIN_LOGS_FOR_CLUSTER, PAGE_TITLE, PAGE_ICON, LAYOUT, INITIAL_SIDEBAR_STATE, COLORS, REQUIRED_COLUMNS,
    UI_UPDATE_INTERVAL_SECONDS, MIN_LOGS_FOR_UI_UPDATE, APP_VERSION
)
from utils import calculate_dashboard_metrics, get_stable_hash
from ui_components import (
    make_summary_serializable, render_sidebar, render_dashboard, render_log_explorer,
    render_ai_analysis_tab, render_advanced_viz_tab
)
from data_manager import process_log_queue # Import queue processing

# --- Page Configuration ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# --- Custom CSS ---
# (Keep CSS here or move to a separate css file and load it)
st.markdown(f"""
<style>
    /* Base & Layout */
    .stApp {{ background-color: {COLORS["background"]}; }}
    .main .block-container {{ padding: 2rem 1.5rem 3rem; max-width: 1400px; margin: auto; }}
    h1, h2, h3 {{ color: {COLORS["primary"]}; font-weight: 600; }}
    h4, h5, h6 {{ color: {COLORS["text"]}; font-weight: 500; }}
    /* Sidebar */
    .st-emotion-cache-16txtl3 {{ padding: 1rem; }}
    .st-emotion-cache-16txtl3 h3 {{ margin-top: 1rem; margin-bottom: 0.5rem; font-size: 1.1em; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 5px; border-bottom: 2px solid #eee; }}
    .stTabs [data-baseweb="tab"] {{ padding: 10px 20px; border-radius: 6px 6px 0 0; background-color: #eee; color: {COLORS["light_text"]}; transition: background-color 0.3s ease; margin-bottom: -2px; font-weight: 500; }}
    .stTabs [aria-selected="true"] {{ background-color: {COLORS["primary"]}; color: white; font-weight: 600; border-bottom: 2px solid {COLORS["primary"]}; }}
    .stTabs [data-baseweb="tab"]:hover {{ background-color: #ddd; }}
    .stTabs [aria-selected="true"]:hover {{ background-color: {COLORS["primary"]}; }}
    /* Cards */
    .card {{ background-color: {COLORS["card"]}; border-radius: 8px; padding: 1.25rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 1rem; border: 1px solid #e9ecef; }}
    .metric-card {{ background-color: {COLORS["card"]}; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; height: 130px; display: flex; flex-direction: column; justify-content: center; border: 1px solid #e9ecef; transition: box-shadow 0.3s ease; }}
    .metric-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
    .metric-label {{ font-size: 14px; color: {COLORS["secondary"]}; margin-bottom: 5px; font-weight: 500; }}
    .metric-value {{ font-size: 26px; font-weight: 600; margin: 5px 0; line-height: 1.2; }}
    /* Log Explorer Entry */
    .log-entry-container {{ margin-bottom: 12px; padding: 12px; border-radius: 6px; background-color: {COLORS["card"]}; border-left: 5px solid; transition: background-color 0.2s ease; }}
    .log-entry-container:hover {{ background-color: #f8f9fa; }}
    .log-entry-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; font-size: 0.9em; }}
    .log-timestamp {{ color: {COLORS["secondary"]}; font-family: monospace; }}
    .log-level {{ font-weight: bold; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; color: white; }}
    .log-module {{ font-weight: bold; color: {COLORS["primary"]}; margin-left: 8px; font-family: monospace; }}
    .log-message {{ font-family: monospace; font-size: 0.95em; color: {COLORS["text"]}; word-break: break-all; margin-top: 5px; }}
    /* Status Indicator */
    .status-indicator {{ padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; display: inline-block; margin-left: 8px; vertical-align: middle;}}
    .status-connected {{ background-color: {COLORS['success']}; color: white; }}
    .status-disconnected {{ background-color: {COLORS['secondary']}; color: white; }}
    .status-connecting {{ background-color: {COLORS['warning']}; color: {COLORS['text']}; }}
    .status-error {{ background-color: {COLORS['error']}; color: white; }}
    /* Buttons */
    .stButton > button {{ border-radius: 6px; border: 1px solid {COLORS['primary']}; background-color: {COLORS['primary']}; color: white; padding: 8px 16px; font-weight: 500; transition: all 0.3s ease; }}
    .stButton > button:hover {{ background-color: #0056b3; border-color: #0056b3; }}
    .stButton > button:disabled {{ background-color: #cccccc; border-color: #cccccc; color: #666666; cursor: not-allowed; }}
    /* Specific Borders */
    .error-card-border {{ border-left-color: {COLORS["error"]}; }}
    .warning-card-border {{ border-left-color: {COLORS["warning"]}; }}
    .info-card-border {{ border-left-color: {COLORS["success"]}; }}
    .debug-card-border {{ border-left-color: {COLORS["secondary"]}; }}
    .parse-error-card-border {{ border-left-color: #FFA500; }}
    /* AI Response */
    .ai-response-box {{ background-color: #f0f9ff; border-left: 5px solid {COLORS['info']}; padding: 15px; border-radius: 5px; margin-top: 10px; font-size: 0.95em; }}
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown(f"""
<div style="background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['info']} 100%); padding: 25px 30px; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    <h1 style="color: white; margin: 0; display: flex; align-items: center; font-weight: 700; font-size: 2.2rem;">
        <span style="font-size: 2.5rem; margin-right: 15px;">{PAGE_ICON}</span> {PAGE_TITLE}
    </h1>
    <p style="color: white; opacity: 0.9; margin-top: 8px; font-size: 1.1rem;">
        Intelligent analysis for real-time streams & log files. (v{APP_VERSION})
    </p>
</div>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes all necessary session state variables if they don't exist."""
    defaults = {
        'log_df': pd.DataFrame(columns=REQUIRED_COLUMNS), 'sse_thread': None, 'sse_stop_event': None,
        'log_queue': queue.Queue(), 'sse_connection_status': "disconnected", 'sse_last_error': None,
        'sse_url': None, 'last_ui_update_time': 0, 'current_data_source': "Real-time Stream",
        'max_stream_logs': 5000, 'n_clusters': 4, 'chart_theme': 'plotly_white',
        'log_page': 1, 'log_explorer_filter_hash': None, 'last_uploaded_file_info': None,
        'sample_logs_loaded': False, 'clusters_summary': None, 'error_profiles': None,
        'log_df_summary': None, 'log_df_summary_cache_key': None, 'ollama_available': None,
        'ollama_status_msg': None, 'ollama_models': [], 'use_ollama': None, 'ollama_model': None,
        'ollama_url': None, '_last_ollama_check_url': None, 'remote_model': None, 'api_key': None,
        'required_columns': REQUIRED_COLUMNS,
        'min_logs_for_cluster': MIN_LOGS_FOR_CLUSTER, # Store config in state if needed by components
        'selected_tab_key': "📊 Dashboard", # Track active tab
        'explorer_prefilter_cluster': None, # Store cluster ID to pre-filter explorer
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# --- Render Sidebar ---
render_sidebar()


# --- Process Log Queue (If Streaming) ---
queue_processed_count = 0
now = time.time()
should_rerun = False
conn_status_before = st.session_state.get('sse_connection_status')
is_streaming_active = conn_status_before in ["connected", "connecting"]
queue_has_items = not st.session_state.get('log_queue', queue.Queue()).empty()
if is_streaming_active or queue_has_items:
    max_logs = st.session_state.get('max_stream_logs', 5000)
    queue_processed_count = process_log_queue(max_logs)
last_update_time = st.session_state.get('last_ui_update_time', 0)
time_since_last_update = now - last_update_time
conn_status_after = st.session_state.get('sse_connection_status')
if queue_processed_count > 0 and (time_since_last_update > UI_UPDATE_INTERVAL_SECONDS or queue_processed_count >= MIN_LOGS_FOR_UI_UPDATE): should_rerun = True
elif queue_has_items and time_since_last_update > (UI_UPDATE_INTERVAL_SECONDS / 2): should_rerun = True
elif is_streaming_active and queue_processed_count == 0 and time_since_last_update > UI_UPDATE_INTERVAL_SECONDS * 1.5: should_rerun = True
elif conn_status_before != conn_status_after: should_rerun = True
if should_rerun: st.session_state['last_ui_update_time'] = now


# --- Main App Area ---
log_df = st.session_state.get('log_df', pd.DataFrame(columns=REQUIRED_COLUMNS))
theme = st.session_state.get('chart_theme')
llm_config = { # Prepare LLM config once
    "use_ollama": st.session_state.get('use_ollama', False), "ollama_model": st.session_state.get('ollama_model'),
    "ollama_url": st.session_state.get('ollama_url'), "remote_model": st.session_state.get('remote_model'),
    "api_key": st.session_state.get('api_key')
}


# --- Display Content Based on State ---
if log_df.empty and not is_streaming_active:
    # Show appropriate message based on selected data source
    data_source_msg_map = { "Upload Log File": "⬅️ Please upload a log file using the sidebar.", "Sample Logs": "⬅️ Click 'Load Sample Logs' in the sidebar.", "Real-time Stream": "⬅️ Enter the SSE URL and click 'Connect to Stream'.", }
    msg = data_source_msg_map.get(st.session_state.get('current_data_source'), "⬅️ Welcome! Select a data source from the sidebar.")
    icon_map = {"Upload Log File": "📤", "Sample Logs": "💡", "Real-time Stream": "📡"}
    icon = icon_map.get(st.session_state.get('current_data_source'), "👋")
    st.info(msg, icon=icon)
elif log_df.empty and st.session_state.get('sse_connection_status') == "connecting":
    st.info("⏳ Attempting to connect to the real-time log stream...", icon="📡")
elif log_df.empty and st.session_state.get('sse_connection_status') == "connected":
     st.info("📡 Connected to stream. Waiting for incoming log data...", icon="⏱️")
else:
    # --- Calculate/Cache Dashboard Metrics ---
    current_df_hash = get_stable_hash(log_df)
    if st.session_state.get("log_df_summary_cache_key") != current_df_hash:
        log_df_summary = calculate_dashboard_metrics(log_df)
        st.session_state['log_df_summary'] = log_df_summary
        st.session_state['log_df_summary_cache_key'] = current_df_hash
    else:
        log_df_summary = st.session_state.get('log_df_summary', {})
    # Make summary serializable for potential use in AI tabs
    log_summary_serializable = make_summary_serializable(log_df_summary)


    # --- Create Main Tabs ---
    tab_titles = ["📊 Dashboard", "🔍 Log Explorer"]
    if not log_df.empty: # Only show analysis/viz tabs if there's data
         tab_titles.extend(["🧠 AI Analysis", "📈 Advanced Visualizations"])
    requested_tab = st.session_state.get('selected_tab_key', "📊 Dashboard")
    if requested_tab in tab_titles:
         default_tab_index = tab_titles.index(requested_tab)
    else: # Fallback if requested tab isn't available (e.g., no data for AI tab)
         default_tab_index = 0
         st.session_state['selected_tab_key'] = tab_titles[0] # Reset state

    tabs = st.tabs(tab_titles)
    tab_map = {title: tab for title, tab in zip(tab_titles, tabs)}

    # Render content within each tab using UI component functions
    with tab_map["📊 Dashboard"]:
        render_dashboard(log_df, log_df_summary, theme)

    with tab_map["🔍 Log Explorer"]:
        render_log_explorer(log_df, llm_config)

    if "🧠 AI Analysis" in tab_map:
        with tab_map["🧠 AI Analysis"]:
            # Pass the *current* log_df, which might include the 'cluster' column
            render_ai_analysis_tab(st.session_state.log_df, llm_config, theme)

    if "📈 Advanced Visualizations" in tab_map:
        with tab_map["📈 Advanced Visualizations"]:
            render_advanced_viz_tab(log_df, theme)


# --- Footer ---
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; margin-top: 25px; padding-top: 15px; border-top: 1px solid #e0e0e0;">
    <p style="color: #6c757d; font-size: 0.9em;">
       {PAGE_ICON} {PAGE_TITLE} v{APP_VERSION} | Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)


# --- Final Rerun Trigger ---
if should_rerun:
     time.sleep(0.05)
     try: st.rerun()
     except Exception as rerun_err: print(f"Error during st.rerun(): {rerun_err}")