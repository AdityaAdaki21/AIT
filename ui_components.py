# ui_components.py
import streamlit as st
import pandas as pd
import numpy as np # Import numpy
import re
import time
import json
from typing import Dict, Optional, List
import plotly.express as px
# --- Project Modules ---
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, INITIAL_SIDEBAR_STATE, COLORS, REQUIRED_COLUMNS,
    UI_UPDATE_INTERVAL_SECONDS, MIN_LOGS_FOR_UI_UPDATE, APP_VERSION, DEFAULT_SSE_LOG_URL, OLLAMA_API_URL,
    CHART_THEME_OPTIONS, CHART_THEME_MAP, LOG_LEVEL_STYLES, DEFAULT_N_CLUSTERS, MIN_LOGS_FOR_CLUSTER,
    MAX_CLUSTERS_HEURISTIC_DIVISOR, LOG_EXPLORER_PAGE_SIZE, OLLAMA_PREFERRED_MODELS, REMOTE_LLM_OPTIONS,
    DEFAULT_REMOTE_MODEL
)
from data_manager import (
    load_uploaded_file, load_sample_logs, start_sse_thread, stop_sse_thread,
)
from llm_interface import (
    check_ollama_availability, get_ollama_models, explain_single_log,
    perform_holistic_analysis, perform_comparative_analysis, perform_temporal_analysis
)
from clustering import run_log_clustering, get_ai_cluster_interpretation
from visualization import (
    create_log_level_pie_chart, create_top_modules_error_bar_chart, create_log_timeline_chart,
    create_latency_histogram, create_latency_boxplot_by_module,
    create_status_code_bar_chart, create_status_code_pie_chart, create_status_module_heatmap,
    create_module_volume_bar_chart, create_module_level_treemap,
    create_detailed_timeseries_chart, create_day_hour_heatmap, create_errors_by_day_chart,
    create_cluster_distribution_chart, create_cluster_error_rate_chart, create_comparative_error_count_chart
)
from utils import get_stable_hash, calculate_dashboard_metrics 
from visualization import ( # Import necessary viz functions
    create_cluster_distribution_chart, create_cluster_error_rate_chart,
    create_comparative_error_count_chart
)

# --- Helper Function for Serialization ---
def make_summary_serializable(summary_dict: Dict) -> Dict:
    """Converts non-JSON serializable types (Timestamp, numpy numbers) in a dict."""
    serializable_summary = {}
    if not isinstance(summary_dict, dict):
         print(f"Warning: make_summary_serializable expected a dict, got {type(summary_dict)}")
         return {}
    for key, value in summary_dict.items():
        if isinstance(value, pd.Timestamp):
            serializable_summary[key] = value.isoformat() if pd.notna(value) else None
        elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
             serializable_summary[key] = int(value)
        elif isinstance(value, (np.float16, np.float32, np.float64)):
             serializable_summary[key] = float(value)
        elif isinstance(value, (np.ndarray,)):
             # Handle nested structures if necessary, basic tolist for now
             try: serializable_summary[key] = value.tolist()
             except: serializable_summary[key] = str(value) # Fallback
        elif isinstance(value, np.bool_):
             serializable_summary[key] = bool(value)
        elif isinstance(value, dict): # Recursively handle nested dicts
             serializable_summary[key] = make_summary_serializable(value)
        elif isinstance(value, list): # Handle lists (check elements?)
             # Basic list handling, assumes elements are mostly serializable or handled above
             # For more complex lists, might need recursive checks
             serializable_summary[key] = value
        else:
            # Check if it's some other unserializable type? For now, assume basic types pass.
            serializable_summary[key] = value
    return serializable_summary


# ... (rest of the ui_components.py file remains the same) ...

# --- Sidebar Rendering ---
def display_sse_status():
    """Displays the SSE connection status indicator in the sidebar."""
    conn_status = st.session_state.get('sse_connection_status', 'disconnected')
    status_class = f"status-{conn_status}"
    status_text = conn_status.capitalize().replace("_", " ")
    icon_map = {"connected": "✅", "connecting": "⏳", "disconnected": "🔌", "error": "❌"}
    status_icon = icon_map.get(conn_status, "")

    st.markdown(f"**Stream Status:** {status_icon} {status_text} <span class='status-indicator {status_class}'></span>", unsafe_allow_html=True)
    if conn_status == 'error' and st.session_state.get('sse_last_error'):
        st.error(f"Error Detail: {st.session_state.get('sse_last_error')}", icon="🚨")


def render_sidebar():
    """Renders the entire sidebar UI and handles related state updates."""
    with st.sidebar:
        st.markdown("### 📁 Data Source")
        data_source_options = ["Real-time Stream", "Upload Log File", "Sample Logs"]
        try:
            current_ds_index = data_source_options.index(st.session_state.get('current_data_source', "Real-time Stream"))
        except ValueError:
            current_ds_index = 0

        data_source = st.radio(
            "Choose data source:",
            options=data_source_options,
            key="data_source_radio",
            index=current_ds_index,
        )

        # --- Handle Data Source Change ---
        if data_source != st.session_state.get('current_data_source'):
            print(f"Switching data source from '{st.session_state.get('current_data_source')}' to '{data_source}'")
            previous_source = st.session_state.get('current_data_source')
            st.session_state['current_data_source'] = data_source # Update state

            # Stop existing stream if switching away from it
            if previous_source == "Real-time Stream" and st.session_state.get('sse_thread') is not None:
                stop_sse_thread()

            # Reset log data and associated state for the new source
            # Keep existing log_df if switching *to* Upload/Sample *from* Stream? Optional.
            # For clean switch, let's always reset the df unless explicitly kept.
            st.session_state['log_df'] = pd.DataFrame(columns=st.session_state.get('required_columns', [])) # Use stored required_cols
            st.session_state['clusters_summary'] = None
            st.session_state['error_profiles'] = None
            st.session_state['log_df_summary'] = None
            st.session_state['log_df_summary_cache_key'] = None
            st.session_state['last_uploaded_file_info'] = None
            st.session_state['sample_logs_loaded'] = False
            st.session_state['log_page'] = 1
            st.session_state['log_explorer_filter_hash'] = None
            # Clear cluster column if present
            if 'cluster' in st.session_state.get('log_df', pd.DataFrame()).columns:
                st.session_state.log_df = st.session_state.log_df.drop(columns=['cluster'], errors='ignore')

            st.rerun() # Rerun to reflect the changes immediately


        # --- Source Specific UI ---
        data_loaded = False
        if data_source == "Upload Log File":
            uploaded_file = st.file_uploader(
                "📤 Upload log file (.log, .txt, .csv, .jsonl)",
                type=["log", "txt", "csv", "jsonl", "json"],
                key="log_file_uploader"
            )
            data_loaded = load_uploaded_file(uploaded_file)
            if not data_loaded and uploaded_file is None and st.session_state.get('last_uploaded_file_info') is not None:
                 # File removed from uploader, clear state if needed (load_uploaded_file handles this now)
                 pass
            elif uploaded_file is None and st.session_state.get('log_df', pd.DataFrame()).empty:
                 st.info("Upload a log file to begin analysis.")

        elif data_source == "Sample Logs":
            load_sample_button = st.button("Load Sample Logs", key="load_sample_button", use_container_width=True)
            should_load_sample = load_sample_button or \
                                 (st.session_state.get('log_df', pd.DataFrame()).empty and \
                                  not st.session_state.get('sample_logs_loaded', False))

            if should_load_sample:
                 data_loaded = load_sample_logs() # Function now handles state updates and success message
                 if data_loaded:
                      st.rerun() # Rerun after loading sample data

            if not st.session_state.get('log_df', pd.DataFrame()).empty:
                 st.info(f"Sample logs ({len(st.session_state.log_df)} entries) loaded.")
            elif not should_load_sample:
                 st.info("Click 'Load Sample Logs' to load demo data.")


        elif data_source == "Real-time Stream":
            st.markdown("### 📡 Real-time Settings")
            sse_url = st.text_input("SSE Log Stream URL", value=st.session_state.get('sse_url', DEFAULT_SSE_LOG_URL), key="sse_url_input")
            st.session_state['sse_url'] = sse_url # Keep URL in state

            display_sse_status() # Show connection status

            conn_status = st.session_state.get('sse_connection_status', 'disconnected')
            is_connected_or_connecting = conn_status in ["connected", "connecting"]
            button_text = "🔌 Disconnect Stream" if is_connected_or_connecting else "🚀 Connect to Stream"
            button_type = "secondary" if is_connected_or_connecting else "primary"

            if st.button(button_text, key="sse_connect_button", type=button_type, use_container_width=True):
                if is_connected_or_connecting:
                    stop_sse_thread()
                    time.sleep(0.1) # Allow state update propagation
                    st.rerun()
                else:
                    if sse_url and sse_url.startswith(("http://", "https://")):
                        start_sse_thread(sse_url)
                        st.rerun()
                    else:
                        st.error("Invalid SSE URL. Must start with http:// or https://")

            # Stream Configuration
            log_count_df = len(st.session_state.get('log_df', pd.DataFrame()))
            st.caption(f"Logs in memory: {log_count_df:,}")
            max_logs_val = st.session_state.get('max_stream_logs', 5000)
            max_logs = st.number_input("Max logs to keep", min_value=100, max_value=100000,
                                       value=max_logs_val, step=500, key="max_stream_logs_input",
                                       help="Limits memory usage. Oldest logs discarded.")
            if max_logs != max_logs_val:
                st.session_state['max_stream_logs'] = max_logs


        # --- Common Settings ---
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")
        current_log_count = len(st.session_state.get('log_df', pd.DataFrame()))
        clustering_enabled = current_log_count >= MIN_LOGS_FOR_CLUSTER

        if clustering_enabled:
            max_possible_clusters = max(2, min(15, current_log_count // MAX_CLUSTERS_HEURISTIC_DIVISOR))
            default_clusters = min(st.session_state.get('n_clusters', DEFAULT_N_CLUSTERS), max_possible_clusters)
            default_clusters = max(2, default_clusters) # Ensure min 2

            n_clusters_val = st.slider(
                "Number of log clusters", min_value=2, max_value=max_possible_clusters,
                value=default_clusters,
                help=f"Group similar logs ({MIN_LOGS_FOR_CLUSTER}+ logs needed). Max: {max_possible_clusters}",
                key="n_clusters_slider"
            )
            if n_clusters_val != st.session_state.get('n_clusters'):
                st.session_state['n_clusters'] = n_clusters_val
                # Invalidate cluster results if N changes
                st.session_state['clusters_summary'] = None
                st.session_state['error_profiles'] = None
                if 'cluster' in st.session_state.get('log_df', pd.DataFrame()).columns:
                    st.session_state.log_df = st.session_state.log_df.drop(columns=['cluster'], errors='ignore')
        else:
            st.caption(f"Clustering disabled (requires ≥ {MIN_LOGS_FOR_CLUSTER} logs. Current: {current_log_count})")
            st.slider("Number of log clusters", 2, 10, DEFAULT_N_CLUSTERS, disabled=True)

        st.markdown("### 🤖 LLM Settings")
        # Check Ollama status once per run if not already checked recently
        ollama_api_url_state = st.session_state.get('ollama_url', OLLAMA_API_URL)
        if 'ollama_available' not in st.session_state or st.session_state.get('_last_ollama_check_url') != ollama_api_url_state:
            is_available, status_msg = check_ollama_availability(ollama_api_url_state)
            st.session_state['ollama_available'] = is_available
            st.session_state['ollama_status_msg'] = status_msg
            st.session_state['_last_ollama_check_url'] = ollama_api_url_state # Track which URL was checked
            st.session_state['ollama_models'] = get_ollama_models(ollama_api_url_state, is_available)

        ollama_is_ready = st.session_state.get('ollama_available', False)

        # Default provider based on availability
        if 'use_ollama' not in st.session_state:
            st.session_state['use_ollama'] = ollama_is_ready

        llm_provider = st.radio("Choose LLM Provider", ["Local Ollama", "Remote OpenRouter"],
                                index=0 if st.session_state['use_ollama'] else 1,
                                horizontal=True, key="llm_provider_radio")
        use_ollama_choice = (llm_provider == "Local Ollama")

        if use_ollama_choice != st.session_state.get('use_ollama'):
            st.session_state['use_ollama'] = use_ollama_choice
            st.rerun()

        # LLM Provider Specific Settings
        if use_ollama_choice:
            ollama_url_input = st.text_input("Ollama API URL", value=ollama_api_url_state, key="ollama_url_input")
            if ollama_url_input != ollama_api_url_state:
                st.session_state['ollama_url'] = ollama_url_input
                # Force recheck on next run by clearing availability state
                st.session_state['ollama_available'] = None
                st.rerun()

            if ollama_is_ready:
                st.success("✅ Ollama Detected")
                available_models = st.session_state.get('ollama_models', [])
                if available_models:
                    current_model = st.session_state.get('ollama_model')
                    default_model = None
                    if current_model in available_models:
                         default_model = current_model
                    else: # Find first preferred model that exists
                        for pref in OLLAMA_PREFERRED_MODELS:
                            pref_base = pref.split(':')[0]
                            matching = [m for m in available_models if m.startswith(pref_base)]
                            if matching:
                                 default_model = sorted(matching)[0]
                                 break
                    if not default_model: default_model = available_models[0] # Fallback

                    try: current_model_index = available_models.index(default_model)
                    except ValueError: current_model_index = 0

                    selected_model = st.selectbox("Select Ollama Model", options=available_models, index=current_model_index, key="ollama_model_select")
                    st.session_state['ollama_model'] = selected_model
                else:
                    st.warning("Ollama connected, but no models found/listed. Pull a model first.")
                    fallback_model = st.text_input("Enter Ollama Model Name", value=st.session_state.get('ollama_model', OLLAMA_PREFERRED_MODELS[0]), key="ollama_model_input_fallback")
                    st.session_state['ollama_model'] = fallback_model
            else:
                st.error(f"❌ Ollama: {st.session_state.get('ollama_status_msg', 'Connection failed')}")
                error_model = st.text_input("Ollama Model Name (if running)", value=st.session_state.get('ollama_model', OLLAMA_PREFERRED_MODELS[0]), key="ollama_model_input_error")
                st.session_state['ollama_model'] = error_model

        else: # Remote OpenRouter
            api_key_state = st.session_state.get('api_key', "") # Use empty string if not set
            api_key_input = st.text_input("OpenRouter API Key", value=api_key_state, type="password", key="openrouter_api_key_input")
            if api_key_input != api_key_state:
                 st.session_state['api_key'] = api_key_input

            current_remote_model = st.session_state.get('remote_model', DEFAULT_REMOTE_MODEL)
            if current_remote_model not in REMOTE_LLM_OPTIONS: current_remote_model = DEFAULT_REMOTE_MODEL
            try: current_remote_index = REMOTE_LLM_OPTIONS.index(current_remote_model)
            except ValueError: current_remote_index = 0

            selected_remote_model = st.selectbox("Select Remote LLM Model", options=REMOTE_LLM_OPTIONS, index=current_remote_index, key="remote_llm_select")
            st.session_state['remote_model'] = selected_remote_model


        # --- Visualization Settings ---
        st.markdown("---")
        st.markdown("### 🎨 Visualization")
        reverse_theme_map = {v: k for k, v in CHART_THEME_MAP.items()}
        current_theme_code = st.session_state.get('chart_theme', 'plotly_white')
        current_theme_label = reverse_theme_map.get(current_theme_code, "Plotly White")
        try: default_theme_index = CHART_THEME_OPTIONS.index(current_theme_label)
        except ValueError: default_theme_index = 1 # Default to Plotly White

        chart_theme_label_select = st.selectbox("Chart Theme", CHART_THEME_OPTIONS, index=default_theme_index, key="chart_theme_select")
        selected_theme_code = CHART_THEME_MAP[chart_theme_label_select]
        if selected_theme_code != st.session_state.get('chart_theme'):
            st.session_state['chart_theme'] = selected_theme_code


        # --- About Section ---
        st.markdown("---")
        st.markdown("### 💡 About")
        st.info(f"""
        **AI Log Analyzer v{APP_VERSION}**
        Intelligent analysis for logs.
        - Dashboard, Explorer, AI Analysis, Adv. Viz.
        """)

# --- Dashboard Rendering ---

def render_dashboard_metrics(log_df_summary: Dict):
    """Displays the key metrics using st.metric."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Logs", value=f"{log_df_summary.get('total_logs', 0):,}")
        st.metric(label="Log Time Span", value=log_df_summary.get('log_time_span_str', 'N/A'))
    with col2:
        err_rate = log_df_summary.get('error_rate', 0.0)
        st.metric(label="Errors", value=f"{log_df_summary.get('error_count', 0):,}",
                  delta=f"{err_rate:.1f}% Error Rate",
                  delta_color="inverse" if err_rate > 5 else "normal")
        st.metric(label="Unique Modules", value=f"{log_df_summary.get('unique_modules', 0)}")
    with col3:
         st.metric(label="Warnings", value=f"{log_df_summary.get('warning_count', 0):,}",
                   delta=f"{log_df_summary.get('warning_rate', 0.0):.1f}% Warning Rate", delta_color="off")
         st.metric(label="Avg. Latency", value=f"{log_df_summary.get('avg_latency', 0.0):.1f} ms")

def render_dashboard(log_df: pd.DataFrame, log_df_summary: Dict, theme: Optional[str]):
    """Renders the dashboard tab."""
    st.markdown("### 📈 Key Metrics Overview")
    st.caption("Summary statistics derived from the currently loaded log data.")
    render_dashboard_metrics(log_df_summary)

    st.markdown("---")
    st.markdown("### 📊 Visual Insights")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("##### Log Level Distribution")
        fig_pie = create_log_level_pie_chart(log_df, theme)
        if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
        else: st.caption("No log level data available.")
    with chart_col2:
        st.markdown("##### Top Modules by Error Count")
        fig_bar = create_top_modules_error_bar_chart(log_df, theme)
        if fig_bar: st.plotly_chart(fig_bar, use_container_width=True)
        else: st.caption("No errors recorded or module data unavailable.")

    st.markdown("##### Log Timeline (by Hour)")
    fig_timeline = create_log_timeline_chart(log_df, theme)
    if fig_timeline: st.plotly_chart(fig_timeline, use_container_width=True)
    else: st.caption("Timestamp data ('datetime' column) required for timeline.")


# --- Log Explorer Rendering ---

def render_log_entry(log_row: pd.Series, index: int, page_num: int, llm_config: Dict):
    """Renders a single log entry with an explain button."""
    level = str(log_row.get("level", "INFO")).upper()
    style = LOG_LEVEL_STYLES.get(level, LOG_LEVEL_STYLES["DEFAULT"])
    level_color = style["color"]
    border_class = style["border"]

    # Use original DataFrame index if available, otherwise row index for unique key
    original_index = log_row.name if hasattr(log_row, 'name') else index
    button_key = f"explain_{original_index}_{page_num}_{index}"
    raw_log = log_row.get('raw', 'Log entry data missing.')

    log_col, btn_col = st.columns([10, 1])
    with log_col:
        st.markdown(f"""
        <div class="log-entry-container {border_class}" style="border-left-color: {level_color};">
            <div class="log-entry-header">
                <span class="log-timestamp">[{log_row.get('timestamp', 'No Timestamp')}]</span>
                <span>
                    <span class="log-level" style="background-color:{level_color};">{level}</span>
                    <span class="log-module">[{log_row.get('module', '?')}]</span>
                </span>
            </div>
            <div class="log-message">{log_row.get('message', 'No Message')}</div>
        </div>
        """, unsafe_allow_html=True)

    with btn_col:
         st.write("") # Spacer
         explain_clicked = st.button("Explain", key=button_key, help="Explain this log entry using AI", use_container_width=True)

    if explain_clicked:
        with st.spinner("🤖 Thinking..."):
            explanation = explain_single_log(raw_log, llm_config)
        st.markdown(f"""<div class="ai-response-box">{explanation}</div>""", unsafe_allow_html=True)
        st.divider()

def render_log_explorer(log_df: pd.DataFrame, llm_config: Dict):
    """Renders the Log Explorer tab with filtering, pagination, and entries."""
    st.markdown("### 🔍 Filter, Search & Explain Logs")
    st.caption("Explore individual log entries. Use filters to narrow down the results.")

    # --- Filtering UI ---
    with st.container(border=True):
        filt_col1, filt_col2, filt_col3 = st.columns([1, 1, 2])
        with filt_col1:
            levels = ["All"] + sorted(log_df["level"].astype(str).str.upper().unique()) if "level" in log_df.columns and log_df["level"].nunique() > 0 else ["All"]
            selected_level = st.selectbox("Filter by Level", levels, key="level_filter")
        with filt_col2:
            modules = ["All"] + sorted(log_df["module"].astype(str).unique()) if "module" in log_df.columns and log_df["module"].nunique() > 0 else ["All"]
            selected_module = st.selectbox("Filter by Module", modules, key="module_filter")
        with filt_col3:
            keyword = st.text_input("Search Raw Log (Keyword)", key="keyword_filter", placeholder="e.g., error, timeout, user_id...")

        with st.expander("Advanced Filters"):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            start_ts, end_ts = None, None # Initialize time range vars
            with adv_col1:
                valid_status = sorted([s for s in log_df["status_code"].astype(str).unique() if s and s != 'N/A']) if "status_code" in log_df.columns else []
                selected_status = st.selectbox("Status Code", ["All"] + valid_status, key="status_filter")
            with adv_col2:
                 valid_envs = sorted([e for e in log_df["env"].astype(str).unique() if e and e != 'unknown']) if "env" in log_df.columns else []
                 selected_env = st.selectbox("Environment", ["All"] + valid_envs, key="env_filter")
            with adv_col3:
                 if 'datetime' in log_df.columns and log_df['datetime'].notna().any():
                     valid_times = log_df['datetime'].dropna()
                     min_time, max_time = valid_times.min(), valid_times.max()
                     if pd.notna(min_time) and pd.notna(max_time) and min_time <= max_time:
                         default_start, default_end = min_time.to_pydatetime(), max_time.to_pydatetime()
                         max_slider_val = max_time + pd.Timedelta(seconds=1) if min_time == max_time else max_time
                         try:
                             selected_range = st.slider(
                                 "Filter by Time Range",
                                 min_value=min_time.to_pydatetime(), max_value=max_slider_val.to_pydatetime(),
                                 value=(default_start, default_end), format="YYYY-MM-DD HH:mm:ss", key="time_range_filter"
                             )
                             start_ts, end_ts = pd.Timestamp(selected_range[0]), pd.Timestamp(selected_range[1])
                         except Exception as slider_err: st.warning(f"Time range slider error: {slider_err}")
                     else: st.caption("Time filter disabled (invalid time data).")
                 else: st.caption("Time filter disabled (no timestamp data).")

    # --- Apply Filters ---
    filtered_df = log_df.copy()
    if selected_level != "All": filtered_df = filtered_df[filtered_df["level"].astype(str).str.upper() == selected_level]
    if selected_module != "All": filtered_df = filtered_df[filtered_df["module"] == selected_module]
    if selected_status != "All": filtered_df = filtered_df[filtered_df["status_code"] == selected_status]
    if selected_env != "All": filtered_df = filtered_df[filtered_df["env"] == selected_env]
    if keyword:
        try: filtered_df = filtered_df[filtered_df["raw"].astype(str).str.contains(re.escape(keyword), case=False, na=False, regex=True)]
        except re.error: filtered_df = filtered_df[filtered_df["raw"].astype(str).str.contains(keyword, case=False, na=False, regex=False)]
    if start_ts and end_ts and 'datetime' in filtered_df.columns and filtered_df['datetime'].notna().any():
         filtered_df = filtered_df[ (filtered_df['datetime'] >= start_ts) & (filtered_df['datetime'] <= end_ts) & (filtered_df['datetime'].notna()) ]

    st.markdown(f"#### 📝 Matching Log Entries ({len(filtered_df):,} found)")

    # --- Pagination ---
    total_rows = len(filtered_df)
    total_pages = max(1, (total_rows + LOG_EXPLORER_PAGE_SIZE - 1) // LOG_EXPLORER_PAGE_SIZE)

    filter_tuple = (selected_level, selected_module, keyword, selected_status, selected_env, start_ts, end_ts, st.session_state.get('current_data_source'))
    current_filter_hash = hash(filter_tuple)
    if current_filter_hash != st.session_state.get('log_explorer_filter_hash'):
         st.session_state['log_page'] = 1
         st.session_state['log_explorer_filter_hash'] = current_filter_hash

    page_number_state = st.session_state.get('log_page', 1)
    validated_page_number = min(max(1, page_number_state), total_pages)
    if validated_page_number != page_number_state: st.session_state['log_page'] = validated_page_number

    pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([1, 1, 3, 1])
    with pag_col1:
        if st.button("⬅️ Prev", disabled=(validated_page_number <= 1), use_container_width=True):
            st.session_state.log_page -= 1
            st.rerun()
    with pag_col2:
         if st.button("Next ➡️", disabled=(validated_page_number >= total_pages), use_container_width=True):
             st.session_state.log_page += 1
             st.rerun()
    with pag_col3:
        page_num_input = st.number_input('Page', min_value=1, max_value=total_pages, step=1, value=validated_page_number, key="log_page_num_input", label_visibility="collapsed")
        if page_num_input != validated_page_number:
            st.session_state.log_page = page_num_input
            st.rerun()
    with pag_col4:
        st.caption(f"Page {validated_page_number} of {total_pages}")

    # --- Display Logs ---
    start_idx = (validated_page_number - 1) * LOG_EXPLORER_PAGE_SIZE
    end_idx = min(start_idx + LOG_EXPLORER_PAGE_SIZE, total_rows)
    paginated_df = filtered_df.iloc[start_idx:end_idx]

    if not paginated_df.empty:
        for i, (_, row) in enumerate(paginated_df.iterrows()):
             render_log_entry(row, i, validated_page_number, llm_config)
    elif total_rows > 0:
         st.info("No logs match the current filter criteria on this page.", icon="😕")
    else:
         st.info("No log entries found matching your filters.", icon="🏝️")

# --- AI Analysis Tab Rendering ---

def render_ai_analysis_tab(log_df: pd.DataFrame, llm_config: Dict, theme: Optional[str]):
    """Renders the AI Analysis tab and its sub-tabs."""
    # ... (Analysis button logic remains the same) ...
    st.markdown("### 🧠 AI-Powered Log Analysis")
    st.caption("Cluster similar logs to find patterns and get AI-driven insights for debugging.")

    analysis_enabled = len(log_df) >= st.session_state.get('min_logs_for_cluster', 10) # Use config/state
    run_analysis_col, status_col = st.columns([1, 2])
    with run_analysis_col:
        run_analysis = st.button(
            "🔄 Run Clustering & Analysis", key="run_analysis_button",
            disabled=not analysis_enabled, help="Group logs and prepare developer-focused summaries.", use_container_width=True
        )
    with status_col:
        if not analysis_enabled: st.warning(f"Need ≥ {st.session_state.get('min_logs_for_cluster', 10)} logs for analysis (current: {len(log_df)}).", icon="⚠️")

    if run_analysis:
        n_clusters = st.session_state.get('n_clusters', 4) # Use config/state
        # Pass the log_df from session state as it might have been modified
        updated_df, summaries, profiles = run_log_clustering(st.session_state.log_df, n_clusters)
        if summaries is not None:
             st.session_state['log_df'] = updated_df
             st.session_state['clusters_summary'] = summaries
             st.session_state['error_profiles'] = profiles
             st.rerun()

    clusters_summary_data = st.session_state.get('clusters_summary')
    error_profiles_data = st.session_state.get('error_profiles', [])

    if clusters_summary_data is not None:
        tab_titles = ["Cluster Explorer"]
        if clusters_summary_data: tab_titles.append("Holistic Analysis")
        if error_profiles_data: tab_titles.append("Comparative Errors")
        if 'datetime' in log_df.columns and log_df['datetime'].notna().any():
            tab_titles.append("Temporal Patterns")

        analysis_tabs = st.tabs(tab_titles)
        analysis_tab_map = {title: tab for title, tab in zip(tab_titles, analysis_tabs)}

        # Render Sub-tabs
        with analysis_tab_map["Cluster Explorer"]:
            render_cluster_explorer_subtab(clusters_summary_data, llm_config) # Pass llm_config

        if "Holistic Analysis" in analysis_tab_map:
            with analysis_tab_map["Holistic Analysis"]:
                 # Pass serializable summary from state if available
                 log_summary_serializable = make_summary_serializable(st.session_state.get('log_df_summary', {}))
                 render_holistic_analysis_subtab(clusters_summary_data, log_summary_serializable, llm_config, theme)

        if "Comparative Errors" in analysis_tab_map:
            with analysis_tab_map["Comparative Errors"]:
                 render_comparative_errors_subtab(error_profiles_data, llm_config, theme)

        if "Temporal Patterns" in analysis_tab_map:
            with analysis_tab_map["Temporal Patterns"]:
                render_temporal_patterns_subtab(log_df, llm_config) # Pass log_df

    elif analysis_enabled:
        st.info("Click 'Run Clustering & Analysis' to start.", icon="💡")


def render_cluster_explorer_subtab(clusters_summary_data: List[Dict], llm_config: Dict):
    """Renders the Cluster Explorer sub-tab content with enhanced details."""
    st.markdown("##### Explore Log Clusters for Debugging")
    st.caption("Select a cluster to view detailed statistics, patterns, samples, and AI action plan.")

    valid_clusters = sorted([cs['cluster_id'] for cs in clusters_summary_data])
    if not valid_clusters: st.info("No valid clusters found."); return

    cluster_options = {cs['cluster_id']: f"Cluster {cs['cluster_id']} ({cs['total_logs']:,} logs, {cs['error_rate']:.1f}% err)" for cs in clusters_summary_data}
    selected_cluster_id = st.selectbox(
        "Select cluster:", options=valid_clusters,
        format_func=lambda x: cluster_options.get(x, f"Cluster {x}"),
        key="cluster_selector"
    )

    current_summary = next((cs for cs in clusters_summary_data if cs['cluster_id'] == selected_cluster_id), None)

    if current_summary:
        # --- Display Enhanced Summary Info ---
        st.markdown(f"**Cluster {selected_cluster_id} Details**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Logs", f"{current_summary['total_logs']:,}")
            st.metric("Error Rate", f"{current_summary.get('error_rate', 0):.1f}%", delta=f"{current_summary.get('error_count', 0)} errors")
            st.metric("Avg Latency", f"{current_summary.get('avg_latency_ms', 0):.1f} ms")
            st.markdown("**Top Modules:**")
            st.json(current_summary.get('top_modules', {}), expanded=False)
            common_patterns = current_summary.get('common_error_patterns', {})
            if common_patterns.get("top_keywords"):
                 st.markdown("**Common Keywords (Errors/Warn):**")
                 st.json(common_patterns["top_keywords"], expanded=False)


        with col2:
            st.metric("Time Span", current_summary.get('time_span', 'N/A'))
            st.metric("Warning Rate", f"{current_summary.get('warning_rate', 0):.1f}%", delta=f"{current_summary.get('warning_count', 0)} warnings")
            st.metric("P95 Latency", f"{current_summary.get('p95_latency_ms', 0):.1f} ms")
            st.markdown("**Top Status Codes:**")
            st.json(current_summary.get('top_status_codes', {}), expanded=False)
            if common_patterns.get("top_codes"):
                 st.markdown("**Common Codes (Errors/Warn):**")
                 st.json(common_patterns["top_codes"], expanded=False)


        # --- AI Action Plan ---
        st.markdown("---")
        st.markdown("**AI Debugging Action Plan**")
        interpret_button = st.button(f"🤖 Generate Action Plan for Cluster {selected_cluster_id}", key=f"interpret_{selected_cluster_id}", use_container_width=True)

        # Add button to filter Log Explorer
        filter_button_col, _ = st.columns([1,3]) # Smaller button column
        with filter_button_col:
             if st.button(f"🔍 Explore Cluster {selected_cluster_id} Logs", key=f"explore_{selected_cluster_id}", help="Go to Log Explorer filtered for this cluster"):
                  # Store filter criteria in session state before rerun
                  st.session_state['explorer_prefilter_cluster'] = selected_cluster_id
                  # Optionally pre-filter other things?
                  st.session_state['selected_tab_key'] = "🔍 Log Explorer" # Request tab switch
                  st.rerun() # Rerun will switch tab and apply filter


        if interpret_button:
            # Ensure summary is serializable before sending to LLM
            serializable_summary = make_summary_serializable(current_summary)
            try:
                summary_str = json.dumps(serializable_summary, indent=2)
                interpretation = get_ai_cluster_interpretation(serializable_summary, llm_config) # Pass serializable dict
                st.markdown(f"""<div class="ai-response-box">{interpretation}</div>""", unsafe_allow_html=True)
            except TypeError as e:
                st.error(f"Failed to serialize cluster summary for AI: {e}")
                print("Problematic Summary:", serializable_summary)


        # --- Sample Logs ---
        st.markdown("---")
        st.markdown(f"**Sample Raw Logs from Cluster {selected_cluster_id}**")
        samples = current_summary.get('sample_logs', [])
        if samples:
            for sample in samples: st.code(sample, language='log')
        else: st.caption("No sample logs available.")

    else: st.warning(f"Could not load summary for Cluster {selected_cluster_id}.")

# Modified to accept serializable summary
def render_holistic_analysis_subtab(clusters_summary_data: List[Dict], log_summary_serializable: Dict, llm_config: Dict, theme: Optional[str]):
    """Renders the Holistic Analysis sub-tab content."""
    st.markdown("##### 🌍 Developer System Health Assessment")
    st.caption("Overall insights focusing on critical issues and investigation pointers.")

    holistic_chart_col1, holistic_chart_col2 = st.columns(2)
    with holistic_chart_col1:
        fig_dist = create_cluster_distribution_chart(clusters_summary_data, theme)
        if fig_dist: st.plotly_chart(fig_dist, use_container_width=True)
    with holistic_chart_col2:
        fig_err_rate = create_cluster_error_rate_chart(clusters_summary_data, theme)
        if fig_err_rate: st.plotly_chart(fig_err_rate, use_container_width=True)

    st.markdown("**AI Holistic Analysis & Investigation Plan**")
    holistic_button = st.button("🤖 Generate Holistic Analysis", key="holistic_analysis", use_container_width=True)
    if holistic_button:
        # Ensure clusters_summary_data is serializable (should be from run_log_clustering)
        serializable_clusters = [make_summary_serializable(cs) for cs in clusters_summary_data]

        try:
            summary_str = json.dumps(log_summary_serializable, indent=2) # Already serialized
            clusters_str = json.dumps(serializable_clusters, indent=2)
            with st.spinner("🧠 Generating system-wide analysis..."):
                holistic_text = perform_holistic_analysis(summary_str, clusters_str, llm_config)
            st.markdown(f"""<div class="ai-response-box">{holistic_text}</div>""", unsafe_allow_html=True)
        except TypeError as e:
            st.error(f"Failed to serialize data for holistic analysis: {e}")
            print("Problematic dashboard summary:", log_summary_serializable)
            print("Problematic cluster summaries:", serializable_clusters)


def render_comparative_errors_subtab(error_profiles_data: List[Dict], llm_config: Dict, theme: Optional[str]):
    """Renders the Comparative Errors sub-tab content."""
    st.markdown("##### ↔️ Comparative Error Analysis for Debugging")
    st.caption("Compares error signatures across clusters to identify distinct failure modes.")

    fig_comp_err = create_comparative_error_count_chart(error_profiles_data, theme)
    if fig_comp_err: st.plotly_chart(fig_comp_err, use_container_width=True)
    else: st.caption("No error clusters to compare.")

    st.markdown("**Error Profile Details by Cluster**")
    if error_profiles_data:
        for profile in error_profiles_data:
            # Display enhanced profile info
            expander_title = f"Cluster {profile['cluster_id']} ({profile['error_count']} errors, P95 Lat: {profile.get('p95_error_latency_ms','N/A')}ms)"
            with st.expander(expander_title):
                st.markdown("**Top Error Modules:**")
                st.json(profile.get('error_modules', {}), expanded=False)
                st.markdown("**Top Error Status Codes:**")
                st.json(profile.get('error_status_codes', {}), expanded=False)
                st.markdown("**Common Error Keywords:**")
                st.json(profile.get('common_error_keywords', {}), expanded=False)
                st.markdown("**Common Error Codes:**")
                st.json(profile.get('common_error_codes', {}), expanded=False)
                st.markdown("**Sample Errors:**")
                for err_sample in profile.get('sample_errors',[]):
                    st.code(err_sample, language='log')
    else: st.caption("No error profiles generated.")

    st.markdown("**AI Comparative Analysis & Hypotheses**")
    comp_button = st.button("🤖 Generate Comparative Analysis", key="comparative_analysis", use_container_width=True, disabled=not error_profiles_data)
    if comp_button and error_profiles_data:
        serializable_profiles = [make_summary_serializable(p) for p in error_profiles_data]
        try:
            profiles_str = json.dumps(serializable_profiles, indent=2)
            with st.spinner("🧠 Comparing error patterns..."):
                comp_text = perform_comparative_analysis(profiles_str, llm_config)
            st.markdown(f"""<div class="ai-response-box">{comp_text}</div>""", unsafe_allow_html=True)
        except TypeError as e:
            st.error(f"Failed to serialize error profiles: {e}")
            print("Problematic Error Profiles:", serializable_profiles)



def render_temporal_patterns_subtab(log_df: pd.DataFrame, llm_config: Dict):
    """Renders the Temporal Patterns sub-tab content."""
    st.markdown("##### ⏰ Temporal Pattern Analysis")
    st.caption("Examines log volume and errors change over time (by hour).")

    time_df_temporal = log_df.dropna(subset=['datetime']).copy()
    if not time_df_temporal.empty and pd.api.types.is_datetime64_any_dtype(time_df_temporal['datetime']):
         if 'level' not in time_df_temporal.columns: time_df_temporal['level'] = 'INFO'
         time_df_temporal['level'] = time_df_temporal['level'].astype(str).str.upper()
         # Handle TZ if present
         if time_df_temporal['datetime'].dt.tz is not None:
             time_df_temporal['hour_of_day'] = time_df_temporal['datetime'].dt.tz_convert('UTC').dt.hour
         else:
             time_df_temporal['hour_of_day'] = time_df_temporal['datetime'].dt.hour

         if pd.api.types.is_numeric_dtype(time_df_temporal['hour_of_day']):
            try:
                hourly_agg = time_df_temporal.groupby('hour_of_day').agg(
                    total_logs=('raw', 'count'),
                    error_count=('level', lambda x: (x == 'ERROR').sum()),
                    avg_latency=('latency', lambda x: pd.to_numeric(x, errors='coerce').fillna(0).mean())
                ).reset_index()
                hourly_agg['error_rate'] = (hourly_agg['error_count'] / hourly_agg['total_logs'].replace(0, np.nan) * 100).fillna(0).round(1)

                if not hourly_agg.empty:
                    theme = st.session_state.get('chart_theme')
                    temp_chart_col1, temp_chart_col2 = st.columns(2)
                    with temp_chart_col1:
                        # Simple bar chart for volume by hour
                        fig_vol = px.bar(hourly_agg, x='hour_of_day', y='total_logs', title='Volume by Hour', template=theme)
                        fig_vol.update_layout(xaxis=dict(tickmode='linear', dtick=2), height=300, margin=dict(t=30,b=0,l=0,r=0), xaxis_title="Hour")
                        st.plotly_chart(fig_vol, use_container_width=True)
                    with temp_chart_col2:
                        # Simple line for error rate
                        fig_err = px.line(hourly_agg, x='hour_of_day', y='error_rate', title='Error Rate (%) by Hour', template=theme, markers=True, range_y=[0,100])
                        fig_err.update_traces(line=dict(color=COLORS["error"]))
                        fig_err.update_layout(xaxis=dict(tickmode='linear', dtick=2), height=300, margin=dict(t=30,b=0,l=0,r=0), xaxis_title="Hour")
                        st.plotly_chart(fig_err, use_container_width=True)

                    st.markdown("**AI Temporal Analysis**")
                    temporal_button = st.button("🤖 Generate Temporal Analysis", key="temporal_analysis", use_container_width=True)
                    if temporal_button:
                        # Ensure hourly_agg dtypes are JSON serializable before converting to string
                        serializable_hourly_agg = hourly_agg.copy()
                        for col in serializable_hourly_agg.select_dtypes(include=np.number).columns:
                            # Check specifically for numpy types if needed, but apply broadly
                             serializable_hourly_agg[col] = serializable_hourly_agg[col].apply(lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x)

                        stats_str = serializable_hourly_agg[['hour_of_day', 'total_logs', 'error_count', 'error_rate', 'avg_latency']].round(1).to_string(index=False)
                        mean_err = serializable_hourly_agg['error_rate'].mean() # Use serializable df for calculation too
                        anom_hours_df = serializable_hourly_agg[serializable_hourly_agg['error_rate'] > max(mean_err * 1.5, 10.0)]
                        anom_str = "Anomalous Hours (High Error Rate):\n" + anom_hours_df[['hour_of_day', 'error_rate']].round(1).to_string(index=False) if not anom_hours_df.empty else "No hours with significantly high error rates."
                        with st.spinner("🧠 Analyzing temporal patterns..."):
                            temp_text = perform_temporal_analysis(stats_str, anom_str, llm_config)
                        st.markdown(f"""<div class="ai-response-box">{temp_text}</div>""", unsafe_allow_html=True)
                else: st.warning("Could not generate hourly aggregates.", icon="⚠️")
            except Exception as e: st.error(f"Temporal aggregation error: {e}", icon="❌")
         else: st.warning("Could not extract valid 'hour_of_day'.", icon="⚠️")
    else: st.info("Valid timestamp data required for temporal analysis.", icon="ℹ️")


# --- Advanced Viz Tab Rendering ---

def render_advanced_viz_tab(log_df: pd.DataFrame, theme: Optional[str]):
    """Renders the Advanced Visualizations tab."""
    st.markdown("### 📈 Advanced Log Data Visualizations")
    st.caption("Deeper dives into latency, status codes, and module interactions.")

    if log_df.empty:
        st.info("Load log data first.", icon="📊")
        return

    # Determine available visualizations
    viz_options = []
    # Check for numeric latency > 0
    if "latency" in log_df.columns and pd.to_numeric(log_df["latency"], errors='coerce').gt(0).any():
        viz_options.append("Latency Deep Dive")
    # Check for 3-digit status codes
    if "status_code" in log_df.columns and log_df["status_code"].astype(str).str.match(r'^\d{3}$').any():
        viz_options.append("HTTP Status Code Patterns")
    # Check for multiple modules
    if "module" in log_df.columns and log_df["module"].nunique() > 1:
        viz_options.append("Module Activity & Errors")
    # Check for valid datetime data
    if 'datetime' in log_df.columns and log_df['datetime'].notna().any():
        viz_options.append("Detailed Time Series Analysis")

    if not viz_options:
        st.warning("Insufficient data for advanced visualizations.", icon="⚠️")
        return

    viz_select_col, viz_display_col = st.columns([1, 3])
    with viz_select_col:
         viz_type = st.radio("Select Visualization:", viz_options, key="viz_type_selector")

    with viz_display_col:
        if viz_type == "Latency Deep Dive":
            st.markdown("#### Latency Distribution & Outliers")
            latency_df_viz = log_df.copy()
            latency_df_viz['latency'] = pd.to_numeric(latency_df_viz["latency"], errors='coerce')
            latency_df_viz = latency_df_viz.dropna(subset=['latency'])
            latency_df_viz = latency_df_viz[latency_df_viz['latency'] > 0]
            if not latency_df_viz.empty:
                lat_viz_c1, lat_viz_c2 = st.columns(2)
                with lat_viz_c1:
                    fig_hist = create_latency_histogram(latency_df_viz, theme)
                    if fig_hist: st.plotly_chart(fig_hist, use_container_width=True)
                with lat_viz_c2:
                    fig_box = create_latency_boxplot_by_module(latency_df_viz, theme)
                    if fig_box: st.plotly_chart(fig_box, use_container_width=True)
                    else: st.caption("Module info needed for box plot.")
            else: st.info("No positive latency data found.")

        elif viz_type == "HTTP Status Code Patterns":
            st.markdown("#### HTTP Status Code Analysis")
            status_df_viz = log_df.copy()
            status_df_viz["status_code_str"] = status_df_viz["status_code"].astype(str).str.strip()
            status_df_viz = status_df_viz[status_df_viz["status_code_str"].str.match(r'^\d{3}$')].copy()
            if not status_df_viz.empty:
                status_df_viz["status_code"] = status_df_viz["status_code_str"].astype(int)
                status_df_viz['status_category'] = status_df_viz['status_code'].apply(
                    lambda x: 'Success (2xx)' if 200 <= x < 300 else
                              'Redirect (3xx)' if 300 <= x < 400 else
                              'Client Error (4xx)' if 400 <= x < 500 else
                              'Server Error (5xx)' if 500 <= x < 600 else 'Other')
                stat_viz_c1, stat_viz_c2 = st.columns(2)
                with stat_viz_c1:
                    fig_bar = create_status_code_bar_chart(status_df_viz, theme)
                    if fig_bar: st.plotly_chart(fig_bar, use_container_width=True)
                with stat_viz_c2:
                     fig_pie = create_status_code_pie_chart(status_df_viz, theme)
                     if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
                fig_heatmap = create_status_module_heatmap(status_df_viz, theme)
                if fig_heatmap:
                    st.markdown("###### Status Code Categories per Module")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            else: st.info("No valid 3-digit status codes found.")

        elif viz_type == "Module Activity & Errors":
             st.markdown("#### Module Log Activity & Error Distribution")
             if 'module' in log_df.columns and log_df['module'].nunique() > 0:
                 mod_viz_c1, mod_viz_c2 = st.columns(2)
                 with mod_viz_c1:
                     fig_mod_bar = create_module_volume_bar_chart(log_df, theme)
                     if fig_mod_bar: st.plotly_chart(fig_mod_bar, use_container_width=True)
                 with mod_viz_c2:
                      fig_treemap = create_module_level_treemap(log_df, theme)
                      if fig_treemap: st.plotly_chart(fig_treemap, use_container_width=True)
                      else: st.info("Log 'level' column needed for Treemap.")
             else: st.info("Missing 'module' data or only one module.")

        elif viz_type == "Detailed Time Series Analysis":
            st.markdown("#### Log Activity Over Time")
            time_df_viz = log_df.dropna(subset=['datetime']).copy()
            if not time_df_viz.empty and pd.api.types.is_datetime64_any_dtype(time_df_viz['datetime']):
                time_df_viz = time_df_viz.sort_values('datetime')
                time_span_hours = (time_df_viz['datetime'].max() - time_df_viz['datetime'].min()).total_seconds() / 3600

                resample_options = {'Auto': None, 'Second': 's', '10 Seconds': '10s', 'Minute': 'min', '5 Minutes': '5min', '15 Minutes': '15min', 'Hour': 'h', 'Day': 'D'}
                # Filter options based on span
                valid_resample_labels = ['Auto']
                if time_span_hours > 0.001: valid_resample_labels.append('Second')
                if time_span_hours > 0.003: valid_resample_labels.append('10 Seconds')
                if time_span_hours > 0.02: valid_resample_labels.append('Minute')
                if time_span_hours > 0.2: valid_resample_labels.append('5 Minutes')
                if time_span_hours > 1: valid_resample_labels.append('15 Minutes')
                if time_span_hours > 4: valid_resample_labels.append('Hour')
                if time_span_hours > 24: valid_resample_labels.append('Day')

                selected_freq_label = st.selectbox("Time Aggregation:", valid_resample_labels, index=0, key="time_agg_select")
                resample_freq_code = resample_options[selected_freq_label]
                if resample_freq_code is None: # Auto logic
                    if time_span_hours <= 0.5: resample_freq_code = '10s'
                    elif time_span_hours <= 2: resample_freq_code = 'min'
                    elif time_span_hours <= 12: resample_freq_code = '5min'
                    elif time_span_hours <= 72: resample_freq_code = 'h'
                    else: resample_freq_code = 'D'
                    st.caption(f"Using auto-selected aggregation: {resample_freq_code}")

                try:
                    if 'level' not in time_df_viz.columns: time_df_viz['level'] = 'INFO'
                    time_df_viz['level'] = time_df_viz['level'].astype(str).str.upper()
                    time_df_indexed = time_df_viz.set_index('datetime')
                    if time_df_indexed.index.tz is not None: time_df_indexed = time_df_indexed.tz_convert('UTC')

                    time_agg = time_df_indexed.resample(resample_freq_code).agg(
                        total_logs=('raw', 'count'),
                        error_count=('level', lambda x: (x == 'ERROR').sum())
                    ).reset_index()
                    time_agg['error_rate'] = (time_agg['error_count'] / time_agg['total_logs'].replace(0, np.nan) * 100).fillna(0).round(1)

                    fig_time_comb = create_detailed_timeseries_chart(time_agg, resample_freq_code, theme)
                    if fig_time_comb: st.plotly_chart(fig_time_comb, use_container_width=True)
                    else: st.info("No data after time aggregation.")

                    if time_span_hours > 24: # Show day/hour patterns only if span > 1 day
                         st.markdown("###### Activity Patterns by Day/Hour")
                         time_pat_c1, time_pat_c2 = st.columns(2)
                         with time_pat_c1:
                             fig_dayhour = create_day_hour_heatmap(time_df_viz, theme)
                             if fig_dayhour: st.plotly_chart(fig_dayhour, use_container_width=True)
                         with time_pat_c2:
                             fig_errday = create_errors_by_day_chart(time_df_viz, theme)
                             if fig_errday: st.plotly_chart(fig_errday, use_container_width=True)

                except Exception as e:
                    st.error(f"Time aggregation/plotting error: {e}", icon="❌")
            else: st.info("Valid timestamp data required.")