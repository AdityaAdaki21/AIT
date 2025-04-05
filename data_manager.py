# data_manager.py
import streamlit as st
import pandas as pd
import requests
import queue
import threading
import time
import random
import traceback
from datetime import datetime
from typing import Optional

# Assuming other modules are in the same directory
from parsing import extract_components
from config import REQUIRED_COLUMNS, SAMPLE_LOG_CSV_FILENAME, SAMPLE_LOG_COUNT
from utils import get_stable_hash

# Import backend safely
try:
    from logs_backend import fetch_logs as fetch_backend_logs
except ImportError:
    st.warning("Could not import `logs_backend.py`. 'Sample Logs' feature may be limited or fail.")
    def fetch_backend_logs():
        st.error("`logs_backend.py` not found or import failed. Cannot load sample logs.")
        return []


# --- File Loading ---

# Cache the parsing of uploaded files based on file info
@st.cache_data(show_spinner="Parsing uploaded log file...")
def _parse_uploaded_file_cached(file_name: str, file_size: int, file_type: str, content_bytes: bytes) -> pd.DataFrame:
    """Cached function to parse file content bytes."""
    try:
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin-1 (ISO-8859-1).")
            try:
                content = content_bytes.decode("latin-1")
            except UnicodeDecodeError:
                st.error("Failed to decode file. Please ensure it's UTF-8 or Latin-1 encoded.")
                return pd.DataFrame(columns=REQUIRED_COLUMNS) # Return empty on decode failure

        if content:
            logs_list = content.splitlines()
            # Pass tuple for caching, include filename hint
            log_df_temp = extract_components(tuple(logs_list), filename=file_name)
            return log_df_temp
        else:
            st.warning("Uploaded file appears to be empty after decoding.")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

    except Exception as e:
        st.error(f"Error reading or processing uploaded file content: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=REQUIRED_COLUMNS)


def load_uploaded_file(uploaded_file):
    """Handles uploaded file, calls cached parsing, updates session state."""
    if uploaded_file is None:
        return False # No file

    current_file_info = (uploaded_file.name, uploaded_file.size, uploaded_file.type)
    last_file_info = st.session_state.get('last_uploaded_file_info')

    # Process only if it's a new file
    if current_file_info != last_file_info:
        print(f"Processing new uploaded file: {uploaded_file.name} ({uploaded_file.size} bytes)")
        content_bytes = uploaded_file.read() # Read bytes once
        parsed_df = _parse_uploaded_file_cached(
            uploaded_file.name, uploaded_file.size, uploaded_file.type, content_bytes
        )

        if not parsed_df.empty:
            st.session_state['log_df'] = parsed_df
            st.session_state['last_uploaded_file_info'] = current_file_info
            st.success(f"✅ Loaded and parsed {len(parsed_df)} entries from {uploaded_file.name}")
            # Clear potentially incompatible analysis results from previous data
            st.session_state['clusters_summary'] = None
            st.session_state['error_profiles'] = None
            st.session_state['log_df_summary'] = None
            st.session_state['log_df_summary_cache_key'] = None
            if 'cluster' in st.session_state.log_df.columns:
                st.session_state.log_df = st.session_state.log_df.drop(columns=['cluster'], errors='ignore')
            return True # Indicates data was loaded
        else:
            # Parsing failed or empty file, clear the last info state
            st.session_state['last_uploaded_file_info'] = None
            st.session_state['log_df'] = pd.DataFrame(columns=REQUIRED_COLUMNS) # Clear potentially old data
            return False # No data loaded

    # File is the same as last time, no action needed unless df is empty
    elif uploaded_file is not None and st.session_state.get('log_df', pd.DataFrame()).empty:
         # Re-process if the same file is present but dataframe is empty (e.g., after switching sources)
         print(f"Re-processing previously uploaded file: {uploaded_file.name}")
         content_bytes = uploaded_file.getvalue() # Use getvalue if already read once by Streamlit
         parsed_df = _parse_uploaded_file_cached(
            uploaded_file.name, uploaded_file.size, uploaded_file.type, content_bytes
         )
         if not parsed_df.empty:
            st.session_state['log_df'] = parsed_df
            st.session_state['last_uploaded_file_info'] = current_file_info # Update state
            # Clear analysis results
            st.session_state['clusters_summary'] = None
            st.session_state['error_profiles'] = None
            st.session_state['log_df_summary'] = None
            st.session_state['log_df_summary_cache_key'] = None
            if 'cluster' in st.session_state.log_df.columns:
                st.session_state.log_df = st.session_state.log_df.drop(columns=['cluster'], errors='ignore')
            return True
         else:
            st.session_state['last_uploaded_file_info'] = None
            st.session_state['log_df'] = pd.DataFrame(columns=REQUIRED_COLUMNS)
            return False
    return False # No new file processed


# --- Sample Log Loading ---

# Cache the processing of sample logs
@st.cache_data(show_spinner="Loading and parsing sample logs...")
def _load_and_parse_sample_logs() -> pd.DataFrame:
    """Fetches and parses sample logs (cached)."""
    try:
        logs = fetch_backend_logs() # From logs_backend.py
        if logs:
            # Infer filename hint for parsing
            file_name_hint = f"{SAMPLE_LOG_CSV_FILENAME}" if isinstance(logs, list) and len(logs) > 1 and ',' in logs[0] else "sample_logs.log"
            if isinstance(logs, list) and len(logs) > 1 and str(logs[0]).strip().startswith('{'):
                file_name_hint = "sample_logs.jsonl"

            log_df_temp = extract_components(tuple(logs), filename=file_name_hint)
            if not log_df_temp.empty:
                return log_df_temp
            else:
                st.error("Failed to parse sample logs.")
                return pd.DataFrame(columns=REQUIRED_COLUMNS)
        else:
             st.warning(f"No sample logs were loaded. Check `{SAMPLE_LOG_CSV_FILENAME}` or `logs_backend.py`.")
             return pd.DataFrame(columns=REQUIRED_COLUMNS)
    except Exception as e:
        st.error(f"Error loading sample logs: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

def load_sample_logs():
    """Loads sample logs using cached function and updates session state."""
    if not st.session_state.get('sample_logs_loaded', False):
        parsed_df = _load_and_parse_sample_logs()
        if not parsed_df.empty:
            st.session_state['log_df'] = parsed_df
            st.session_state['sample_logs_loaded'] = True
            st.success(f"✅ Loaded and parsed {len(parsed_df)} sample log entries.")
             # Clear analysis results from previous data
            st.session_state['clusters_summary'] = None
            st.session_state['error_profiles'] = None
            st.session_state['log_df_summary'] = None
            st.session_state['log_df_summary_cache_key'] = None
            if 'cluster' in st.session_state.log_df.columns:
                st.session_state.log_df = st.session_state.log_df.drop(columns=['cluster'], errors='ignore')
            return True # Data loaded
        else:
            st.session_state['sample_logs_loaded'] = True # Mark as attempted even if failed
            st.session_state['log_df'] = pd.DataFrame(columns=REQUIRED_COLUMNS) # Clear df
            return False # Data not loaded
    # Already loaded
    return False # No *new* data loaded this time

# --- SSE Client ---

def sse_client_thread(url: str, log_q: queue.Queue, stop_event: threading.Event):
    """Connects to SSE endpoint, puts raw log strings into the queue."""
    headers = {'Accept': 'text/event-stream'}
    retry_delay = 1
    max_retry_delay = 30
    session_id = random.randint(1000, 9999)
    thread_name = f"SSE-{session_id}"
    print(f"[{datetime.now()}] [{thread_name}] Thread: Starting...")
    st.session_state['sse_last_error'] = None # Use st.session_state for cross-thread communication

    while not stop_event.is_set():
        try:
            st.session_state['sse_connection_status'] = "connecting"
            print(f"[{datetime.now()}] [{thread_name}] Thread: Connecting to {url}...")
            with requests.Session() as session:
                response = session.get(url, stream=True, headers=headers, timeout=(10, 60)) # (connect, read)
                response.raise_for_status() # Check for 4xx/5xx errors immediately

            print(f"[{datetime.now()}] [{thread_name}] Thread: Connection successful (Status: {response.status_code}).")
            st.session_state['sse_connection_status'] = "connected"
            st.session_state['sse_last_error'] = None
            retry_delay = 1 # Reset retry delay

            for line in response.iter_lines():
                if stop_event.is_set():
                    print(f"[{datetime.now()}] [{thread_name}] Thread: Stop event detected during line iteration.")
                    break
                if line:
                    try:
                        decoded_line = line.decode('utf-8', errors='replace')
                        if decoded_line.startswith('data:'):
                            log_data = decoded_line[len('data:'):].strip()
                            if log_data: log_q.put(log_data)
                    except UnicodeDecodeError as ude:
                         print(f"[{datetime.now()}] [{thread_name}] Thread: Unicode decode error: {ude}. Line: {line[:100]}...")
                         log_q.put(f'{{"timestamp": "{datetime.now().isoformat()}", "level": "PARSE_ERROR", "module": "sse_client", "message": "Unicode decode error receiving log"}}')
                    except Exception as line_proc_e:
                         print(f"[{datetime.now()}] [{thread_name}] Thread: Error processing line: {line_proc_e}. Line: {line[:100]}...")
                         log_q.put(f'{{"timestamp": "{datetime.now().isoformat()}", "level": "PARSE_ERROR", "module": "sse_client", "message": "Error processing received line: {line_proc_e}"}}')

            if not stop_event.is_set(): # Stream ended from server side
                print(f"[{datetime.now()}] [{thread_name}] Thread: Stream ended by server.")
                st.session_state['sse_connection_status'] = "disconnected"
                st.session_state['sse_last_error'] = "Stream ended by server."
                break # Exit loop, requires manual reconnect

        except requests.exceptions.ConnectionError as e: error_msg = f"Connection Error: {e}."
        except requests.exceptions.Timeout as e: error_msg = f"Connection Timeout: {e}."
        except requests.exceptions.HTTPError as e:
             error_msg = f"HTTP Error: {e.response.status_code} {e.response.reason}."
             if e.response.status_code in [404, 401, 403]:
                 print(f"[{datetime.now()}] [{thread_name}] Thread: Unrecoverable HTTP error {e.response.status_code}. Stopping.")
                 stop_event.set()
        except requests.exceptions.RequestException as e: error_msg = f"Network Request Exception: {type(e).__name__} - {e}"
        except Exception as e:
            error_msg = f"Unexpected Error in SSE thread: {type(e).__name__} - {e}"
            traceback.print_exc()

        if not stop_event.is_set():
            print(f"[{datetime.now()}] [{thread_name}] Thread: {error_msg}. Retrying in {retry_delay:.1f}s...")
            st.session_state['sse_connection_status'] = "error"
            st.session_state['sse_last_error'] = error_msg
            wait_time = retry_delay + random.uniform(0, 1)
            start_wait = time.time()
            while time.time() - start_wait < wait_time:
                 if stop_event.wait(timeout=0.5): break
            retry_delay = min(retry_delay * 1.5, max_retry_delay)

        if stop_event.is_set():
             print(f"[{datetime.now()}] [{thread_name}] Thread: Stop event detected after error/retry wait.")
             break

    final_status = "disconnected" if stop_event.is_set() or st.session_state.get('sse_last_error') == "Stream ended by server." else "error"
    st.session_state['sse_connection_status'] = final_status
    # Clean up thread references in session state only if this thread is the one registered
    current_thread_id = threading.get_ident()
    sse_thread_obj = st.session_state.get('sse_thread')
    if sse_thread_obj and sse_thread_obj.ident == current_thread_id:
        st.session_state['sse_thread'] = None
        st.session_state['sse_stop_event'] = None
    print(f"[{datetime.now()}] [{thread_name}] Thread: Stopped. Final Status: {final_status}")

def start_sse_thread(url: str):
    """Starts the SSE client thread."""
    if st.session_state.get('sse_thread') is not None:
        print("SSE thread already running.")
        return

    # Clear previous logs for a fresh stream view
    st.session_state['log_df'] = pd.DataFrame(columns=REQUIRED_COLUMNS)
    st.session_state['log_queue'] = queue.Queue()
    st.session_state['sse_last_error'] = None
    st.session_state['clusters_summary'] = None # Clear analysis on new connect
    st.session_state['error_profiles'] = None
    st.session_state['log_df_summary'] = None
    st.session_state['log_df_summary_cache_key'] = None
    if 'cluster' in st.session_state.get('log_df', pd.DataFrame()).columns:
        st.session_state.log_df = st.session_state.log_df.drop(columns=['cluster'], errors='ignore')


    stop_event = threading.Event()
    st.session_state['sse_stop_event'] = stop_event
    thread = threading.Thread(
        target=sse_client_thread,
        args=(url, st.session_state['log_queue'], stop_event),
        daemon=True
    )
    st.session_state['sse_thread'] = thread
    st.session_state['sse_connection_status'] = "connecting" # Set status *before* starting
    thread.start()
    print("SSE client thread started.")
    time.sleep(0.1) # Allow thread to initialize

def stop_sse_thread():
    """Signals the SSE client thread to stop."""
    stop_event = st.session_state.get('sse_stop_event')
    if stop_event:
        print("Signaling SSE thread to stop.")
        stop_event.set()
        # State updates (like status='disconnected') happen within the thread itself upon stopping
    else:
        print("No active SSE stop event found.")
    # Reset status immediately for UI responsiveness, thread will confirm later
    st.session_state['sse_connection_status'] = "disconnected"
    st.session_state['sse_last_error'] = None
    # Thread object is cleared by the thread itself upon exit


# --- Log Queue Processing ---

# Cache the parsing of log batches from the queue
# Note: Caching this might be tricky if logs have identical content but different timestamps.
# A simple approach is to hash the tuple of log lines.
@st.cache_data
def _parse_log_batch_cached(log_lines_tuple: tuple) -> pd.DataFrame:
    """Cached parsing for batches of logs from the queue."""
    # filename=None as source is queue, not file
    return extract_components(log_lines_tuple, filename=None)

def process_log_queue(max_logs_to_keep: int) -> int:
    """Checks queue, parses logs, updates DataFrame, enforces limits."""
    logs_processed_this_run = 0
    new_log_lines = []
    log_queue_obj = st.session_state.get('log_queue')

    if log_queue_obj and not log_queue_obj.empty():
        start_time = time.time()
        current_queue_size = log_queue_obj.qsize()
        batch_size = min(current_queue_size, 150) # Process up to 150 at once

        if batch_size > 0:
            try:
                for _ in range(batch_size):
                    if time.time() - start_time > 0.2: # Limit time spent pulling from queue
                        break
                    new_log_lines.append(log_queue_obj.get_nowait())
                logs_processed_this_run = len(new_log_lines)
            except queue.Empty:
                logs_processed_this_run = len(new_log_lines)
            except Exception as q_err:
                print(f"Error reading from queue: {q_err}")
                logs_processed_this_run = len(new_log_lines)

        if new_log_lines:
            # Use the cached parsing function for the batch
            new_df = _parse_log_batch_cached(tuple(new_log_lines))

            if not new_df.empty:
                current_df = st.session_state.get('log_df', pd.DataFrame(columns=REQUIRED_COLUMNS))
                try:
                    # --- Concatenation and Limit Enforcement ---
                    if current_df.empty:
                        combined_df = new_df
                    else:
                        # Ensure dtypes match before concat (especially datetime)
                        for col in ['datetime']: # Add other sensitive columns if needed
                            if col in current_df.columns and col in new_df.columns:
                                current_df[col] = pd.to_datetime(current_df[col], errors='coerce')
                                new_df[col] = pd.to_datetime(new_df[col], errors='coerce')
                        combined_df = pd.concat([current_df, new_df], ignore_index=True, sort=False)

                    # Enforce max logs limit
                    if len(combined_df) > max_logs_to_keep:
                        combined_df = combined_df.iloc[-max_logs_to_keep:]

                    # Ensure datetime column type consistency after concat/slice
                    if 'datetime' in combined_df.columns:
                         combined_df['datetime'] = pd.to_datetime(combined_df['datetime'], errors='coerce')

                    # Update session state (reset index for clean display)
                    st.session_state['log_df'] = combined_df.reset_index(drop=True)
                    return logs_processed_this_run
                except Exception as concat_err:
                     print(f"Error during DataFrame concat/update: {concat_err}")
                     traceback.print_exc()
                     return 0 # Indicate 0 processed successfully on major error
            else:
                 print("Warning: extract_components returned empty DataFrame for new logs batch.")
                 return 0
    return 0 # No logs processed
