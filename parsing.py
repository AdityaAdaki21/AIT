# parsing.py
import pandas as pd
import json
import re
import io
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# Assuming utils.py and config.py are in the same directory
from utils import parse_timestamp, is_csv_file
from config import REQUIRED_COLUMNS

def _parse_jsonl(logs: List[str]) -> Tuple[List[Dict], bool]:
    """Parses logs assuming JSON Lines format."""
    data = []
    json_detected = False
    first_valid_line = next((line.strip() for line in logs if isinstance(line, str) and line.strip()), None)

    if not first_valid_line or not (first_valid_line.startswith('{') and first_valid_line.endswith('}')):
        return [], False # Not JSONL

    json_detected = True
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
            module = str(entry.get('components', entry.get('module', entry.get('service', 'unknown'))))
            status_code = str(entry.get('status_code', entry.get('status', 'N/A')))
            env = str(entry.get('environment', entry.get('env', 'unknown')))

            latency_val = entry.get('latency_ms', entry.get('latency', entry.get('response_time', None)))
            latency_ms = 0.0
            if latency_val is not None:
                latency_num = pd.to_numeric(latency_val, errors='coerce')
                if pd.notna(latency_num):
                    is_seconds = 'response_time' in entry and ('latency_ms' not in entry and 'latency' not in entry) and latency_val == entry.get('response_time')
                    if is_seconds and abs(latency_num) < 50: # Heuristic threshold for seconds
                        latency_ms = latency_num * 1000
                    else:
                        latency_ms = latency_num
            latency_ms = float(latency_ms) if pd.notna(latency_ms) else 0.0

            message_base = str(entry.get('message', ''))
            extra_details = []
            for key in ['log_code', 'process_id', 'cpu_usage', 'request_id', 'user_id', 'trace_id']:
                 val = entry.get(key)
                 if val is not None and str(val).strip():
                     formatted_key = key.replace('_',' ').title()
                     extra_details.append(f"{formatted_key}: {val}")
            message = f"{message_base} ({', '.join(extra_details)})" if extra_details else message_base
            if not message.strip(): message = f"Log Event (Code: {entry.get('log_code', 'N/A')})"

            data.append({
                "timestamp": timestamp, "level": level, "module": module, "message": message,
                "raw": line, "status_code": status_code, "latency": latency_ms, "env": env
            })

        except json.JSONDecodeError:
            data.append({ "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "level": "PARSE_ERROR", "module": "parser", "message": f"JSON Decode Error line {i+1}", "raw": line[:200], "status_code": "N/A", "latency": 0.0, "env": "parser_error", "datetime": pd.NaT })
        except Exception as e:
             data.append({ "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "level": "PARSE_ERROR", "module": "parser", "message": f"Field Extraction Error line {i+1}: {e}", "raw": line[:200], "status_code": "N/A", "latency": 0.0, "env": "parser_error", "datetime": pd.NaT })

    return data, json_detected

def _parse_csv(logs: List[str], filename: Optional[str]) -> Tuple[List[Dict], bool]:
    """Parses logs assuming CSV format."""
    data = []
    csv_detected = is_csv_file(filename=filename, content=logs)
    if not csv_detected:
        return [], False

    try:
        log_data_str = "\n".join(filter(None, [str(l).strip() for l in logs]))
        if not log_data_str: raise ValueError("No valid content for CSV parsing.")

        df_temp = pd.read_csv(io.StringIO(log_data_str), on_bad_lines='warn', skipinitialspace=True)
        df_temp.columns = [col.strip().lower().replace(' ', '_') for col in df_temp.columns] # Sanitize

        rename_map = {
            'time': 'timestamp', 'log_level': 'level', 'service': 'module', 'component': 'module', 'components': 'module',
            'msg': 'message', 'text': 'message', 'status': 'status_code',
            'latency_ms': 'latency', 'response_time_ms': 'latency', 'response_time': 'latency_seconds',
            'duration_ms': 'latency', 'environment': 'env'
        }
        df_temp.rename(columns=rename_map, inplace=True)

        if 'latency_seconds' in df_temp.columns:
            latency_ms_from_sec = pd.to_numeric(df_temp['latency_seconds'], errors='coerce') * 1000
            if 'latency' not in df_temp.columns:
                 df_temp['latency'] = latency_ms_from_sec
            else:
                 df_temp['latency'] = pd.to_numeric(df_temp['latency'], errors='coerce')
                 df_temp['latency'].fillna(latency_ms_from_sec, inplace=True)
            df_temp.drop(columns=['latency_seconds'], inplace=True)

        # Define required columns for CSV context
        required_csv_cols = [col for col in REQUIRED_COLUMNS if col not in ['datetime', 'raw']]

        for col in required_csv_cols:
            if col not in df_temp.columns:
                 if col == 'latency': df_temp[col] = 0.0
                 elif col == 'level': df_temp[col] = 'INFO'
                 elif col == 'status_code': df_temp[col] = 'N/A'
                 elif col == 'env': df_temp[col] = 'unknown'
                 else: df_temp[col] = '' # module, message, timestamp

        # Convert types with error handling BEFORE creating 'raw'
        df_temp['latency'] = pd.to_numeric(df_temp['latency'], errors='coerce').fillna(0.0)
        df_temp['level'] = df_temp['level'].astype(str).str.upper()
        df_temp['status_code'] = df_temp['status_code'].astype(str)
        df_temp['module'] = df_temp['module'].astype(str)
        df_temp['env'] = df_temp['env'].astype(str)
        df_temp['message'] = df_temp['message'].astype(str)
        df_temp['timestamp'] = df_temp['timestamp'].astype(str)

        df_temp['raw'] = df_temp.apply(lambda row: ','.join(row.astype(str)), axis=1)

        # Select and structure data
        final_cols_to_select = [col for col in REQUIRED_COLUMNS if col in df_temp.columns and col != 'datetime']
        if 'raw' in df_temp.columns:
            final_cols_to_select.append('raw')

        for record in df_temp[final_cols_to_select].to_dict('records'):
             entry = {col: record.get(col, '') for col in REQUIRED_COLUMNS if col != 'datetime'} # Ensure all keys exist
             entry.update(record) # Override defaults
             # Handle specific defaults if needed (though handled above now)
             if 'latency' not in record: entry['latency'] = 0.0
             data.append(entry)

        return data, True

    except Exception as e:
        print(f"[Warning] CSV parsing failed: {e}. Trying generic line parsing.")
        traceback.print_exc()
        return [], False # Indicate CSV detection failed or errored

def _parse_regex(logs: List[str]) -> List[Dict]:
    """Fallback regex-based parsing for generic log lines."""
    data = []
    ts_patterns = [
        r'\[?(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)\]?', # ISOish
        r'\[?(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\]?',              # Syslog
        r'\[?(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})\]?'                 # Apache CLF
    ]
    level_patterns = [
        (r'\b(ERROR|ERR|CRITICAL|FATAL)\b', "ERROR"), (r'\b(WARN|WARNING)\b', "WARNING"),
        (r'\b(DEBUG)\b', "DEBUG"), (r'\b(INFO|NOTICE)\b', "INFO")
    ]
    module_patterns = [ r'\[([\w\-.]{3,})\]', r'\s([\w\-.]{3,}):', r'^([\w\-.]{3,})\s' ]
    status_patterns = [ r'(?:status|code|status_code)\W*(\d{3})\b', r'\s(\d{3})\s' ]
    latency_patterns = [ r'(\d+\.?\d*)\s?(?:ms|milliseconds)\b' ]
    env_patterns = [ r'\[(production|prod|staging|stag|development|dev|test)\]' ]

    for log_line in logs:
        line = str(log_line).strip()
        if not line: continue

        entry = {"raw": line, "level": "INFO", "module": "unknown", "status_code": "N/A", "latency": 0.0, "env": "unknown", "timestamp": ""}
        extracted_parts = {}

        for key, patterns in [('timestamp', ts_patterns), ('status_code', status_patterns), ('latency', latency_patterns), ('env', env_patterns)]:
             for pattern in patterns:
                 match = re.search(pattern, line, re.IGNORECASE)
                 if match:
                     found_val = match.group(1)
                     entry[key] = found_val
                     extracted_parts[key] = match.group(0)
                     break

        for pattern, lvl in level_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                entry["level"] = lvl
                extracted_parts["level"] = match.group(0)
                break

        for pattern in module_patterns:
             matches = re.finditer(pattern, line)
             for match in matches:
                 candidate = match.group(1)
                 if candidate and candidate.upper() not in ["ERROR", "WARN", "INFO", "DEBUG", "WARNING", "CRITICAL", "FATAL", "NOTICE"] and len(candidate) > 2:
                     entry["module"] = candidate
                     extracted_parts["module"] = match.group(0)
                     break
             if "module" in extracted_parts: break

        if isinstance(entry["latency"], str):
             try: entry["latency"] = float(entry["latency"])
             except ValueError: entry["latency"] = 0.0

        # Clean message
        message_cleaned = line
        for key in ['timestamp', 'level', 'module', 'env', 'latency', 'status_code']:
            if key in extracted_parts:
                try:
                    escaped_part = re.escape(extracted_parts[key])
                    message_cleaned = re.sub(escaped_part, '', message_cleaned, count=1)
                except re.error:
                     message_cleaned = message_cleaned.replace(extracted_parts[key], '', 1)

        entry["message"] = message_cleaned.strip(' []:-')
        if not entry["message"]: entry["message"] = entry["raw"] # Fallback

        data.append(entry)
    return data


def extract_components(logs_tuple: tuple, filename: Optional[str] = None) -> pd.DataFrame:
    """
    Extracts structured components from logs (handles JSONL, CSV, Regex).
    Accepts a tuple of log lines for cachability.
    Returns a DataFrame with standardized columns defined in config.REQUIRED_COLUMNS.
    """
    logs = list(logs_tuple) # Convert back to list internally
    if not logs:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    data = []
    parsed = False

    # 1. Try JSONL
    data, parsed = _parse_jsonl(logs)
    if parsed:
        print("Detected and parsed logs as JSONL format.")

    # 2. Try CSV if JSONL failed
    if not parsed:
        data, parsed = _parse_csv(logs, filename)
        if parsed:
            print("Detected and parsed logs as CSV format.")

    # 3. Fallback to Regex if others failed
    if not parsed and logs:
        print("Falling back to regex-based parsing for unrecognized format.")
        data = _parse_regex(logs)

    # --- Final DataFrame Creation and Standardization ---
    if not data:
        print("Warning: Log parsing yielded no structured data.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    try:
        df_extracted = pd.DataFrame(data)

        # Ensure all required columns exist, adding defaults if necessary
        for col in REQUIRED_COLUMNS:
            if col not in df_extracted.columns:
                if col == 'latency': df_extracted[col] = 0.0
                elif col == 'datetime': df_extracted[col] = pd.NaT
                elif col == 'level': df_extracted[col] = 'INFO'
                elif col == 'status_code': df_extracted[col] = 'N/A'
                elif col == 'env': df_extracted[col] = 'unknown'
                elif col == 'raw' and 'message' in df_extracted.columns: df_extracted[col] = df_extracted['message']
                elif col == 'raw': df_extracted[col] = ''
                elif col == 'message' and 'raw' in df_extracted.columns: df_extracted[col] = df_extracted['raw']
                elif col == 'message': df_extracted[col] = ''
                else: df_extracted[col] = 'unknown' # Default for module if missing

        # Standardize types and handle missing values AFTER ensuring columns exist
        df_extracted['latency'] = pd.to_numeric(df_extracted['latency'], errors='coerce').fillna(0.0).astype(float)
        df_extracted['status_code'] = df_extracted['status_code'].astype(str).fillna('N/A')
        df_extracted['level'] = df_extracted['level'].astype(str).fillna('INFO').str.upper()
        df_extracted['module'] = df_extracted['module'].astype(str).fillna('unknown')
        df_extracted['env'] = df_extracted['env'].astype(str).fillna('unknown')
        df_extracted['raw'] = df_extracted['raw'].astype(str).fillna('')
        df_extracted['message'] = df_extracted['message'].astype(str).fillna('')
        df_extracted.loc[df_extracted['message'].str.strip() == '', 'message'] = df_extracted['raw'] # Fallback message to raw if empty

        # Parse timestamp string into datetime objects AFTER filling NaNs
        df_extracted['timestamp'] = df_extracted['timestamp'].astype(str).fillna('')
        df_extracted['datetime'] = df_extracted['timestamp'].apply(parse_timestamp)

        # Final check and reorder columns
        final_cols = [col for col in REQUIRED_COLUMNS if col in df_extracted.columns]
        df_extracted = df_extracted[final_cols]
        # Add any completely missing required cols (shouldn't happen with loop above, but safeguard)
        for col in REQUIRED_COLUMNS:
            if col not in df_extracted.columns:
                if col == 'latency': df_extracted[col] = 0.0
                elif col == 'datetime': df_extracted[col] = pd.NaT
                else: df_extracted[col] = '' # Adjust default if needed
        # Enforce exact order
        df_extracted = df_extracted[REQUIRED_COLUMNS]

        return df_extracted

    except Exception as e:
        print(f"Error creating final DataFrame: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
