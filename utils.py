# utils.py
import pandas as pd
from datetime import datetime, timedelta
import re
import json
from typing import Optional, Tuple

def parse_timestamp(ts_str: Optional[str]) -> pd.Timestamp:
    """Parses timestamp strings into datetime objects, handling common formats."""
    if not isinstance(ts_str, str) or not ts_str.strip():
        return pd.NaT # Return Not-a-Time for invalid input

    ts_str = ts_str.strip()
    # Try common formats directly (ordered by likely frequency or specificity)
    formats = [
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f", # ISO-like with microseconds
        "%Y-%m-%d %H:%M:%S,%f",                        # ISO-like with comma microseconds
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",       # ISO-like without microseconds
        "%Y/%m/%d %H:%M:%S",                            # Slash separators
        "%d/%b/%Y:%H:%M:%S",                            # Apache CLF (timezone offset usually needs separate handling if present)
        "%b %d %H:%M:%S",                               # Syslog style (assumes current year, handles Dec->Jan rollover)
    ]
    now = datetime.now()
    dt = pd.NaT
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str, fmt)
            # Handle missing year in syslog format
            if fmt == "%b %d %H:%M:%S" and dt.year == 1900: # Default year is 1900
                dt = dt.replace(year=now.year)
                # Check for year rollover (e.g., log from Dec, current time Jan)
                if dt > now + timedelta(days=1): # If parsed date is significantly in the future
                    dt = dt.replace(year=now.year - 1)
            break # Success
        except ValueError:
            continue # Try next format

    # Fallback using pandas for wider format compatibility if specific formats failed
    if pd.isna(dt):
        try:
            dt_pd = pd.to_datetime(ts_str, errors='coerce')
            if pd.notna(dt_pd):
                 dt = dt_pd.to_pydatetime() # Convert to Python datetime if possible
            else:
                 dt = pd.NaT
        except Exception:
             dt = pd.NaT

    if isinstance(dt, datetime):
        return pd.Timestamp(dt)
    else:
        return pd.NaT # Return NaT if still not parsed

def is_csv_file(filename: Optional[str] = None, content: Optional[list] = None) -> bool:
    """Checks if filename suggests CSV or if content looks like CSV."""
    if filename and filename.lower().endswith('.csv'):
        return True
    if content and len(content) > 0:
        sample_lines = [str(line).strip() for line in content if str(line).strip()][:5] # Check first 5 lines
        if not sample_lines: return False
        if all(line.count(',') >= 1 for line in sample_lines):
             header = sample_lines[0].lower()
             if any(kw in header for kw in ['timestamp', 'time', 'level', 'status', 'message', 'module']):
                 return True
    return False

def preprocess_log_message(log: str) -> str:
    """Basic preprocessing for NLP: lowercase and simplify non-alphanumeric chars, keeping essentials."""
    # Keep common log punctuation: :, [], -, =, ., % but remove others
    # Also keep / and _ often found in paths or identifiers
    if not isinstance(log, str):
        log = str(log)
    return re.sub(r'[^\w\s,:\[\]\-=\.\%\/\_]', '', log.lower())

def get_stable_hash(data) -> int:
    """Creates a stable hash for cache keys from dicts, lists, or DataFrames."""
    if isinstance(data, pd.DataFrame):
        # Hash based on shape and a sample of the data converted to string
        # Consider hashing column names and dtypes too for more robustness
        sample_str = data.head().to_string() + str(data.shape) + "".join(data.columns) + "".join(map(str, data.dtypes))
        return hash(sample_str)
    elif isinstance(data, (dict, list)):
        try:
            # Sort dicts by key for stable JSON representation
            return hash(json.dumps(data, sort_keys=True))
        except TypeError: # Handle unhashable types if they sneak in
            return hash(str(data)) # Fallback to string representation hash
    else:
        return hash(data)

def format_time_delta(time_delta: timedelta) -> str:
    """Formats a timedelta into a human-readable string (e.g., 1d 2h 5m)."""
    if not isinstance(time_delta, timedelta) or pd.isna(time_delta):
        return "N/A"

    days = time_delta.days
    total_seconds = time_delta.seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0: parts.append(f"{minutes}m")
    if not parts and seconds >= 0: # Show seconds only if duration < 1 min
         microseconds = time_delta.microseconds
         if seconds > 0 or microseconds == 0:
             parts.append(f"{seconds}s")
         elif microseconds > 0: # Show ms if seconds is 0
             parts.append(f"{microseconds // 1000}ms")

    return " ".join(parts) if parts else "~0s" # Default if very short


def calculate_dashboard_metrics(df: pd.DataFrame) -> dict:
    """Calculates key metrics for the dashboard."""
    metrics = {
        'total_logs': 0, 'error_count': 0, 'warning_count': 0, 'error_rate': 0.0,
        'warning_rate': 0.0, 'unique_modules': 0, 'avg_latency': 0.0,
        'log_start_time': None, 'log_end_time': None, 'log_time_span_str': "N/A"
    }
    if df.empty:
        return metrics

    metrics['total_logs'] = len(df)

    if 'level' in df.columns:
        # Use .astype(str) for robustness against non-string types
        level_series = df['level'].astype(str).str.upper()
        metrics['error_count'] = int(level_series.eq("ERROR").sum())
        metrics['warning_count'] = int(level_series.eq("WARNING").sum())

    if 'module' in df.columns:
        metrics['unique_modules'] = df["module"].nunique()

    if metrics['total_logs'] > 0:
        metrics['error_rate'] = round(metrics['error_count'] / metrics['total_logs'] * 100, 1)
        metrics['warning_rate'] = round(metrics['warning_count'] / metrics['total_logs'] * 100, 1)

    if 'latency' in df.columns:
        valid_latency = pd.to_numeric(df["latency"], errors='coerce').fillna(0)
        positive_latency = valid_latency[valid_latency > 0]
        if not positive_latency.empty:
            metrics['avg_latency'] = positive_latency.mean()

    if 'datetime' in df.columns and df['datetime'].notna().any():
        valid_times = df['datetime'].dropna()
        if not valid_times.empty:
            metrics['log_start_time'] = valid_times.min()
            metrics['log_end_time'] = valid_times.max()
            time_delta = metrics['log_end_time'] - metrics['log_start_time']
            metrics['log_time_span_str'] = format_time_delta(time_delta)
        else: metrics['log_time_span_str'] = "No valid times"
    else: metrics['log_time_span_str'] = "No timestamp data"

    return metrics
