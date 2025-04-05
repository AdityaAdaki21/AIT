# clustering.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import traceback
from collections import Counter # For finding common terms
import re # For pattern matching
from typing import List, Dict, Tuple, Optional

# --- other imports ---
from utils import preprocess_log_message, get_stable_hash, format_time_delta
from config import (
    TFIDF_MAX_DF, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE, TFIDF_MAX_FEATURES,
    MIN_LOGS_FOR_CLUSTER, REQUIRED_COLUMNS
)
from llm_interface import analyze_cluster_summary

# --- _perform_kmeans_clustering (remains the same) ---
@st.cache_data(show_spinner="⏳ Clustering logs...")
def _perform_kmeans_clustering(
    _log_messages_tuple: tuple,
    n_clusters: int,
    _num_messages: int # Add number of messages explicitly to cache key
    ) -> Tuple[Optional[np.ndarray], Optional[TfidfVectorizer]]:
    # ... (implementation as previously corrected) ...
    log_messages = list(_log_messages_tuple)
    if len(log_messages) < n_clusters or len(log_messages) < 2:
        print(f"Not enough messages ({len(log_messages)}) for {n_clusters} clusters.")
        return None, None
    try:
        clean_logs = [preprocess_log_message(msg) for msg in log_messages]
        vectorizer = TfidfVectorizer(
            stop_words='english', max_df=TFIDF_MAX_DF, min_df=TFIDF_MIN_DF,
            ngram_range=TFIDF_NGRAM_RANGE, max_features=TFIDF_MAX_FEATURES
        )
        X = vectorizer.fit_transform(clean_logs)
        if X.shape[0] < n_clusters or X.shape[1] == 0:
             print(f"TF-IDF Vectorization resulted in insufficient data shape: {X.shape}. Cannot cluster.")
             return None, None
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        if len(labels) != len(log_messages):
             print(f"CRITICAL ERROR: K-Means returned {len(labels)} labels for {len(log_messages)} inputs!")
             return None, None
        return labels, vectorizer
    except Exception as e:
        print(f"Error during K-Means clustering: {e}")
        traceback.print_exc()
        return None, None

def _extract_common_patterns(messages: pd.Series, top_n: int = 5) -> Dict:
    """Extract common keywords or error codes from messages."""
    # Simple keyword extraction (can be improved with regex)
    # Exclude common words and very short words
    common_words = set(['the', 'a', 'is', 'in', 'it', 'and', 'of', 'to', 'error', 'failed', 'exception', 'request', 'response', 'for', 'on', 'at', 'with'])
    word_counts = Counter()
    # Regex for potential error codes (e.g., E123, ABC_XYZ, 5xx)
    code_regex = r'\b([A-Z]+[_\-]?\d+|\d{3}|[A-Z]{3,}(?:_[A-Z]+)+)\b'
    code_counts = Counter()

    for msg in messages:
        if not isinstance(msg, str): continue
        # Extract potential codes
        codes = re.findall(code_regex, msg)
        code_counts.update(c for c in codes if len(c) > 2) # Count codes

        # Extract words for keywords
        words = re.findall(r'\b[a-zA-Z]{3,}\b', msg.lower()) # Words with 3+ letters
        word_counts.update(w for w in words if w not in common_words)

    # Get top N, prioritize codes slightly if available
    top_keywords = {k: v for k, v in word_counts.most_common(top_n)}
    top_codes = {k: v for k, v in code_counts.most_common(top_n)}

    return {"top_keywords": top_keywords, "top_codes": top_codes}


def run_log_clustering(log_df: pd.DataFrame, n_clusters: int) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[List[Dict]]]:
    """
    Performs clustering and generates enhanced summaries for developers.
    """
    if 'cluster' in log_df.columns:
        log_df = log_df.drop(columns=['cluster'], errors='ignore')

    cluster_col = 'raw' if 'raw' in log_df.columns else 'message'
    if cluster_col not in log_df.columns:
         st.error(f"Cannot cluster: Missing '{cluster_col}' column.")
         return log_df.assign(cluster=-1), None, None

    valid_logs_for_clustering = log_df[cluster_col].dropna().astype(str)
    log_indices = valid_logs_for_clustering.index
    num_valid_logs = len(valid_logs_for_clustering)

    if num_valid_logs < MIN_LOGS_FOR_CLUSTER:
        st.warning(f"Insufficient logs ({num_valid_logs}) for clustering (minimum {MIN_LOGS_FOR_CLUSTER}). Skipping.")
        return log_df.assign(cluster=-1), None, None

    actual_n_clusters = min(n_clusters, num_valid_logs)
    if actual_n_clusters < 2:
        st.warning("Need at least 2 distinct log patterns to form clusters. Skipping clustering.")
        return log_df.assign(cluster=-1), None, None
    if actual_n_clusters < n_clusters:
        st.warning(f"Reduced number of clusters to {actual_n_clusters} due to limited distinct log patterns.")

    labels, _vectorizer = _perform_kmeans_clustering(
        tuple(valid_logs_for_clustering.tolist()), actual_n_clusters, num_valid_logs
    )

    if labels is None:
        st.error("Clustering process failed. See console logs for details.")
        return log_df.assign(cluster=-1), None, None
    if len(labels) != num_valid_logs:
         st.error(f"Clustering label mismatch ({len(labels)} vs {num_valid_logs}). Aborting assignment.")
         return log_df.assign(cluster=-1), None, None

    cluster_labels_df = pd.DataFrame({'cluster': labels}, index=log_indices)
    updated_log_df = log_df.join(cluster_labels_df)
    updated_log_df['cluster'] = updated_log_df['cluster'].fillna(-1).astype(int)

    clusters_summary_list = []
    error_profiles_list = []
    valid_clusters = sorted([c for c in updated_log_df['cluster'].unique() if c >= 0])

    if not valid_clusters:
         st.warning("Clustering ran but did not produce any valid cluster groups (>= 0).")
         return updated_log_df, None, None

    for cluster_id in valid_clusters:
        cluster_df = updated_log_df[updated_log_df["cluster"] == cluster_id]
        total = len(cluster_df)
        if total == 0: continue

        # --- Calculate Enhanced Summary Stats ---
        cluster_df['level_upper'] = cluster_df['level'].astype(str).str.upper()
        errors = int(cluster_df["level_upper"].eq("ERROR").sum())
        warnings = int(cluster_df["level_upper"].eq("WARNING").sum())
        error_rate = round(errors / total * 100, 1) if total > 0 else 0.0
        warning_rate = round(warnings / total * 100, 1) if total > 0 else 0.0

        top_modules_series = cluster_df["module"].value_counts().head(3)
        top_modules = {k: int(v) for k, v in top_modules_series.items()}

        top_status_series = cluster_df["status_code"][cluster_df["status_code"].astype(str).ne('N/A')].value_counts().head(3)
        top_status = {k: int(v) for k, v in top_status_series.items()}

        cluster_latency = pd.to_numeric(cluster_df["latency"], errors='coerce').fillna(0)
        positive_latency = cluster_latency[cluster_latency > 0]
        avg_latency = float(round(positive_latency.mean(), 1)) if not positive_latency.empty else 0.0
        p95_latency = float(round(positive_latency.quantile(0.95), 1)) if not positive_latency.empty else 0.0 # Added P95

        # Get timestamp range for the cluster
        first_ts, last_ts = None, None
        time_span_str = "N/A"
        if 'datetime' in cluster_df.columns and cluster_df['datetime'].notna().any():
             valid_times = cluster_df['datetime'].dropna()
             if not valid_times.empty:
                  first_ts = valid_times.min()
                  last_ts = valid_times.max()
                  time_span_str = format_time_delta(last_ts - first_ts) # Use helper

        # Extract common patterns from error/warning messages in this cluster
        error_warn_messages = cluster_df[cluster_df['level_upper'].isin(['ERROR', 'WARNING'])]['message']
        common_patterns = {}
        if not error_warn_messages.empty:
             common_patterns = _extract_common_patterns(error_warn_messages)

        # Get diverse raw samples
        sample_logs = []
        for lvl in ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'PARSE_ERROR']:
            level_samples = cluster_df[cluster_df['level_upper'] == lvl]['raw'].head(2).tolist()
            sample_logs.extend(level_samples)
            if len(sample_logs) >= 4: break
        if len(sample_logs) < 4:
             other_samples = cluster_df[~cluster_df['level_upper'].isin(['ERROR', 'WARNING', 'INFO', 'DEBUG', 'PARSE_ERROR'])]['raw'].head(4 - len(sample_logs)).tolist()
             sample_logs.extend(other_samples)
        samples = sample_logs[:4]
        # --- End Enhanced Summary Stats ---

        cluster_summary = {
            "cluster_id": int(cluster_id),
            "total_logs": total,
            "error_count": errors,
            "warning_count": warnings,
            "error_rate": error_rate,
            "warning_rate": warning_rate,
            "top_modules": top_modules,
            "top_status_codes": top_status,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency, # Added
            "first_timestamp": first_ts.isoformat() if first_ts else None, # Added
            "last_timestamp": last_ts.isoformat() if last_ts else None, # Added
            "time_span": time_span_str, # Added
            "common_error_patterns": common_patterns, # Added
            "sample_logs": samples
        }
        clusters_summary_list.append(cluster_summary)

        if errors > 0:
            error_logs_df = cluster_df[cluster_df["level_upper"] == "ERROR"]
            error_modules_series = error_logs_df["module"].value_counts().head(3)
            error_modules = {k: int(v) for k, v in error_modules_series.items()}
            error_status_series = error_logs_df["status_code"][error_logs_df["status_code"].astype(str).ne('N/A')].value_counts().head(3)
            error_status = {k: int(v) for k, v in error_status_series.items()}
            err_latency = pd.to_numeric(error_logs_df["latency"], errors='coerce').fillna(0)
            avg_err_latency = float(round(err_latency[err_latency > 0].mean(), 1)) if not err_latency[err_latency > 0].empty else 0.0
            p95_err_latency = float(round(err_latency[err_latency > 0].quantile(0.95), 1)) if not err_latency[err_latency > 0].empty else 0.0 # Added P95
            sample_errors = error_logs_df["raw"].head(3).tolist()

            # Extract common patterns specifically from error messages
            error_only_patterns = _extract_common_patterns(error_logs_df['message']) if not error_logs_df.empty else {}

            error_profiles_list.append({
                "cluster_id": int(cluster_id),
                "error_count": errors,
                "error_rate": error_rate,
                "error_modules": error_modules,
                "error_status_codes": error_status,
                "avg_error_latency_ms": avg_err_latency,
                "p95_error_latency_ms": p95_err_latency, # Added
                "common_error_keywords": error_only_patterns.get("top_keywords", {}), # Added
                "common_error_codes": error_only_patterns.get("top_codes", {}), # Added
                "sample_errors": sample_errors
            })

    st.success(f"✅ Clustering complete. Found {len(valid_clusters)} clusters.", icon="📊")
    return updated_log_df, clusters_summary_list, error_profiles_list


# --- get_ai_cluster_interpretation (remains the same, but benefits from enhanced summary) ---
@st.cache_data(show_spinner="🧠 Getting AI interpretation for cluster...")
def get_ai_cluster_interpretation(cluster_summary: Dict, llm_config: Dict) -> str:
    """Uses the LLM to interpret a single cluster's summary."""
    # ... (implementation is unchanged) ...
    if not cluster_summary or not llm_config:
        return "Error: Missing cluster summary or LLM configuration for interpretation."

    cluster_id = cluster_summary.get("cluster_id", "N/A")
    try:
        # Ensure the input dict is serializable (Timestamps are now strings)
        # We still need the helper from ui_components for numpy types before dumping
        # --> This suggests maybe the serialization helper should live in utils.py
        # For now, assume the calling function serializes properly.
        summary_str = json.dumps(cluster_summary, indent=2) # Might still fail if numpy types remain
        interpretation = analyze_cluster_summary(cluster_id, summary_str, llm_config)
        return interpretation
    except TypeError as te:
        print(f"JSON Serialization Error during AI interpretation request for cluster {cluster_id}: {te}")
        print(f"Problematic summary data passed: {cluster_summary}")
        return f"Error: Could not serialize cluster summary for AI analysis due to data type issues (check console)."
    except Exception as e:
        print(f"Error getting AI interpretation for cluster {cluster_id}: {e}")
        traceback.print_exc()
        return f"Error generating AI interpretation for cluster {cluster_id}. Check logs."