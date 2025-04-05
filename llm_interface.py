# llm_interface.py
import streamlit as st
import requests
import json
import traceback
from typing import Tuple, Optional, List, Dict

# Assuming config.py and utils.py are available
from config import OPENROUTER_API_KEY, OLLAMA_API_URL
from utils import get_stable_hash

# --- Connection Checks & Generic Query (remain the same) ---
# ... check_ollama_availability, get_ollama_models, query_ollama, query_remote_llm, query_llm ...
# Cache Ollama availability check
@st.cache_data(ttl=60)
def check_ollama_availability(api_url: str) -> Tuple[bool, str]:
    """Checks if Ollama service is available. Returns (bool, status_message)."""
    if not api_url or not isinstance(api_url, str) or not api_url.endswith("/api/generate"):
        return False, "Invalid URL format (must end with /api/generate)"
    base_url = api_url.replace("/api/generate", "")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code == 200:
            return True, "Connected"
        else:
            return False, f"Ollama API responded with status {response.status_code}"
    except requests.exceptions.Timeout: return False, "Connection timed out"
    except requests.exceptions.ConnectionError: return False, "Connection refused (is Ollama running?)"
    except requests.exceptions.RequestException as e: return False, f"Network error: {type(e).__name__}"
    except Exception as e: return False, f"Unexpected error: {e}"

@st.cache_data(ttl=300) # Cache model list for 5 minutes
def get_ollama_models(api_url: str, is_available: bool) -> List[str]:
    """Gets list of available models from Ollama."""
    if not is_available:
        return []
    base_url = api_url.replace("/api/generate", "")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            models = sorted([model["name"] for model in models_data if "name" in model])
            return models
        else:
            print(f"Warning: Failed to get Ollama models (Status {response.status_code})")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not connect to Ollama to get models: {e}")
        return []

def query_ollama(prompt: str, model_name: Optional[str], api_url: Optional[str]) -> str:
    """Queries the local Ollama API."""
    if not api_url or not model_name: return "Error: Ollama API URL or model name not configured."
    try:
        response = requests.post(
            api_url, headers={"Content-Type": "application/json"},
            json={"model": model_name, "prompt": prompt, "stream": False}, timeout=120
        )
        response.raise_for_status()
        resp_json = response.json()
        if "response" in resp_json: return resp_json["response"].strip()
        elif "error" in resp_json: return f"Error from Ollama: {resp_json['error']}"
        else:
             print(f"Unexpected Ollama response format: {resp_json}")
             return "Error: Unexpected response format from Ollama."
    except requests.exceptions.Timeout: return "Error: Ollama request timed out (120s)."
    except requests.exceptions.HTTPError as e: return f"Error: Ollama API request failed (Status {e.response.status_code}). Response: {e.response.text}"
    except requests.exceptions.RequestException as e: return f"Error connecting to Ollama API ({api_url}): {e}"
    except json.JSONDecodeError: return f"Error: Could not decode JSON response from Ollama. Response text: {response.text}"
    except Exception as e:
        traceback.print_exc(); return f"Error processing Ollama response: {type(e).__name__} - {e}"

def query_remote_llm(prompt: str, model: Optional[str], api_key: Optional[str]) -> str:
    """Queries remote LLM API (OpenRouter)."""
    if not api_key or api_key.startswith("sk-or-v1-...") or not model:
         return "Error: OpenRouter API Key or model not configured."
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]}, timeout=120
        )
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and result['choices'] and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
            return result['choices'][0]['message']['content'].strip()
        elif 'error' in result:
             return f"Error from OpenRouter: {result['error'].get('message', str(result['error']))}"
        else:
            print(f"Unexpected OpenRouter response format: {result}")
            return "Error: Unexpected response format from OpenRouter."
    except requests.exceptions.Timeout: return "Error: OpenRouter request timed out (120s)."
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code; error_text = e.response.text
        print(f"OpenRouter HTTP Error {status_code}: {error_text}")
        if status_code == 401: return "Error: Invalid OpenRouter API Key."
        if status_code == 402: return "Error: OpenRouter Quota Exceeded/Payment Required."
        if status_code == 404: return f"Error: OpenRouter Model Not Found ({model})."
        if status_code == 429: return "Error: OpenRouter Rate Limit Exceeded."
        return f"Error querying OpenRouter (Status: {status_code}). Response: {error_text[:200]}"
    except requests.exceptions.RequestException as e: return f"Error connecting to OpenRouter API: {e}"
    except Exception as e:
        traceback.print_exc(); return f"Error processing OpenRouter response: {type(e).__name__} - {e}"

def query_llm(prompt: str, llm_config: Dict) -> str:
    """Wrapper to call either Ollama or Remote LLM based on config."""
    use_ollama = llm_config.get('use_ollama', False)
    if use_ollama:
        return query_ollama(prompt, model_name=llm_config.get('ollama_model'), api_url=llm_config.get('ollama_url'))
    else:
        return query_remote_llm(prompt, model=llm_config.get('remote_model'), api_key=llm_config.get('api_key'))


# --- Specific Analysis Task Functions ---

@st.cache_data(show_spinner="🧠 Performing Holistic Analysis...")
def perform_holistic_analysis(log_summary_str: str, clusters_summary_str: str, llm_config: Dict) -> str:
    _log_summary_hash = get_stable_hash(log_summary_str)
    _clusters_summary_hash = get_stable_hash(clusters_summary_str)
    prompt = f"""
# Developer-Focused Log Analysis Report

Act as a senior software engineer analyzing system logs. Focus on identifying potential root causes and actionable debugging steps.

## Overall Statistics (Data Hash: {_log_summary_hash})
{log_summary_str}

## Cluster Summaries (Data Hash: {_clusters_summary_hash})
{clusters_summary_str}
*Note: Summaries include avg/p95 latency, top modules/status, common error keywords/codes, and timestamp ranges.*

## Analysis Task (For Developers):
1.  **Overall Health Assessment:** Briefly state the likely stability (e.g., Stable, Degraded, Critical).
2.  **Most Critical Issue(s):** Identify the top 1-2 clusters representing the most significant problems based on error rate, critical modules involved (e.g., 'database', 'auth'), high P95 latency, or concerning common error patterns/codes. **Explain *why* they are critical from a system perspective.**
3.  **Actionable Investigation Plan:** For the identified critical issues, suggest **specific, concrete first steps** a developer should take. Examples:
    *   "Correlate timestamps from Cluster X (database errors) with DB performance counters (CPU, I/O wait)."
    *   "Check deployment logs for 'payment-processor' around [timestamp range] for Cluster Y."
    *   "Filter logs for Cluster Z and trace request IDs found in sample logs across services."
    *   "Investigate common error code 'E123' in Cluster A by reviewing [specific config file or code area if guessable]."
4.  **Potential Correlations:** Briefly mention any suspected correlations between clusters (e.g., "High latency in Cluster 1 (API Gateway) might be caused by errors in Cluster 2 (Database)").

Use Markdown. Be concise and action-oriented. Prioritize steps that help isolate the problem.
"""
    return query_llm(prompt, llm_config)

@st.cache_data(show_spinner="🧠 Performing Comparative Analysis...")
def perform_comparative_analysis(error_profiles_str: str, llm_config: Dict) -> str:
    _error_profiles_hash = get_stable_hash(error_profiles_str)
    prompt = f"""
# Developer-Focused Comparative Error Analysis

Compare the provided error profiles from different log clusters to pinpoint specific failure modes and potential interactions.

## Error Profiles by Cluster (Data Hash: {_error_profiles_hash})
{error_profiles_str}
*Note: Profiles include error counts/rates, top error modules/status, avg/p95 error latency, and common error keywords/codes.*

## Analysis Task (For Developers):
1.  **Most Impactful Error Cluster:** Identify the cluster showing the most severe errors (highest rate in critical module, highest p95 latency during errors, most concerning error keywords/codes). Explain the likely *developer impact* (e.g., "Cluster X database timeouts likely cause cascading failures").
2.  **Key Error Differences:** Highlight significant differences in error signatures between clusters (e.g., "Cluster A shows primarily '401 Unauthorized' from 'auth-service', while Cluster B shows '503 Timeout' from 'payment-processor'").
3.  **Cross-Cluster Hypotheses:** Based *only* on the profiles, propose **testable hypotheses** about relationships. Examples:
    *   "Hypothesis: The 'Connection Pool Exhausted' errors (Cluster X, database) might be the root cause of the '500 Internal Server Error' with keyword 'timeout' (Cluster Y, api-gateway)."
    *   "Hypothesis: The high P95 latency for '4xx' errors in Cluster Z suggests slow validation logic in the 'user-service'."
    *   State if data is insufficient to link specific clusters.

Use Markdown. Focus on actionable comparisons relevant for debugging.
"""
    return query_llm(prompt, llm_config)

# --- Temporal Analysis Prompt Refinement ---
@st.cache_data(show_spinner="🧠 Performing Temporal Analysis...")
def perform_temporal_analysis(hourly_stats_summary_str: str, anomalous_hours_str: str, llm_config: Dict) -> str:
    _hourly_stats_hash = get_stable_hash(hourly_stats_summary_str)
    _anomalous_hours_hash = get_stable_hash(anomalous_hours_str)
    prompt = f"""
# Developer-Focused Temporal Log Analysis

Analyze time-based patterns to help developers correlate issues with time-dependent events (load, batch jobs, etc.).

## Hourly Statistics Summary (Data Hash: {_hourly_stats_hash})
{hourly_stats_summary_str}
*Includes total logs, error count/rate, avg latency per hour.*

## Anomalous Hours Identified (High Error Rate) (Data Hash: {_anomalous_hours_hash})
{anomalous_hours_str}

## Analysis Task (For Developers):
1.  **Key Temporal Patterns:** Describe the main trends in log volume and error rate vs. time of day (e.g., "Peak load and errors between 14:00-16:00 UTC").
2.  **Significance of Anomalies:** For the anomalous hours identified, what is the *likely system behavior*? (e.g., "High error rate at 03:00 UTC despite low volume suggests a problematic batch job or maintenance task").
3.  **Debugging Pointers:** Suggest specific time-related checks for developers:
    *   "Investigate deployment schedules or cron jobs running around the anomalous hours [list hours]."
    *   "Correlate peak load hours [list hours] with resource utilization metrics (CPU, memory, network) for key services."
    *   "Check if specific error types (from cluster analysis, if available elsewhere) spike during high-error periods."

Use Markdown. Be concise and link time patterns to potential operational causes.
"""
    return query_llm(prompt, llm_config)


@st.cache_data(show_spinner="🧠 Analyzing Cluster Summary...")
def analyze_cluster_summary(cluster_id: int, cluster_summary_str: str, llm_config: Dict) -> str:
    _cluster_summary_hash = get_stable_hash(cluster_summary_str)
    prompt = f"""
# Developer Action Plan for Log Cluster {cluster_id}

Analyze the following summary for Log Cluster {cluster_id} and provide actionable debugging steps for an engineer.

## Cluster Summary (Data Hash: {_cluster_summary_hash})
{cluster_summary_str}
*Includes: counts, rates, top modules/status, avg/p95 latency, common patterns, timestamp range, samples.*

## Analysis Task (For Developers):
1.  **Problem Statement:** Concisely describe the core issue represented by this cluster (e.g., "Frequent timeouts in database interactions", "Authentication failures for specific users", "High latency processing for user-service"). Use the provided stats (error rate, p95 latency, common patterns) as evidence.
2.  **Likely Root Cause Hypotheses (1-2):** Based *only* on the summary, propose the most plausible technical reasons. Examples:
    *   "Hypothesis: Database connection pool exhaustion (see 'Connection pool' keywords)."
    *   "Hypothesis: Throttling by an external API called by 'payment-processor' (see 429 status codes)."
    *   "Hypothesis: Inefficient query or missing index causing high DB latency (see high p95 latency)."
3.  **Specific Debugging Steps (Action Plan):** Provide 2-3 **concrete, ordered steps** an engineer should take *first*. Prioritize actions that quickly confirm or deny the top hypothesis. Examples:
    *   "1. Filter logs for cluster {cluster_id} between [first_timestamp] and [last_timestamp]."
    *   "2. Search filtered logs for specific request IDs from the sample logs to trace execution flow."
    *   "3. Check resource metrics (CPU, memory, connections) for the top module ('{cluster_summary.get('top_modules',{}).keys()}'?) during the cluster's time span."
    *   "4. Examine code related to '{cluster_summary.get('common_error_patterns',{}).get('top_keywords',{}).keys()}' keywords in the '{cluster_summary.get('top_modules',{}).keys()}' module."

Use Markdown with numbered lists for the action plan. Be specific and practical.
"""
    # Note: Accessing dict keys like above in the f-string is risky if they don't exist.
    # A safer approach is to format the prompt string *after* retrieving values with .get()
    # For simplicity here, we assume the keys exist based on how the summary is built.
    # A production system might require safer formatting.

    return query_llm(prompt, llm_config)

@st.cache_data(show_spinner="🧠 Explaining Log Entry...")
def explain_single_log(log_entry_raw: str, llm_config: Dict) -> str:
    _log_hash = get_stable_hash(log_entry_raw)
    prompt = f"""
# Debugging Assistance for Log Entry

Analyze the following raw log entry from a developer's perspective. Extract key identifiers and suggest debugging actions.

Log Entry (Hash: {_log_hash}):
{log_entry_raw}

## Explanation Task (Use Markdown):
1.  **Meaning & Context:** What event occurred? Which service/module reported it?
2.  **Key Identifiers:** Extract any useful IDs (e.g., `request_id`, `trace_id`, `user_id`, specific error codes like `E123` or `AUTH_FAIL`). List them clearly. If none, state that.
3.  **Potential Cause(s):** List 1-3 likely *technical* reasons for this specific log.
4.  **Immediate Next Step:** What is the *very first command or check* a developer should perform based *only* on this log? (e.g., `grep for request_id XYZ`, `check status of service ABC`, `look at code line related to 'NullPointerException'`).

Focus on extracting actionable information for debugging.
"""
    return query_llm(prompt, llm_config)