import random
from datetime import datetime, timedelta
import os
import pandas as pd

# Try to import config, handle gracefully if running standalone
try:
    from config import SAMPLE_LOG_COUNT, SAMPLE_LOG_CSV_FILENAME
except ImportError:
    print("Warning: config.py not found. Using default sample log settings.")
    SAMPLE_LOG_COUNT = 300
    SAMPLE_LOG_CSV_FILENAME = 'api_logs.csv'

# Rest of the file remains the same as your provided version...

def generate_sample_logs(num_logs=SAMPLE_LOG_COUNT): # Use config default
    """Generate sample log entries for demonstration purposes"""
    log_levels = ["INFO", "WARNING", "ERROR"]
    modules = ["auth-service", "api-gateway", "database", "cache", "user-service", "payment-processor"]
    environments = ["production", "development"]

    api_endpoints = [
        "/api/v1/users", "/api/v1/auth/login", "/api/v1/products",
        "/api/v1/orders", "/api/v1/payment"
    ]
    status_codes = [200, 201, 400, 401, 403, 404, 500, 503]
    status_weights = [70, 10, 5, 5, 2, 3, 3, 2]
    error_messages = [
        "Connection timeout", "Invalid authentication token", "Database query failed",
        "Resource not found", "Permission denied", "Internal server error",
        "Service unavailable", "Rate limit exceeded"
    ]
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    logs = []

    for _ in range(num_logs):
        log_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
        timestamp = log_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Milliseconds
        module = random.choice(modules)
        env = random.choice(environments)
        endpoint = random.choice(api_endpoints)
        latency = random.randint(10, 100)
        if random.random() < 0.1: latency = random.randint(500, 2000)
        status_code = random.choices(status_codes, weights=status_weights, k=1)[0]

        if status_code >= 500: level = "ERROR"
        elif status_code >= 400: level = random.choices(["WARNING", "ERROR"], weights=[70, 30], k=1)[0]
        else: level = random.choices(["INFO", "WARNING"], weights=[95, 5], k=1)[0]

        if level == "ERROR": message = f"Failed req {random.randint(1000,9999)} for {endpoint}: {random.choice(error_messages)}"
        elif level == "WARNING": message = f"High latency on {endpoint} ({latency}ms)" if latency > 500 else f"Client error on {endpoint}"
        else: message = f"Req {random.randint(1000,9999)} processed for {endpoint}"

        log_line = f"{timestamp} [{level}] [{module}] [{env}] {message} - status={status_code} latency={latency}ms request_id=req-{random.randint(10000, 99999)}"
        logs.append(log_line)

    # Add specific patterns
    for _ in range(20):
        log_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
        timestamp = log_time.strftime("%Y-%m-%d %H:%M:%S")
        logs.append(f"{timestamp} [ERROR] [database] [production] Connection pool exhausted - reconnecting - Pool size: {random.randint(10, 50)}")
    for _ in range(15):
        log_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
        timestamp = log_time.strftime("%Y-%m-%d %H:%M:%S")
        logs.append(f"{timestamp} [WARN] [auth-service] [production] Rate limiting for IP {random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}")

    random.shuffle(logs)
    return logs


def fetch_logs_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df = df.astype(str)
        header = ','.join(df.columns.tolist())
        log_lines = [header] + df.apply(lambda row: ','.join(row), axis=1).tolist()
        print(f"Successfully read {len(log_lines)-1} log entries from {csv_path}")
        return log_lines
    except FileNotFoundError: print(f"Error: CSV file not found at {csv_path}")
    except Exception as e: print(f"Error reading CSV {csv_path}: {str(e)}")
    return None

def fetch_logs():
    csv_filename = SAMPLE_LOG_CSV_FILENAME # Use config constant
    try:
        script_dir = os.path.dirname(__file__)
        csv_path = os.path.join(script_dir, csv_filename)
    except NameError:
         csv_path = csv_filename # Fallback to CWD

    print(f"Attempting to load sample logs from: {csv_path}")
    if os.path.exists(csv_path):
        logs = fetch_logs_from_csv(csv_path)
        if logs: return logs
        else: print(f"Failed to read {csv_path}. Falling back to generated logs.")
    else:
        print(f"{csv_filename} not found. Generating sample logs.")

    return generate_sample_logs() # Uses SAMPLE_LOG_COUNT from config


# Example usage (for testing this script directly)
# if __name__ == "__main__":
#     sample_logs = fetch_logs()
#     if sample_logs:
#         print(f"Fetched/Generated {len(sample_logs)} log entries.")
#         # Print first 5 and last 5
#         for log in sample_logs[:5]:
#             print(log)
#         print("...")
#         for log in sample_logs[-5:]:
#             print(log)
#     else:
#         print("Failed to fetch or generate sample logs.")

# --- END OF FILE logs_backend.py ---
