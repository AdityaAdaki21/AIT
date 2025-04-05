# --- START OF FILE logs_backend.py ---

# logs_backend.py
import random
from datetime import datetime, timedelta
import os
import pandas as pd

def generate_sample_logs(num_logs=200):
    """Generate sample log entries for demonstration purposes"""
    log_levels = ["INFO", "WARNING", "ERROR"]
    modules = ["auth-service", "api-gateway", "database", "cache", "user-service", "payment-processor"]
    environments = ["production", "development"]

    api_endpoints = [
        "/api/v1/users",
        "/api/v1/auth/login",
        "/api/v1/products",
        "/api/v1/orders",
        "/api/v1/payment"
    ]

    status_codes = [200, 201, 400, 401, 403, 404, 500, 503]
    status_weights = [70, 10, 5, 5, 2, 3, 3, 2]  # Weighted probabilities

    error_messages = [
        "Connection timeout",
        "Invalid authentication token",
        "Database query failed",
        "Resource not found",
        "Permission denied",
        "Internal server error",
        "Service unavailable",
        "Rate limit exceeded"
    ]

    # Create start time (24 hours ago)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)

    logs = []

    for _ in range(num_logs):
        # Generate timestamp
        log_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
        timestamp = log_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Add milliseconds

        # Select module and environment
        module = random.choice(modules)
        env = random.choice(environments)

        # Select API endpoint
        endpoint = random.choice(api_endpoints)

        # Generate latency (usually low, occasionally high)
        latency = random.randint(10, 100)
        if random.random() < 0.1:  # 10% chance of high latency
            latency = random.randint(500, 2000)

        # Generate status code weighted by frequency
        status_code = random.choices(status_codes, weights=status_weights, k=1)[0]

        # Determine log level based on status code
        if status_code >= 500:
            level = "ERROR"
        elif status_code >= 400:
            level = random.choices(["WARNING", "ERROR"], weights=[70, 30], k=1)[0]
        else:
            level = random.choices(["INFO", "WARNING"], weights=[95, 5], k=1)[0]

        # Generate message content
        if level == "ERROR":
            message = f"Failed request {random.randint(1000,9999)} for {endpoint}: {random.choice(error_messages)}"
        elif level == "WARNING":
            message = f"Potential issue on {endpoint}: High latency detected ({latency}ms)" if latency > 500 else f"Client error scenario on {endpoint}"
        else:
            message = f"Request {random.randint(1000,9999)} processed successfully for {endpoint}"

        # Format as a more structured line resembling common log formats
        log_line = f"{timestamp} [{level}] [{module}] [{env}] {message} - status={status_code} latency={latency}ms request_id=req-{random.randint(10000, 99999)}"
        logs.append(log_line)

    # Add some specific patterns for clustering
    for _ in range(20):
        log_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
        timestamp = log_time.strftime("%Y-%m-%d %H:%M:%S")
        logs.append(f"{timestamp} [ERROR] [database] [production] Connection pool exhausted - reconnecting - Pool size: {random.randint(10, 50)}")

    for _ in range(15):
        log_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
        timestamp = log_time.strftime("%Y-%m-%d %H:%M:%S")
        logs.append(f"{timestamp} [WARN] [auth-service] [production] Rate limiting applied for IP {random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)} - Too many login attempts")

    # Shuffle the logs
    random.shuffle(logs)

    return logs


def fetch_logs_from_csv(csv_path):
    """Reads logs from a CSV file and returns them as a list of strings (header + rows)."""
    try:
        df = pd.read_csv(csv_path)
        
        # Format the logs for better display and analysis
        logs = []
        
        # Add the header as the first row
        logs.append(','.join(df.columns.tolist()))
        
        # Process each row in the dataframe
        for _, row in df.iterrows():
            # Format timestamp (replace T with space for better readability)
            timestamp = row['timestamp'].replace('T', ' ')
            
            # Determine log level based on error and status_code
            error = int(row['error'])
            status_code = int(row['status_code'])
            
            # Create formatted log entries that include more information
            if error == 1:
                level = "ERROR"
                # Create a detailed error message
                message = f"[{timestamp}] {row['api_id']} [{row['env']}]: Error detected - Status: {status_code} - Latency: {row['latency_ms']}ms - CPU: {row['simulated_cpu_cost']} - Memory: {row['simulated_memory_mb']}MB - Request: {row['request_id']}"
            elif status_code >= 400:
                level = "WARNING"
                message = f"[{timestamp}] {row['api_id']} [{row['env']}]: High status code - Status: {status_code} - Latency: {row['latency_ms']}ms - Bytes: {row['bytes_transferred']}"
            else:
                level = "INFO"
                message = f"[{timestamp}] {row['api_id']} [{row['env']}]: Request processed - Status: {status_code} - Latency: {row['latency_ms']}ms - Response time: {row['response_time']}ms"
            
            logs.append(message)
            
        return logs
    except Exception as e:
        print(f"Error reading sample logs: {str(e)}")
        # Fallback to generated logs if file can't be read
        return generate_sample_logs(300)