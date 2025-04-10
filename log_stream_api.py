# --- START OF FILE log_stream_api.py ---

import uvicorn  # Import uvicorn for running the app
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware # To allow requests from Streamlit frontend
import asyncio
import random
from datetime import datetime
import json
import time

# --- FastAPI App Setup ---
app = FastAPI(
    title="Real-time Log Simulator API",
    description="Streams simulated logs in JSON format via Server-Sent Events (SSE).",
    version="1.1.0" # Version bump
)

# --- CORS Middleware ---
# Allow requests from your Streamlit app's origin (adjust if needed)
# Use "*" for development, but be more specific in production.
origins = [
    "http://localhost",
    "http://localhost:8501", # Default Streamlit port
    # Add the origin where your Streamlit app is deployed if applicable
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Log Generation Parameters ---
LOG_LEVELS = ["INFO", "WARNING", "ERROR", "DEBUG"]
LEVEL_WEIGHTS = [70, 15, 10, 5] # Weighted probabilities
COMPONENTS = ["auth-service", "api-gateway", "database", "cache", "user-service", "payment-processor", "simulation"]
COMPONENT_WEIGHTS = [15, 15, 20, 10, 15, 10, 15]
LOG_CODES = ["REQ_START", "REQ_END", "DB_QUERY", "AUTH_SUCCESS", "AUTH_FAIL", "CACHE_HIT", "CACHE_MISS", "ERROR_500", "VALIDATION_ERR", "PROCESSING"]
STATUS_CODES_SUCCESS = [200, 201, 204]
STATUS_CODES_CLIENT_ERR = [400, 401, 403, 404, 429]
STATUS_CODES_SERVER_ERR = [500, 502, 503, 504]
ALL_STATUS_CODES = STATUS_CODES_SUCCESS + STATUS_CODES_CLIENT_ERR + STATUS_CODES_SERVER_ERR
STATUS_WEIGHTS = [60]*len(STATUS_CODES_SUCCESS) + [5]*len(STATUS_CODES_CLIENT_ERR) + [2]*len(STATUS_CODES_SERVER_ERR) # Adjust weights as needed

# --- Log Generation Function ---
def generate_log_entry() -> dict:
    """Generates a single simulated log entry as a dictionary."""
    # Use microsecond precision for more realistic timestamps if needed, but stick to seconds for compatibility with example
    # timestamp = datetime.now().isoformat(timespec='microseconds')
    timestamp = datetime.now().isoformat(timespec='seconds') # Match example format
    log_level = random.choices(LOG_LEVELS, weights=LEVEL_WEIGHTS, k=1)[0]
    component = random.choices(COMPONENTS, weights=COMPONENT_WEIGHTS, k=1)[0]
    log_code = random.choice(LOG_CODES)
    process_id = random.randint(10000, 30000)

    # Determine status code based on level somewhat
    if log_level == "ERROR":
        status_code = random.choice(STATUS_CODES_SERVER_ERR + STATUS_CODES_CLIENT_ERR)
    elif log_level == "WARNING":
        # Warnings can occur even on success (e.g., high latency) or client errors
        status_code = random.choice(STATUS_CODES_CLIENT_ERR + STATUS_CODES_SUCCESS + [200, 200]) # Skew warnings towards client errors/success
    else: # INFO or DEBUG
        status_code = random.choices(ALL_STATUS_CODES, weights=STATUS_WEIGHTS, k=1)[0]

    # Simulate response time (seconds) - higher for errors/warnings potentially
    base_time = random.uniform(0.01, 0.5)
    if log_level == "ERROR":
        response_time = base_time + random.uniform(0.5, 2.0)
    elif log_level == "WARNING":
        response_time = base_time + random.uniform(0.1, 0.8)
    else:
        response_time = base_time
    response_time = round(response_time, 6) # Keep precision

    # Simulate CPU usage
    cpu_usage = round(random.uniform(0.0, 15.0) + (response_time * 10), 2) # Loosely related to response time

    # Create a descriptive message
    message = f"Log event: {log_code} from {component}."
    if log_code.startswith("REQ"):
        message = f"Request {random.randint(100,999)} for {component} {log_code.split('_')[1].lower()}."
    elif "DB" in log_code:
        message = f"Database query execution: {random.choice(['SELECT', 'UPDATE', 'INSERT'])} on table {random.choice(['users', 'orders', 'products'])}."
    elif "AUTH" in log_code:
        message = f"Authentication attempt {log_code.split('_')[1].lower()} for user {random.randint(1000, 2000)}."
    elif log_level == "ERROR":
        message = f"Error condition {status_code} encountered in {component}: {random.choice(['Timeout', 'Connection Refused', 'Resource Limit Exceeded', 'Null Pointer Exception'])}."
    elif log_level == "WARNING" and status_code >= 400 :
         message = f"Potential issue detected during {log_code} in {component}: Status {status_code}."
    elif log_level == "WARNING" and response_time > 0.5 :
         message = f"High response time ({response_time:.3f}s) observed for {log_code} in {component}."


    log_entry = {
        "timestamp": timestamp,         # String, ISO format seconds precision
        "log_level": log_level,         # String (INFO, WARNING, ERROR, DEBUG)
        "components": component,        # String (service name) -> Mapped to 'module' in app
        "log_code": log_code,           # String (internal code)
        "status_code": status_code,     # Integer (HTTP status or similar)
        "process_id": process_id,       # Integer
        "message": message,             # String (main log message)
        "response_time": response_time, # Float (seconds) -> Converted to ms latency in app
        "cpu_usage": cpu_usage,         # Float (percentage or similar)
        "request_id": f"req-{random.randint(100000, 999999)}", # String
        "environment": random.choice(["production", "staging", "development"]) # String -> Mapped to 'env' in app
    }
    return log_entry

# --- SSE Streaming Endpoint ---
@app.get("/stream-logs")
async def stream_logs(request: Request):
    """
    Endpoint to stream simulated logs using Server-Sent Events (SSE).
    Sends logs in the format: `data: {"key": "value", ...}\n\n`
    """
    async def event_generator():
        log_counter = 0
        start_time = time.time()
        try:
            while True:
                # Check if client is still connected before generating/sending
                if await request.is_disconnected():
                    print(f"Client disconnected after {log_counter} logs sent in {time.time() - start_time:.2f} seconds.")
                    break

                # Generate a log entry
                log_entry = generate_log_entry()

                # Convert dict to JSON string
                json_log = json.dumps(log_entry)

                # Format for SSE: "data: <json_string>\n\n"
                sse_formatted_log = f"data: {json_log}\n\n"

                yield sse_formatted_log
                log_counter += 1

                # Simulate varying time between logs (e.g., 10ms to 200ms)
                await asyncio.sleep(random.uniform(0.01, 0.2))

        except asyncio.CancelledError:
            print(f"Log streaming task cancelled after {log_counter} logs sent.")
        except Exception as e:
            print(f"Error during log streaming: {e}")
        finally:
            print("Log stream generator finished.")

    # Return the streaming response
    # Headers are important for SSE
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no", # Useful for Nginx proxying
    }
    return StreamingResponse(event_generator(), headers=headers)

# --- Root Endpoint (Optional) ---
@app.get("/")
async def root():
    return {"message": "Log Simulator API is running. Access /stream-logs for SSE stream."}

# --- Main Execution ---
if __name__ == "__main__":
    # Run the FastAPI app using uvicorn
    # You can change host and port as needed
    # reload=True is useful for development
    print("Starting Log Simulator API on http://0.0.0.0:8000")
    uvicorn.run("log_stream_api:app", host="0.0.0.0", port=8000, reload=True)
# --- END OF FILE log_stream_api.py ---
