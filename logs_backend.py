# logs_backend.py

def fetch_logs():
    return [
        "[2025-04-01 12:32:45] ERROR auth: Invalid token for user admin",
        "[2025-04-01 12:33:02] WARNING database: Slow query detected",
        "[2025-04-01 12:33:30] INFO server: Started on port 8080",
        "[2025-04-01 12:34:01] ERROR auth: User admin failed login 3 times",
        "[2025-04-01 12:35:12] INFO monitor: CPU usage at 85%",
        "[2025-04-01 12:35:45] ERROR cache: Redis timeout",
        "[2025-04-01 12:36:22] WARNING api: Request took 9s",
        "[2025-04-01 12:37:30] ERROR database: Connection pool exhausted",
        "[2025-04-01 12:38:45] INFO scheduler: Job completed",
        "[2025-04-01 12:39:00] ERROR disk: Disk space low",
        "[2025-04-01 12:39:20] WARNING config: Deprecated setting used"
    ]
