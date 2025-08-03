# Create legal_assistant/services/metrics.py
from datetime import datetime
import time

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "processing_times": []
        }
    
    def record_request(self, endpoint: str, duration: float, status: int):
        self.metrics["requests"] += 1
        if status >= 400:
            self.metrics["errors"] += 1
        self.metrics["processing_times"].append(duration)
