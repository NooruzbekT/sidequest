"""In-memory метрики сервиса: счётчики запросов/ошибок, latency, распределение моделей.

Достаточно для одного процесса API (требование ТЗ — живая страница статистики);
при горизонтальном масштабировании заменяется Prometheus-клиентом.
"""

from collections import Counter, deque

import numpy as np

_recommendation_requests = 0
_errors = 0
_served_by: Counter = Counter()
_latencies_ms: deque = deque(maxlen=1000)


def record_recommendation(model_name: str, model_version: str, latency_ms: float) -> None:
    global _recommendation_requests
    _recommendation_requests += 1
    _served_by[f"{model_name} {model_version}"] += 1
    _latencies_ms.append(latency_ms)


def record_error() -> None:
    global _errors
    _errors += 1


def snapshot() -> dict:
    lat = np.array(_latencies_ms) if _latencies_ms else np.array([0.0])
    return {
        "recommendation_requests": _recommendation_requests,
        "errors": _errors,
        "latency_ms_mean": round(float(lat.mean()), 2),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
        "served_by": dict(_served_by),
    }
