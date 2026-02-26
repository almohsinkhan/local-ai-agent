import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

try:
    from langsmith import traceable as _langsmith_traceable
except Exception:
    _langsmith_traceable = None


def _env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def langsmith_traceable(
    *,
    name: str | None = None,
    run_type: str = "chain",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if _langsmith_traceable is None:
        def _identity(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return _identity
    return _langsmith_traceable(name=name, run_type=run_type)


def timed(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        label = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _env_true("ENABLE_TIME_TRACKING", "true"):
                return func(*args, **kwargs)

            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                print(f"[timing] {label}: {elapsed_ms:.2f} ms")

        return wrapper

    return decorator


@contextmanager
def timed_block(name: str):
    if not _env_true("ENABLE_TIME_TRACKING", "true"):
        yield
        return

    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        print(f"[timing] {name}: {elapsed_ms:.2f} ms")
