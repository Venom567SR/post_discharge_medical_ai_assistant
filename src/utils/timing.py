"""Performance timing utilities for monitoring system operations."""

import time
from functools import wraps
from typing import Callable, Any
from loguru import logger


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", log: bool = True):
        self.name = name
        self.log = log
        self.start_time = None
        self.elapsed_ms = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        end_time = time.perf_counter()
        self.elapsed_ms = (end_time - self.start_time) * 1000
        
        if self.log:
            logger.info(f"{self.name} took {self.elapsed_ms:.2f}ms")
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ms if self.elapsed_ms is not None else 0.0


def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Usage:
        @timed
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        logger.info(f"{func.__name__} executed in {elapsed_ms:.2f}ms")
        return result
    
    return wrapper


def async_timed(func: Callable) -> Callable:
    """
    Decorator to time async function execution.
    
    Usage:
        @async_timed
        async def my_async_function():
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        logger.info(f"{func.__name__} (async) executed in {elapsed_ms:.2f}ms")
        return result
    
    return wrapper


class OperationTimer:
    """Track multiple operation timings."""
    
    def __init__(self):
        self.timings: dict[str, list[float]] = {}
    
    def record(self, operation: str, duration_ms: float):
        """Record a timing for an operation."""
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration_ms)
    
    def get_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.timings or not self.timings[operation]:
            return {}
        
        timings = self.timings[operation]
        return {
            "count": len(timings),
            "total_ms": sum(timings),
            "avg_ms": sum(timings) / len(timings),
            "min_ms": min(timings),
            "max_ms": max(timings)
        }
    
    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all operations."""
        return {
            operation: self.get_stats(operation)
            for operation in self.timings.keys()
        }
    
    def reset(self):
        """Clear all timings."""
        self.timings.clear()


# Global timer instance for tracking operations
global_timer = OperationTimer()
