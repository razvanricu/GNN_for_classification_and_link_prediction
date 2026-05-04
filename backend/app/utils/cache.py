from cachetools import LRUCache
from functools import wraps

from app.config import get_settings


def cached_prediction(maxsize: int = None):
    """LRU cache decorator for prediction functions."""
    def decorator(func):
        settings = get_settings()
        cache = LRUCache(maxsize=maxsize or settings.LRU_CACHE_SIZE)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key from string args
            key = (func.__name__,) + args[1:] + tuple(sorted(kwargs.items()))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        return wrapper
    return decorator
