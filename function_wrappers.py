from functools import wraps
from typing import Callable, Any

class VersionedCallable:
    version: str
    description: str

    def __call__(self, *args, **kwargs) -> Any:
        pass

def versioned_function(version: str, description: str = ""):
    """Function versioning decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        
        class VersionedWrapper(VersionedCallable):
            def __call__(self, *args, **kwargs) -> Any:
                return wrapper(*args, **kwargs)

        versioned_wrapper = VersionedWrapper()
        versioned_wrapper.version = version
        versioned_wrapper.description = description
        return versioned_wrapper
    return decorator