import re
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

IMPORT_RE = re.compile(r"^[a-zA-Z_][\w\.]*:[a-zA-Z_][\w]*$")


# Please write decorator once. It caches the return value of function on the first call
# and always returns it independant of arguments.
# Decorated object must also have method release which releases cached value.
T = TypeVar("T")


def once(f: Callable[..., T]) -> Callable[..., T]:
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if not hasattr(wrapper, "_cached_value"):
            wrapper._cached_value = f(*args, **kwargs)
        return wrapper._cached_value

    def release() -> None:
        if hasattr(wrapper, "_cached_value"):
            del wrapper._cached_value

    wrapper.release = release
    return wrapper


def asynconce(f: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    @wraps(f)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        if not hasattr(wrapper, "_cached_value"):
            wrapper._cached_value = await f(*args, **kwargs)
        return wrapper._cached_value

    def release() -> None:
        if hasattr(wrapper, "_cached_value"):
            del wrapper._cached_value

    wrapper.release = release
    return wrapper
