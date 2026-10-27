"""Small decorator toolbox."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger("findex.timing")


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Log how long each call of ``fn`` took, at DEBUG level.

    ``@timed`` above ``def f`` is exactly ``f = timed(f)``. ``functools.wraps`` copies
    ``__name__``, ``__doc__``, ``__wrapped__`` ... from ``fn`` onto the wrapper; without it
    ``f.__name__`` would be ``"wrapper"``.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug("%s took %.1f ms", fn.__qualname__, (time.perf_counter() - t0) * 1000)

    return wrapper
