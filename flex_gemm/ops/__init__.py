from . import grid_sample
from . import spconv
from . import serialize


class Backend:
    """Backend choices for sparse convolution."""
    TRITON = "triton"
    TILELANG = "tilelang"


BACKEND = Backend.TRITON  # Default backend


def set_backend(backend: Backend):
    global BACKEND
    BACKEND = backend
