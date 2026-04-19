import os

USE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_USE_AUTOTUNE_CACHE', '1') == '1'
AUTOSAVE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE', '1') == '1'
USE_AUTOTUNE_RUNTIME = os.environ.get('FLEX_GEMM_USE_AUTOTUNE_RUNTIME', '1') == '1'
"Whether to run autotuning when the cache is missing."
AUTOTUNE_CACHE_PATH = os.environ.get(
    'FLEX_GEMM_AUTOTUNE_CACHE_PATH',
    os.path.expanduser('~/.flex_gemm/autotune_cache.json')
)

IS_CUDA_EXTENSION_AVAILABLE = None
"""Whether the CUDA extension is available. This is determined at runtime.
If CUDA extension is required but not available, consider re-installing flex_gemm [cuda] option to build the extension."""

USE_CUDA_EXTENSION = True
"Whether to use CUDA extension for hashmap-based neighbor map construction. Ignored if the extension is not available."

_USE_PYTORCH_FOR_TEST = False
"Internal debugging flag to indicate whether we are using the pure PyTorch implementation for reference testing. "
