from .triton import *
from .. import config as pkg_config
try:
	from . import cuda
	from .cuda import *
	if hasattr(cuda, "hashmap_build_submanifold_conv_neighbour_map_cuda"):	
		pkg_config.IS_CUDA_EXTENSION_AVAILABLE = True
	else:
		pkg_config.USE_CUDA_EXTENSION = pkg_config.IS_CUDA_EXTENSION_AVAILABLE = False
except ImportError:
	pkg_config.USE_CUDA_EXTENSION = pkg_config.IS_CUDA_EXTENSION_AVAILABLE = False