import importlib


__imported = {}


def __getattr__(name):
    if name in __imported:
        return __imported[name]
    else:
        try:
            mod = importlib.import_module('.'+name, __name__)
            __imported[name] = mod
            return mod
        except ImportError:
            raise AttributeError(f"module {__name__} has no attribute {name}")


if __name__ == "__main__":
    from . import triton
    from . import tilelang
    from . import cuda
