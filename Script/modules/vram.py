import ctypes

_libcudart = None


def _get_libcudart():
    global _libcudart
    if _libcudart is None:
        _libcudart = ctypes.CDLL("libcudart.so.12")
    return _libcudart


def get_vram_usage_mb() -> tuple[float, float]:
    """(used_mb, total_mb) を返す。Jetson の統合メモリ対応版。"""
    try:
        lib = _get_libcudart()
        free  = ctypes.c_size_t()
        total = ctypes.c_size_t()
        lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        used_mb  = (total.value - free.value) / 1024 ** 2
        total_mb = total.value / 1024 ** 2
        return used_mb, total_mb
    except Exception:
        return 0.0, 0.0
