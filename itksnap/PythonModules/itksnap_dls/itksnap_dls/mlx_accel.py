"""
MLX acceleration layer for pre/post-processing on Apple Silicon.

Provides MLX-accelerated versions of array operations used in the server's
encode/decode pipeline. Falls back transparently to NumPy when MLX is not
available (non-Apple-Silicon platforms or MLX not installed).
"""

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def threshold_result(arr: np.ndarray) -> np.ndarray:
    """Threshold segmentation result: values > 0 become 1, else 0, as int8.

    On Apple Silicon with MLX, this runs on the unified-memory GPU.
    Falls back to NumPy otherwise.
    """
    if HAS_MLX:
        mx_arr = mx.array(arr)
        mx_result = mx.where(mx_arr > 0, 1, 0).astype(mx.int8)
        return np.array(mx_result, copy=False)
    else:
        return np.where(arr > 0, 1, 0).astype(np.int8)


def decode_image_array(raw_bytes: bytes, dtype, shape: tuple) -> np.ndarray:
    """Decode raw bytes into a shaped numpy array.

    On Apple Silicon with MLX, uses MLX for the reshape/type conversion.
    Falls back to NumPy otherwise.
    """
    array = np.frombuffer(raw_bytes, dtype=dtype)
    if HAS_MLX:
        mx_arr = mx.array(array).reshape(shape)
        return np.array(mx_arr, copy=False)
    else:
        return array.reshape(shape)
