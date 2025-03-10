import numpy as np
from numpy.typing import NDArray


def mode(arr: NDArray[np.float16]) -> NDArray[np.float16] | None:
    int_arr = arr[~np.isnan(arr)].astype(np.int32)
    if int_arr.size == 0:
        return None
    counts = np.bincount(int_arr)
    max_count = np.max(counts)
    return np.argwhere(counts == max_count).flatten().astype(np.float16)
