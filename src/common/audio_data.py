import librosa
import numpy as np
from typing import DefaultDict, Dict, Tuple
from numpy.typing import NDArray
from collections import defaultdict
from functools import cached_property

from common import db_to_strength


class AudioData:

    def __init__(
        self,
        init_data: NDArray[np.float32] = np.array([]),
        sample_rate: float = 44100,
    ):
        self._data: DefaultDict[int, np.float32] = defaultdict(lambda: np.float32(0))
        self._sample_rate: float = sample_rate
        self.set_range((0, len(init_data)), init_data)

    @classmethod
    def from_file(cls, file_name: str, sample_rate: (float | None) = None, db: float = 0):
        signal: NDArray[np.float32]

        signal, sr = librosa.load(  # type: ignore
            file_name,
            sr=sample_rate,
            dtype=np.float32,
        )

        signal *= db_to_strength(db)
        return cls(signal, sr)

    @cached_property
    def array(self) -> NDArray[np.float32]:
        keys = self._data.keys()
        if len(keys) == 0:
            return np.array([])
        return np.array([self._data[i] for i in range(max(keys) + 1)])

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def slice(self, start: int, end: int) -> "AudioData":
        return AudioData(self.array[start:end])

    def at(self, idx: int) -> np.float32:
        return self._data[idx]

    def set(self, idx: int, val: np.float32):
        assert idx >= 0
        self._data[idx] = val
        self._clear_array_cache()

    def set_range(self, r: Tuple[int, int], vals: NDArray[np.float32]):
        assert r[1] - r[0] == len(vals)
        update = {r[0] + i: v for i, v in enumerate(vals)}
        self._data.update(update)
        self._clear_array_cache()

    def add(self, idx: int, val: np.float32):
        self.set(idx, self.at(idx) + val)

    def add_range(self, r: Tuple[int, int], vals: NDArray[np.float32]):
        assert r[1] - r[0] == len(vals)
        self.set_range(r, np.array([self.at(i) for i in range(*r)]) + vals)

    def pad_start(self, n: int):
        new_dict: Dict[int, np.float32] = dict()
        for key, val in self._data.items():
            new_dict[key + n] = val
        self._data = defaultdict(lambda: np.float32(0), new_dict)
        self._clear_array_cache()

    def _clear_array_cache(self):
        try:
            del self.array
        except:
            pass
