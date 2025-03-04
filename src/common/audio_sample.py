import os
import random
import librosa
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from numpy.typing import NDArray
from dotenv import dotenv_values
from functools import cache

from common.conversions import db_to_strength
from common.audio_data import AudioData
from common.structures.pitch import Pitch


@dataclass(frozen=True)
class AudioSampleManagerConfig:
    """
    :param src: The folder in which the sample file reside (i.e. `data/<src>`), with each file consisting of chromatic ascending notes of the same timbre.
    :param sample_rate: The sample rate at which to load each sample file.
    :param range: The range supported.
    :param beats_per_minute: The tempo at which each note is 1 beat.
    """

    src: str
    sample_rate: int = 44100
    range: Tuple[Pitch, Pitch] = (
        Pitch.from_str("C3"),
        Pitch.from_str("G6"),
    )
    beats_per_minute: int = 60


@dataclass(frozen=True)
class AudioSampleTimbreProperties:
    """
    :param start_shift: The number of indices to shift the audio sample by in the positive time direction.
    :param start_sample_idx: The index of the start of the audio sample.
    :param end_sample_idx: The index of the end of the audio sample.
    :param ease_in_factor: The envelope's fade-in duration relative to sample's entire duration.
    :param ease_out_factor: The envelope's fade-out duration relative to sample's entire duration.
    """

    start_shift: int = 0
    start_sample_idx: int = 0
    end_sample_idx: int = 38000
    ease_in_factor: float = 0
    ease_out_factor: float = 0.1
    db: float = 0

    @cache
    def get_envelope(self, num_samples: int) -> NDArray[np.float32]:
        num_ease_in_samples = int(num_samples * self.ease_in_factor)
        num_ease_out_samples = int(num_samples * self.ease_out_factor)
        # Creates a window that starts at 0, ramps up to 1, stays at 1, ramps down to 0.
        window_start = np.hamming(num_ease_in_samples * 2)[:num_ease_in_samples]
        window_end = np.hamming(num_ease_out_samples * 2)[num_ease_out_samples:]
        window_middle = np.ones(num_samples - len(window_start) - len(window_end))
        return np.concatenate(
            [window_start, window_middle, window_end],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class AudioSample:
    audio: AudioData
    timbre_properties: AudioSampleTimbreProperties


class SkipFileOnSampleLoad(Exception):
    pass


class AudioSampleManager:

    def __init__(self, config: AudioSampleManagerConfig):
        self._sample_data: Dict[Tuple[str, Pitch], AudioSample] = dict()
        self._timbre_data: Dict[str, AudioSampleTimbreProperties] = dict()
        self._config = config
        for timbre_file in os.listdir(self._samples_dir):
            try:
                self._load_file(os.path.join(self._samples_dir, timbre_file))
            except SkipFileOnSampleLoad:
                pass
            except:
                print(f"Failed to load {timbre_file}.")
            else:
                print(f"Loaded {timbre_file}.")

    @property
    def _samples_dir(self) -> str:
        return os.path.join("../data/samples", self._config.src)

    def _load_timbre_properties(self, timbre: str) -> AudioSampleTimbreProperties:
        settings: Dict[str, Any] = dict()
        try:
            arg_types = AudioSampleTimbreProperties.__annotations__  # { <ARG>: <TYPE> }
            path = os.path.join(self._samples_dir, f"{timbre}.timbre")
            for arg, val in dotenv_values(path).items():
                if arg in arg_types:
                    settings[arg] = arg_types[arg](val)
        except:
            pass
        return AudioSampleTimbreProperties(**settings)

    def _load_file(self, path: str):
        timbre, extension = os.path.basename(path).split(".")
        if extension != "wav":
            raise SkipFileOnSampleLoad()
        timbre_properties = self._load_timbre_properties(timbre)
        self._timbre_data[timbre] = timbre_properties
        # throws an exception if load failed
        audio = AudioData(
            librosa.load(  # type: ignore
                path,
                sr=self.sample_rate,
                dtype=np.float32,
            )[0]
            * db_to_strength(timbre_properties.db)
        )

        def splice_audio(index: int) -> AudioData:
            def to_sample_time(position: float) -> int:
                m = self.sample_rate * 60 / self._config.beats_per_minute
                return int(m * position)

            sample_time = to_sample_time(index)
            return audio.slice(
                sample_time + timbre_properties.start_sample_idx,
                sample_time + timbre_properties.end_sample_idx,
            )

        for i, pitch_value in enumerate(
            range(self._config.range[0].value, self._config.range[1].value + 1)
        ):
            self._sample_data[(timbre, Pitch(pitch_value))] = AudioSample(
                audio=splice_audio(i),
                timbre_properties=timbre_properties,
            )

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    def get_sample(self, timbre: str, pitch: Pitch) -> AudioSample:
        return self._sample_data[(timbre, Pitch(pitch.value))]

    def get_random_timbre(self) -> str:
        return random.choice(list(self._timbre_data.keys()))
