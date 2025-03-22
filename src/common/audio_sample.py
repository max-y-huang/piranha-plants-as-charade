import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from numpy.typing import NDArray
from functools import cache

from common.util import is_wav_media_type
from common.audio_data import AudioData
from common.structures.pitch import Pitch

from env import load_env


SAMPLES_DIR = "../data/samples"


@dataclass(frozen=True)
class AudioSampleCollectionConfig:
    """
    :param name: The folder in which the sample file reside (i.e. `data/<name>`), with each file consisting of chromatic ascending notes of the same timbre.
    :param sample_rate: The sample rate at which to load each sample file.
    :param range: The range supported.
    :param beats_per_minute: The tempo at which each note is 1 beat.
    """

    name: str
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


class AudioSampleCollection:

    def __init__(self, config: AudioSampleCollectionConfig):
        self._sample_data: Dict[Tuple[str, Pitch], AudioSample] = dict()
        self._timbre_data: Dict[str, AudioSampleTimbreProperties] = dict()
        self._config = config
        for timbre_file in os.listdir(os.path.join(SAMPLES_DIR, self._config.name)):
            try:
                self._load_timbre_file(
                    os.path.join(SAMPLES_DIR, self._config.name, timbre_file)
                )
            except SkipFileOnSampleLoad:
                pass
            except:
                print(f"Failed to load {timbre_file}.")
            else:
                print(f"Loaded {timbre_file}.")

    def _load_timbre_file(self, path: str):
        dir, file_name = os.path.split(path)
        timbre = os.path.splitext(file_name)[0]
        if not is_wav_media_type(path):
            raise SkipFileOnSampleLoad()
        timbre_properties: AudioSampleTimbreProperties = load_env(
            AudioSampleTimbreProperties,
            os.path.join(dir, f"{timbre}.timbre"),
        )
        self._timbre_data[timbre] = timbre_properties
        # throws an exception if load failed
        audio = AudioData.from_file(
            path,
            sample_rate=self.sample_rate,
            db=timbre_properties.db,
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

    def get_random_timbre(self, seed: Optional[int] = None) -> str:
        random.seed(seed)
        return random.choice(list(self._timbre_data.keys()))


AUDIO_SAMPLE_LIBRARY = {
    sample_name: AudioSampleCollection(
        load_env(
            AudioSampleCollectionConfig,
            os.path.join(SAMPLES_DIR, sample_name, "config"),
            default_args={"name": sample_name},
        ),
    )
    for sample_name in os.listdir(SAMPLES_DIR)
    if os.path.isdir(os.path.join(SAMPLES_DIR, sample_name))
}
print(f"Loaded audio samples from {list(AUDIO_SAMPLE_LIBRARY.keys())}.")
