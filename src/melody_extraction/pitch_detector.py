from dataclasses import dataclass
import librosa
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from common.audio_data import AudioData


@dataclass(frozen=True)
class PitchDetectorConfig:
    offset_range: Tuple[float, float] = (-50, 50)
    offset_n: int = 7


class PitchDetector:
    @classmethod
    def _get_pitches(
        cls,
        audio: AudioData,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Predicts the pitches for the given audio.

        :param audio: The audio to predict the pitches for.
        :return: A tuple containing the time and the predicted pitches in Hz.
        """
        f0: NDArray[np.float64]
        _voiced_flag: NDArray[np.bool_]
        voiced_probs: NDArray[np.float64]
        t: NDArray[np.float64]

        f0, _voiced_flag, voiced_probs = librosa.pyin(  # type: ignore
            audio.array,
            fmin=librosa.note_to_hz("C3"),  # type: ignore
            fmax=librosa.note_to_hz("G6"),  # type: ignore
            sr=audio.sample_rate,
        )
        t = librosa.times_like(f0, sr=audio.sample_rate)  # type: ignore
        f0[voiced_probs < 0.5] = np.nan

        return t, f0

    @classmethod
    def _shift_pitch(
        cls, pitch: NDArray[np.float64], cents_offset: float
    ) -> NDArray[np.float64]:
        return pitch.copy() * (2 ** (cents_offset / 1200))

    @classmethod
    def _snap_hz(
        cls, pitch: NDArray[np.float64], shift_candidates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Snaps the given pitches to the closest tones, considering a set of constant shifts.

        :param pitch: The pitches to snap in Hz.
        :param shift_candidates: The shifts to consider in cents.
        :return: The snapped pitches in Hz.
        """
        snapped_pitches: NDArray[np.float64]

        # Compute all shifts.
        shifted_pitches = np.array(
            [cls._shift_pitch(pitch, shift) for shift in shift_candidates]
        )
        snapped_pitches = librosa.midi_to_hz(np.round(librosa.hz_to_midi(shifted_pitches)))  # type: ignore

        # Take the shift that minimizes the squared error.
        losses = np.nansum((snapped_pitches - shifted_pitches) ** 2, axis=1)
        best_shift = shift_candidates[np.argmin(losses)]

        return cls._shift_pitch(pitch, best_shift)

    @classmethod
    def detect(
        cls, audio: AudioData, config: PitchDetectorConfig
    ) -> Tuple[NDArray[np.float64], NDArray[np.float16]]:
        """
        Detects the pitches for the given audio. The pitches are quantized to MIDI notes.

        :param audio: The audio to detect the pitches for.
        :param config: The configuration for the pitch detector.
        :return: A tuple containing the time and the detected pitches in MIDI notes.
        """
        t, pitch = cls._get_pitches(audio)

        snapped_pitch = cls._snap_hz(
            pitch, np.linspace(*config.offset_range, config.offset_n)
        )

        pitch_midi = np.round(librosa.hz_to_midi(snapped_pitch))  # type: ignore

        return t, pitch_midi
