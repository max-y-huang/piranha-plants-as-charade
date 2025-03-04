from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from midiutil.MidiFile import MIDIFile  # type: ignore

import common.roll as roll
from common.note_collection import NoteCollection
from common.audio_data import AudioData
from common.audio_sample import (
    AudioSample,
    AudioSampleManager,
    AudioSampleManagerConfig,
)


@dataclass(frozen=True)
class InstrumentExportConfig:
    pass


class Instrument(ABC):

    def __init__(self, parent: roll.Roll, name: str):
        self._parent = parent
        self._name = name
        self._notes = NoteCollection()

    @property
    def name(self) -> str:
        return self._name

    @property
    def notes(self) -> NoteCollection:
        return self._notes

    @property
    @abstractmethod
    def export_config(self) -> InstrumentExportConfig:
        pass

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any):
        pass


@dataclass(frozen=True)
class MIDIInstrumentExportConfig(InstrumentExportConfig):
    instrument_id: int


class MIDIInstrument(Instrument):

    @property
    @abstractmethod
    def export_config(self) -> MIDIInstrumentExportConfig:
        pass

    def add_self_to_midi_track(self, midi: MIDIFile, track: int):
        time = 0  # start at the beginning
        channel = track
        instrument_id = self.export_config.instrument_id

        midi.addTrackName(track, time, self.name)  # type: ignore
        midi.addTempo(track, time, self._parent.beats_per_minute)  # type: ignore
        midi.addTimeSignature(  # type: ignore
            track,
            time,
            self._parent.beats_per_measure,
            math.floor(math.sqrt(self._parent.beat_duration)),
            24,
        )
        midi.addProgramChange(track, channel, time, instrument_id)  # type: ignore

        time_scale = self._parent.beat_duration / self._parent.quantization
        for note in self.notes.list():
            midi.addNote(  # type: ignore
                track,
                channel,
                note.pitch.value,
                note.start * time_scale,
                note.duration * time_scale,
                100,
            )


@dataclass(frozen=True)
class SampledInstrumentExportConfig(InstrumentExportConfig):
    sample_src: str


class SampledInstrument(Instrument):

    @property
    @abstractmethod
    def export_config(self) -> SampledInstrumentExportConfig:
        pass

    def generate_audio_from_samples(self, config: roll.RollExportConfig) -> AudioData:
        def to_sample_time(time: float) -> int:
            m = config.sample_rate * 60 / self._parent.beats_per_minute
            m *= self._parent.beat_duration / self._parent.quantization
            return int(m * time)

        def get_shift_size(sample: AudioSample) -> int:
            return config.start_padding_size + sample.timbre_properties.start_shift

        sample_manager = AudioSampleManager(
            AudioSampleManagerConfig(src=self.export_config.sample_src)
        )

        audio_data = AudioData()
        for note in self.notes.list():
            timbre = sample_manager.get_random_timbre()
            sample = sample_manager.get_sample(timbre, note.pitch)
            note_len = min(len(sample.audio.array), to_sample_time(note.duration))
            start_time = to_sample_time(note.start) + get_shift_size(sample)
            note_range = (start_time, start_time + note_len)

            audio_data.add_range(
                note_range,
                sample.audio.slice(0, note_len).array
                * sample.timbre_properties.get_envelope(note_len),
            )

        return audio_data
