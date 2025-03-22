from __future__ import annotations

import math
from midiutil.MidiFile import MIDIFile  # type: ignore

from common.util import db_to_strength
from common.note_collection import NoteCollection
from common.audio_data import AudioData
from common.audio_sample import (
    AudioSample,
    AUDIO_SAMPLE_LIBRARY,
)

import instruments.base as instruments

import export.arrangement as arrangement


class Part:

    def __init__(
        self,
        arrangement_metadata: arrangement.ArrangementMetadata,
        instrument: instruments.Instrument,
        notes: NoteCollection,
    ):
        self._arrangement_metadata = arrangement_metadata
        self._instrument = instrument
        self._notes = notes

    @property
    def notes(self):
        return self._notes


class MIDIPart(Part):

    def __init__(
        self,
        arrangement_metadata: arrangement.ArrangementMetadata,
        instrument: instruments.MIDIInstrument,
        notes: NoteCollection,
    ):
        super().__init__(arrangement_metadata, instrument, notes)

    def add_notes_to_track(self, midi: MIDIFile, track: int):
        assert isinstance(self._instrument, instruments.MIDIInstrument)

        time = 0  # start at the beginning
        channel = self._instrument.export_config.channel
        instrument_id = self._instrument.export_config.instrument_id
        volume = self._instrument.export_config.volume
        time_signature = (
            self._arrangement_metadata.beats_per_measure,
            round(
                math.log2(self._arrangement_metadata.beat_duration)
            ),  # expressed as a power of 2
        )

        midi.addTrackName(track, time, self._instrument.name)  # type: ignore
        midi.addTempo(track, time, self._arrangement_metadata.beats_per_minute)  # type: ignore
        midi.addTimeSignature(  # type: ignore
            track,
            time,
            time_signature[0],
            time_signature[1],
            24,  # metronome tick per quarter note
        )
        midi.addProgramChange(track, channel, time, instrument_id)  # type: ignore

        time_scale = (
            self._arrangement_metadata.beat_duration
            / self._arrangement_metadata.quantization
        )
        for note in self.notes.list():
            midi.addNote(  # type: ignore
                track,
                channel,
                note.pitch.value,
                note.start * time_scale,
                note.duration * time_scale,
                volume,
            )


class SampledPart(Part):

    def __init__(
        self,
        arrangement_metadata: arrangement.ArrangementMetadata,
        instrument: instruments.SampledInstrument,
        notes: NoteCollection,
    ):
        super().__init__(arrangement_metadata, instrument, notes)

    def get_audio_data(self, config: arrangement.ArrangementExportConfig) -> AudioData:
        assert isinstance(self._instrument, instruments.SampledInstrument)

        def to_sample_time(time: float) -> int:
            m = config.sample_rate * 60 / self._arrangement_metadata.beats_per_minute
            m *= (
                self._arrangement_metadata.beat_duration
                / self._arrangement_metadata.quantization
            )
            return int(m * time)

        def get_shift_size(sample: AudioSample) -> int:
            return config.start_padding_size + sample.timbre_properties.start_shift

        sample_manager = AUDIO_SAMPLE_LIBRARY[self._instrument.export_config.name]

        audio_data = AudioData()
        for i, note in enumerate(self.notes.list()):
            timbre = sample_manager.get_random_timbre(i)
            sample = sample_manager.get_sample(timbre, note.pitch)
            note_len = min(len(sample.audio.array), to_sample_time(note.duration))
            start_time = to_sample_time(note.start) + get_shift_size(sample)
            note_range = (start_time, start_time + note_len)

            audio_data.add_range(
                note_range,
                sample.audio.slice(0, note_len).array
                * db_to_strength(self._instrument.export_config.db)
                * sample.timbre_properties.get_envelope(note_len),
            )

        return audio_data
