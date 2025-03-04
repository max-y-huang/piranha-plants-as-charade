from typing import FrozenSet

from common.roll import Roll
from common.structures.chord import Chord
from common.structures.note import Note
from common.structures.pitch import Pitch
from common.structures.interval import Interval

from generation.chord_progression import ChordProgression
from generation.instruments.base import (
    MIDIInstrument,
    MIDIInstrumentExportConfig,
)

from .chord_voicer import PianoChordVoicer


class Piano(MIDIInstrument):

    def __init__(self, parent: Roll, name: str):
        super().__init__(parent, name)

    @property
    def export_config(self) -> MIDIInstrumentExportConfig:
        return MIDIInstrumentExportConfig(
            instrument_id=1,  # acoustic grand
        )

    def generate(self, chord_progression: ChordProgression):
        chord_voicings = PianoChordVoicer.generate(chord_progression)
        for i, chord in enumerate(chord_progression.chords):
            self._add_stride_pattern(
                chord.chord,
                chord_voicings[i],
                chord.start_time,
                chord.end_time,
            )

    def _add_stride_pattern(
        self,
        chord: Chord,
        chord_voicing: FrozenSet[Pitch],
        start: int,
        end: int,
    ):
        for i, time in enumerate(range(start, end)):
            if i % 8 == 0:
                self.notes.add(
                    Note(
                        pitch=chord.root,
                        start=time,
                        duration=self._parent.Duration(1 / 2),
                    ).reoctave_near_pitch(Pitch.from_str("C3"))
                )
            elif i % 8 == 4:
                prev_pitch_set = self.notes.get_pitches_at_time(
                    time - self._parent.Duration(1)
                )
                self.notes.add(
                    Note(
                        pitch=chord.root + Interval.from_str("5"),
                        start=time,
                        duration=self._parent.Duration(1 / 2),
                    ).reoctave_near_pitch(
                        list(prev_pitch_set)[0],
                        position="below",
                    )
                )
            elif i % 4 == 2:
                self.notes.add(
                    *[
                        Note(
                            pitch=pitch,
                            start=time,
                            duration=self._parent.Duration(1 / 4),
                        )
                        for pitch in chord_voicing
                    ]
                )
