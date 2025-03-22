from common.note_collection import NoteCollection
from common.structures.note import Note
from common.structures.pitch import Pitch
from common.structures.interval import Interval

from generation.chord_progression import ChordProgression

from instruments.base import (
    SampledInstrument,
    SampledInstrumentExportConfig,
)

from export.arrangement import ArrangementMetadata
from export.part import SampledPart


class VoiceHarmony(SampledInstrument):

    def __init__(self, name: str):
        super().__init__(name)

    @property
    def export_config(self) -> SampledInstrumentExportConfig:
        return SampledInstrumentExportConfig(
            name="piranha_plant",
            db=-7.5,
        )

    def generate(
        self,
        melody: NoteCollection,
        chord_progression: ChordProgression,
        arrangement_metadata: ArrangementMetadata,
    ) -> SampledPart:
        # TODO: Make the voice generation better.
        target_pitch = Pitch(0)
        scale = [
            (
                arrangement_metadata.key + Interval.from_str(str(interval))
            ).reoctave_near_pitch(target_pitch, position="above")
            for interval in range(1, 8)
        ]
        notes = NoteCollection()
        for note in melody.list():
            candidate_a = note.pitch - Interval.from_str("b3")
            candidate_a_reoctaved = candidate_a.reoctave_near_pitch(
                target_pitch,
                position="above",
            )
            candidate_b = note.pitch - Interval.from_str("3")
            if candidate_a_reoctaved in scale:
                pitch = candidate_a
            else:
                pitch = candidate_b
            notes.add(Note(pitch, note.start, note.duration))
        return SampledPart(arrangement_metadata, self, notes)
