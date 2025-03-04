from common.roll import Roll
from common.note_collection import NoteCollection

from generation.chord_progression import ChordProgression
from generation.instruments.sampled_instrument import (
    SampledInstrument,
    SampledInstrumentExportConfig,
)


class Voice(SampledInstrument):

    def __init__(self, parent: Roll, name: str):
        super().__init__(parent, name)

    @property
    def export_config(self) -> SampledInstrumentExportConfig:
        return SampledInstrumentExportConfig(
            sample_src="piranha_plant",
        )

    def generate(
        self,
        note_collection: NoteCollection,
        chord_progression: ChordProgression,
    ):
        self.notes.add(*note_collection.list())
