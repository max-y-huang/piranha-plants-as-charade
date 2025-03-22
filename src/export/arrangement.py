from __future__ import annotations

import os
import tempfile
import scipy.io.wavfile as wav  # type: ignore
from dataclasses import dataclass
from typing import Tuple, List
from midiutil.MidiFile import MIDIFile  # type: ignore

from common.util import db_to_strength
from common.audio_data import AudioData
from common.structures.pitch import Pitch

import export.part as part


@dataclass(frozen=True)
class ArrangementMetadata:
    """
    :param beats_per_minute: The beats per minute in terms of the time signature beat.
    :param quantization: The minimum unit of time (e.g. quantization = 16 means quantize by 16th notes)
    :param time_signature: The time signature of the song.
    :param key: The key of the song. Assumed to be major.
    """

    beats_per_minute: int
    time_signature: Tuple[int, int]
    key: Pitch = Pitch.from_str("C")
    quantization: int = 16

    @property
    def beats_per_measure(self) -> int:
        return self.time_signature[0]

    @property
    def beat_duration(self) -> int:
        return self.time_signature[1]

    def Duration(self, duration: float) -> int:
        return round(duration / self.beat_duration * self.quantization)

    def Time(self, measure: int, beat: float) -> int:
        return self.Duration((measure * self.beats_per_measure + beat))


@dataclass(frozen=True)
class ArrangementExportConfig:
    """
    TODO
    """

    output_path: str
    sample_rate: int = 44100
    start_padding: float = 0.5  # in seconds
    soundfont_url: str = (
        "https://github.com/musescore/MuseScore/raw/refs/heads/master/share/sound/MS%20Basic.sf3"
    )
    soundfont_path: str = "../data/soundfonts/ms_basic.sf3"
    midi_db: float = 11
    sample_db: float = -3

    @property
    def start_padding_size(self) -> int:
        return int(self.start_padding * self.sample_rate)


class Arrangement:

    def __init__(
        self,
        metadata: ArrangementMetadata,
        parts: List[part.Part],
    ):
        self._metadata = metadata
        self._parts = parts

    def export(self, config: ArrangementExportConfig):

        midi_parts, sampled_parts = self._get_parts_by_type()

        output = self._midi_to_audio_data(config, midi_parts)

        # Add sampled instruments' WAV datas onto MIDI WAV data.
        for part in sampled_parts:
            array = part.get_audio_data(config).array
            output.add_range((0, len(array)), array * db_to_strength(config.sample_db))

        # Export combined WAV data and close temporary files.
        wav.write(config.output_path, config.sample_rate, output.array)  # type: ignore

    def _get_parts_by_type(
        self,
    ) -> Tuple[
        List[part.MIDIPart],
        List[part.SampledPart],
    ]:
        midi_parts: List[part.MIDIPart] = list()
        sampled_parts: List[part.SampledPart] = list()

        for part_ in self._parts:
            if isinstance(part_, part.MIDIPart):
                midi_parts.append(part_)
            elif isinstance(part_, part.SampledPart):
                sampled_parts.append(part_)

        return (midi_parts, sampled_parts)

    def _midi_to_audio_data(
        self,
        config: ArrangementExportConfig,
        parts: List[part.MIDIPart],
    ) -> AudioData:
        if len(parts) == 0:
            return AudioData()

        # Download soundfont if missing.
        if not os.path.exists(config.soundfont_path):
            os.makedirs(os.path.dirname(config.soundfont_path), exist_ok=True)
            os.system(f"curl -o {config.soundfont_path} -L {config.soundfont_url}")

        # Create temporary files.
        midi_file = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        midi_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

        # Create MIDI data.
        midi_data = MIDIFile(
            numTracks=len(parts),
            ticks_per_quarternote=self._metadata.quantization,
        )
        for track, part_ in enumerate(parts):
            part_.add_notes_to_track(midi_data, track)

        # Write MIDI data to MIDI file.
        midi_data.writeFile(midi_file)  # type: ignore
        midi_file.flush()

        # Close files before FluidSynth call.
        midi_file.close()
        midi_wav_file.close()

        # Convert MIDI file to WAV file and load WAV file.
        os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
        os.system(
            f"fluidsynth -ni {config.soundfont_path} {midi_file.name} -F {midi_wav_file.name} -r {config.sample_rate}"
        )
        output = AudioData.from_file(
            midi_wav_file.name,
            sample_rate=config.sample_rate,
            db=config.midi_db,
        )
        output.pad_start(config.start_padding_size)

        # Delete temp files.
        os.remove(midi_file.name)
        os.remove(midi_wav_file.name)

        return output
