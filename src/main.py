import os
import time
from typing import Callable, Any

from logger import LOGGER
from env import ENV

from common.note_collection import NoteCollection

from melody_extraction.signal import SignalMelodyExtractor

from generation.chord_progression import ChordProgression
from generation.viterbi import ViterbiChordProgressionGenerator
from instruments.voice_melody import VoiceMelody
from instruments.voice_harmony import VoiceHarmony
from instruments.piano import Piano
from instruments.bass_drum import BassDrum
from instruments.snare_drum import SnareDrum

from export.arrangement import Arrangement, ArrangementExportConfig, ArrangementMetadata
from export.arrangement_generator import ArrangementGenerator


def timed(label: str) -> Any:
    capitalized_label = label[0].upper() + label[1:]

    def decorator(function: Callable[[], None]):
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            time_stamp = time.time()
            LOGGER.info(f"Starting {label}...")
            ret = function(*args, **kwargs)
            elapsed_time = time.time() - time_stamp
            LOGGER.info(f"{capitalized_label} completed in {elapsed_time} seconds!")
            return ret

        return wrapper

    return decorator


@timed("melody extraction")
def extract_melody(
    input_path: str,
    arrangement_metadata: ArrangementMetadata,
) -> NoteCollection:
    melody_extractor = SignalMelodyExtractor()
    melody = melody_extractor.extract_melody(arrangement_metadata, input_path)
    return melody


@timed("chord generation")
def generate_chords(
    melody: NoteCollection,
    arrangement_metadata: ArrangementMetadata,
) -> ChordProgression:
    chord_progression_generator = ViterbiChordProgressionGenerator(
        arrangement_metadata, melody
    )
    return chord_progression_generator.generate()


@timed("arrangement generation")
def generate_arrangement(
    melody: NoteCollection,
    chord_progression: ChordProgression,
    arrangement_metadata: ArrangementMetadata,
) -> Arrangement:
    arrangement_generator = ArrangementGenerator(
        melody,
        chord_progression,
        arrangement_metadata,
    )
    arrangement_generator.add_instrument("Voice 1", VoiceMelody)
    arrangement_generator.add_instrument("Voice 2", VoiceHarmony)
    arrangement_generator.add_instrument("Piano", Piano)
    arrangement_generator.add_instrument("Bass Drum", BassDrum)
    arrangement_generator.add_instrument("Snare Drum", SnareDrum)
    return arrangement_generator.generate()


@timed("audio export")
def export_audio(output_path: str, arrangement: Arrangement):
    arrangement.export(ArrangementExportConfig(output_path))


async def generate(input_path: str) -> str:

    name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(ENV.OUTPUT_DIR, f"{name}.wav")

    # MVP assumptions.
    arrangement_metadata = ArrangementMetadata(
        beats_per_minute=110,
        time_signature=(4, 4),
        quantization=16,
    )

    melody = extract_melody(input_path, arrangement_metadata)
    chord_progression = generate_chords(melody, arrangement_metadata)
    arrangement = generate_arrangement(melody, chord_progression, arrangement_metadata)
    export_audio(output_path, arrangement)

    return output_path
