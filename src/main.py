import os

from env import ENV

from common.roll import Roll, RollConfig, RollExportConfig

from melody_extraction.melody_extractor import MelodyExtractor

from generation.chord_progression_generator import ChordProgressionGenerator
from generation.instruments.voice import Voice
from generation.instruments.piano import Piano
from generation.instruments.bass_drum import BassDrum
from generation.instruments.snare_drum import SnareDrum


async def generate(input_path: str) -> str:

    name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(ENV.OUTPUT_DIR, f"{name}.wav")

    # MVP assumptions.
    roll_config = RollConfig(
        beats_per_minute=110,
        time_signature=(4, 4),
        quantization=16,
    )

    melody_extractor = MelodyExtractor()
    melody = melody_extractor.extract_melody(roll_config, input_path)

    chord_progression_generator = ChordProgressionGenerator(roll_config, melody)
    chord_progression = chord_progression_generator.generate()

    roll = Roll(melody, chord_progression, roll_config)
    roll.add_instrument("Voice 1", Voice)
    roll.add_instrument("Piano", Piano)
    roll.add_instrument("Bass Drum", BassDrum)
    roll.add_instrument("Snare Drum", SnareDrum)
    roll.generate()
    roll.export(RollExportConfig(output_path))

    return output_path
