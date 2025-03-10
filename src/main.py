import os

from env import ENV

from melody_extraction.melody_extractor import MelodyExtractor
from generation.instruments.voice import Voice
from common.roll import RollExportConfig


# TODO: Create generation process.
async def generate(input_path: str) -> str:

    name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(ENV.OUTPUT_DIR, f"{name}.wav")

    roll, notes = MelodyExtractor.extract_melody(input_path)
    v = roll.add_instrument("voice1", Voice)
    v.generate(notes)

    roll.export(RollExportConfig(output_path))

    return output_path
