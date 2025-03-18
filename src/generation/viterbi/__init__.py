import numpy as np

from common.structures.chord import Chord, ChordQuality
from common.arrangement import ArrangementMetadata
from common.note_collection import NoteCollection

from generation.chord_progression import ChordProgression
from generation.viterbi.transition_matrix import TransitionMatrix
from generation.viterbi.observation_function import ObservationFunction
from generation.chord_progression_generator import ChordProgressionGenerator
from generation.viterbi.viterbi_index import ViterbiIndex


class ViterbiChordProgressionGenerator(ChordProgressionGenerator):
    """
    A chord progression generator using the Vitberi algorithm.

    Note that we are starting at the end of the melody and working backwards.

    TODO:
    - Need to determine key of melody.

    :param arrangement_metadata: The arrangement metadata.
    :param melody: The melody to generate the chord progression for.
    """

    def __init__(
        self, arrangement_metadata: ArrangementMetadata, melody: NoteCollection
    ):
        self._priors = np.zeros((ViterbiIndex.TOTAL_STATES))
        # End on tonic major
        # TODO: Determine key of melody.
        self._priors[
            ViterbiIndex.from_chord(
                Chord(arrangement_metadata.key, ChordQuality.Maj)
            ).index
        ] = 1
        self._transition_matrix = TransitionMatrix(arrangement_metadata.key).matrix
        self._observation_fn = ObservationFunction()

        self._melody = melody
        self._arrangement_metadata = arrangement_metadata

    def generate(self) -> ChordProgression:
        melody_end = self._melody.list()[-1].start + self._melody.list()[-1].duration
        q = self._arrangement_metadata.quantization
        end = int(np.ceil(melody_end / q) * q)

        chord_progression = ChordProgression(start_time=0, end_time=end)

        measure_duration = self._arrangement_metadata.Duration(
            self._arrangement_metadata.beats_per_measure
        )
        hop_size = round(measure_duration / 2)

        # Total number of observations (melody chunks).
        T = int(np.ceil(chord_progression.end_time / hop_size))

        # DP table for path probabilities.
        probs = np.ones((ViterbiIndex.TOTAL_STATES, T))
        # DP table for backtracking (points to previous state).
        parent = np.zeros((ViterbiIndex.TOTAL_STATES, T))

        # Initialize with priors and first observation.
        probs[:, 0] = self._priors * self._observation_fn.get_score(
            np.arange(ViterbiIndex.TOTAL_STATES),
            notes=self._melody,
            start_time=0,
            hop_size=hop_size,
        )

        t = 1
        for time in reversed(range(0, melody_end - hop_size, hop_size)):
            for i in range(ViterbiIndex.TOTAL_STATES):
                probs[i, t] = np.max(
                    probs[:, t - 1] * self._transition_matrix[:, i]
                ) * self._observation_fn.get_score(i, self._melody, time, hop_size)
                parent[i, t] = np.argmax(
                    probs[:, t - 1] * self._transition_matrix[:, i]
                )

            if probs[:, t].sum() == 0:
                print("WARNING: probabilities converged to zero.")

            t += 1

        # Reconstruct the optimal path.
        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(probs[:, T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = parent[path[t + 1], t + 1]

        # Build chord progression.
        i = 0
        prev_chord = -1
        for t in reversed(range(T)):
            if path[t] != prev_chord:
                chord_progression.add_chords(
                    (ViterbiIndex(path[t]).to_chord(), i * hop_size)
                )
            i += 1
            prev_chord = path[t]

        return chord_progression
