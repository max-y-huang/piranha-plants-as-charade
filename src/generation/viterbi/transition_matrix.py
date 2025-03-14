import numpy as np
from common.structures.chord import Chord, ChordQuality
from common.structures.pitch import Pitch
from common.structures.interval import Interval

from generation.viterbi.viterbi_index import ViterbiIndex


class TransitionMatrix:
    """
    The transition matrix for the Viterbi chord progression generator. This matrix
    represents the probability of transitioning from one chord to another, without
    considering the melody.

    Note that since we are working backwards, the transition matrix should be
    interpreted as follows:
        matrix[current_chord, previous_chord] = probability of transitioning from prev to current

    The rows are normalized such that they sum to 1.

    Current rules:
    - A chord may be preceded by I, IV, V in terms of the key
    - A chord may be preceded by V7 in terms of the current chord
    - No dom7 after dom7

    Considerations to think about:
    - Should the diagonal (same chord) be treated differently?
    - Probabilities of special cadences (eg. V-I)?
    - Data collection for transition matrix?
    """

    def __init__(self, key: Pitch = Pitch.from_str("C")):
        self._matrix = (
            np.zeros((ViterbiIndex.TOTAL_STATES, ViterbiIndex.TOTAL_STATES)) / 100
        )

        I_chord = ViterbiIndex.from_chord(Chord(key, ChordQuality.Maj)).index
        IV_chord = ViterbiIndex.from_chord(
            Chord(key + Interval.from_str("4"), ChordQuality.Maj)
        ).index
        V_chord = ViterbiIndex.from_chord(
            Chord(key + Interval.from_str("5"), ChordQuality.Maj)
        ).index

        self._matrix[:, [I_chord, IV_chord, V_chord]] = 1

        for c in range(ViterbiIndex.TOTAL_STATES):
            chord = ViterbiIndex(c).to_chord()
            if chord.quality != ChordQuality.Dom7:
                v7 = ViterbiIndex.from_chord(chord.get_V7()).index
                self._matrix[c, v7] = 2

        for r in range(ViterbiIndex.TOTAL_STATES):
            self._matrix[r, :] /= np.sum(self._matrix[r, :])

    @property
    def matrix(self):
        return self._matrix
