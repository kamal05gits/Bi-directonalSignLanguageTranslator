import numpy as np

from app.data.import_fingerspelling import normalize_letter_label, resample_sequence


def test_fingerspelling_labels_map_to_sentence_tokens():
    assert normalize_letter_label("A") == "a"
    assert normalize_letter_label(" ") == "space"
    assert normalize_letter_label(".") == "period"


def test_resample_sequence_has_fixed_shape():
    frames = [np.full(126, index, dtype=np.float32) for index in range(4)]
    result = resample_sequence(frames, 8)
    assert result.shape == (8, 126)
    assert result[0, 0] == 0
    assert result[-1, 0] == 3
