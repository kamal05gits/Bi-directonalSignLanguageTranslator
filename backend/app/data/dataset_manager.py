import json
from pathlib import Path
import numpy as np


def stratified_split(labels, seed=42, validation_ratio=.2, test_ratio=.2):
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(np.asarray(labels) == label)
        rng.shuffle(indices)
        n_test = int(round(len(indices) * test_ratio))
        n_validation = int(round(len(indices) * validation_ratio))
        test.extend(indices[:n_test]); validation.extend(indices[n_test:n_test+n_validation]); train.extend(indices[n_test+n_validation:])
    return np.asarray(train), np.asarray(validation), np.asarray(test)


class DatasetManager:
    def __init__(self, paths):
        self.root = Path(paths.data) / "sequences"; self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
    def save_sequence(self, label, sequence, sample_id, split="train", group_id=None):
        safe_id = "".join(c for c in sample_id if c.isalnum() or c in "-_")
        path = self.root / f"{safe_id}.npy"; np.save(path, np.asarray(sequence, dtype=np.float32))
        record = {"label": label, "path": path.name, "split": split, "group_id": group_id}
        with self.index_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        return path
    def load_sequences(self, sequence_length, split=None):
        records = [] if not self.index_path.exists() else [json.loads(line) for line in self.index_path.read_text().splitlines() if line]
        records = [r for r in records if split is None or r["split"] == split]
        arrays, labels = [], []
        for record in records:
            value = np.load(self.root / record["path"])
            if value.shape[0] == sequence_length: arrays.append(value); labels.append(record["label"])
        if not arrays: return np.empty((0, sequence_length, 0), dtype=np.float32), records, labels
        return np.stack(arrays), records, labels
