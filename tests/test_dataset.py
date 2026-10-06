import numpy as np
from app.config import ProjectPaths
from app.data.dataset_manager import DatasetManager, stratified_split


def test_stratified_split_has_no_overlap():
    y = np.repeat(np.arange(2), 10)
    train, validation, test = stratified_split(y, seed=7)
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert set(train) | set(validation) | set(test) == set(range(len(y)))


def test_dataset_manager_filters_source_splits(tmp_path):
    manager = DatasetManager(ProjectPaths.from_root(tmp_path))
    sequence = np.zeros((3, 126), dtype=np.float32)
    manager.save_sequence("a", sequence, sample_id="a_train", split="train", group_id="video-a")
    manager.save_sequence("a", sequence, sample_id="a_test", split="test", group_id="video-b")
    _, _, train_labels = manager.load_sequences(3, split="train")
    _, _, test_labels = manager.load_sequences(3, split="test")
    assert train_labels == ["a"]
    assert test_labels == ["a"]
