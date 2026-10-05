import csv

from app.data.import_cislr import _pick_column, _read_records
from app.ml.train_alphabet import discover_records


def test_alphabet_csv_layout_is_discovered(tmp_path):
    (tmp_path / "train.csv").write_text("label,p0,p1,p2,p3\nA,0,1,2,3\nB,3,2,1,0\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("label,p0,p1,p2,p3\nA,0,1,2,3\nB,3,2,1,0\n", encoding="utf-8")
    records = discover_records(tmp_path)
    assert len(records) == 4
    assert {record.label for record in records} == {"A", "B"}


def test_cislr_csv_columns_are_detected(tmp_path):
    path = tmp_path / "dataset.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "category"])
        writer.writeheader()
        writer.writerow({"video_id": "a.mp4", "category": "hello"})
    records = _read_records(path)
    assert records[0].video == "a.mp4"
    assert records[0].label == "hello"
    assert _pick_column(["video_id", "category"], ("video", "video_id")) == "video_id"
