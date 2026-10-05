import csv
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VideoRecord:
    video: str
    label: str

def _pick_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns: return candidate
    raise ValueError(f"None of {candidates} found")

def _read_records(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows: return []
    video = _pick_column(rows[0], ("video", "video_id", "path", "file"))
    label = _pick_column(rows[0], ("label", "category", "gloss"))
    return [VideoRecord(row[video], row[label]) for row in rows]
