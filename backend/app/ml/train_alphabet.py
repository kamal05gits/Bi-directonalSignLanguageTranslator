import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AlphabetRecord:
    label: str
    values: list[float]
    split: str

def discover_records(root):
    records = []
    for split in ("train", "validation", "test"):
        path = Path(root) / f"{split}.csv"
        if not path.exists(): continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                records.append(AlphabetRecord(row["label"], [float(v) for k,v in row.items() if k != "label"], split))
    return records
