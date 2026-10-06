from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    @classmethod
    def from_root(cls, root):
        root = Path(root); return cls(root, root / "data")
