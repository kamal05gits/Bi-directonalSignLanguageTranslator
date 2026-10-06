from dataclasses import dataclass
from typing import Any


@dataclass
class HandDetectionResult:
    hands: list[list[tuple[float, float, float]]]
    handedness: list[str]
    annotated_frame: Any = None
