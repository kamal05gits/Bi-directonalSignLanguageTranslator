# Development roadmap and acceptance gates

| Phase | Deliverable | Verification |
|---|---|---|
| 1 | Requirements, architecture, contracts | `docs/architecture.md`, README and module boundaries |
| 2 | Environment and optional-dependency guards | `python -m compileall`, import-safe unit tests |
| 3 | OpenCV capture and preprocessing | camera smoke test; invalid-frame tests |
| 4 | MediaPipe detector and overlay | live preview; missing dependency error |
| 5 | landmark feature extraction and collector | `.npy` shape `(T,126)`; dataset validation |
| 6 | spatial encoder + LSTM | train command saves the four model artifacts |
| 7 | confidence-aware inference | low-confidence and duplicate suppression tests |
| 8 | real-time worker integration | camera remains responsive while inference runs |
| 9 | token/sentence layer | original token trace plus editable text |
| 10 | translation/TTS | explicit dictionary result and non-blocking TTS |
| 11 | complete Qt screens | Live, Collection, Training, Evaluation, Settings, About |
| 12 | external dataset integration | ISL-Fingerspelling importer, source split metadata, dataset license notes |
| 13 | testing and documentation | pytest, measured evaluation report, limitations |

## Current repository status

All phases have an implementation path. Hardware-dependent phases become operational only after dependencies, camera permissions, collected data and a trained model are present. The untrained state is intentionally visible and safe; it is not treated as successful recognition.
