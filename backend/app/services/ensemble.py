"""Soft-voting ensemble that combines the three recognition models.

The three bundled models live in different input/label spaces:

- ``alphabet``: single-frame photo CNN -> one of 26 letters.
- ``fingerspelling``: MLP over a 126-value hand-landmark vector -> one of 26
  letters (the same label space as the alphabet model).
- ``word``: CISLR/I3D video classifier -> one of 82 whole words.

The two letter models are merged with a soft vote: each model contributes
its full ranked distribution, per-label probabilities are averaged, and
agreement between the photo and landmark models keeps the consensus
confident while disagreement dilutes it. The word model lives in its own
label space, so it is kept as a separate candidate; it becomes the primary
prediction only when it clears its own (lower) threshold *and* outscores the
letter consensus, so a weak word guess never overrides two agreeing letter
models.

Everything here is pure logic over :class:`Prediction` lists — no TensorFlow,
no I/O — so it is fully unit-testable in the minimal CI environment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .keras_classifier import Prediction

LETTER_MODELS = ("alphabet", "fingerspelling")
WORD_MODEL = "word"
DEFAULT_TOP_K = 5


class EnsembleError(ValueError):
    """Raised when no model produced a usable prediction."""


class ModelSource:
    """One model's contribution (or failure) to a combined prediction."""

    __slots__ = ("model", "ran", "ok", "label", "confidence", "top_predictions", "detail", "client_error")

    def __init__(
        self,
        model: str,
        *,
        ran: bool,
        ok: bool = False,
        label: str | None = None,
        confidence: float | None = None,
        top_predictions: Sequence[Prediction] = (),
        detail: str = "ok",
        client_error: bool = False,
    ) -> None:
        self.model = model
        self.ran = ran  # an input was provided and prediction was attempted
        self.ok = ok  # the model produced a ranking
        self.label = label
        self.confidence = confidence
        self.top_predictions = tuple(top_predictions)
        self.detail = detail  # "ok", "no input provided", or an error message
        self.client_error = client_error  # True when the failure was caused by the request payload


def not_run(model: str, detail: str = "no input provided") -> ModelSource:
    return ModelSource(model, ran=False, detail=detail)


def merge_letter_distributions(
    distributions: Iterable[Sequence[Prediction]], top_k: int = DEFAULT_TOP_K
) -> list[Prediction]:
    """Average ranked letter distributions into one consensus ranking.

    Every model contributes its confidence for each label it produced; a
    label one model omitted scores zero from that model. Two models that
    agree on a letter therefore keep a high consensus confidence, while a
    disagreement splits the vote and lowers it.
    """
    distributions = [list(distribution) for distribution in distributions if distribution]
    if not distributions:
        return []
    totals: dict[str, float] = {}
    for distribution in distributions:
        for prediction in distribution:
            totals[prediction.label] = totals.get(prediction.label, 0.0) + prediction.confidence
    merged = [
        Prediction(label=label, confidence=min(1.0, total / len(distributions)))
        for label, total in totals.items()
    ]
    merged.sort(key=lambda item: item.confidence, reverse=True)
    return merged[: max(1, top_k)]


def combine(
    sources: Sequence[ModelSource], letter_threshold: float, word_threshold: float
) -> "CombinedPrediction":
    """Merge per-model results into one combined prediction.

    ``letter_threshold`` and ``word_threshold`` mirror the thresholds used by
    the individual ``/api/predict/*`` endpoints, so a combined result is
    accepted under exactly the same policy as its parts.
    """
    successful = [source for source in sources if source.ok and source.top_predictions]
    if not successful:
        raise EnsembleError("No model produced a prediction.")

    letter_sources = [source for source in successful if source.model in LETTER_MODELS]
    word_source = next((source for source in successful if source.model == WORD_MODEL), None)
    word_top = word_source.top_predictions[0] if word_source else None

    letter_ranking = merge_letter_distributions([source.top_predictions for source in letter_sources])
    letter_top = letter_ranking[0] if letter_ranking else None

    agreement: bool | None = None
    if len(letter_sources) >= 2:
        agreement = all(source.label == letter_sources[0].label for source in letter_sources)

    word_wins = (
        word_top is not None
        and word_top.confidence >= word_threshold
        and (letter_top is None or word_top.confidence > letter_top.confidence)
    )

    if word_wins or letter_top is None:
        primary, threshold = word_top, word_threshold
    else:
        primary, threshold = letter_top, letter_threshold

    letter_names = " + ".join(source.model for source in letter_sources)
    if word_wins:
        method = f"word model ({word_top.confidence:.0%}), outscoring the letter consensus"
    elif len(letter_sources) >= 2:
        method = f"soft vote of {letter_names} ({'models agreed' if agreement else 'models disagreed'})"
    elif len(letter_sources) == 1:
        method = f"{letter_sources[0].model} model only"
    else:
        method = "word model only"

    return CombinedPrediction(
        label=primary.label,
        confidence=primary.confidence,
        accepted=primary.confidence >= threshold,
        agreement=agreement,
        method=method,
        word=word_top,
        sources=tuple(sources),
        top_predictions=tuple(letter_ranking),
    )


class CombinedPrediction:
    __slots__ = ("label", "confidence", "accepted", "agreement", "method", "word", "sources", "top_predictions")

    def __init__(
        self,
        *,
        label: str,
        confidence: float,
        accepted: bool,
        agreement: bool | None,
        method: str,
        word: Prediction | None,
        sources: tuple[ModelSource, ...],
        top_predictions: tuple[Prediction, ...],
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.accepted = accepted
        self.agreement = agreement
        self.method = method
        self.word = word
        self.sources = sources
        self.top_predictions = top_predictions
