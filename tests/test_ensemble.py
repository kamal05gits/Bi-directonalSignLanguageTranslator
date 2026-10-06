"""Unit tests for the soft-voting ensemble (no TensorFlow needed)."""

from __future__ import annotations

import pytest
from app.services.ensemble import (
    EnsembleError,
    ModelSource,
    combine,
    merge_letter_distributions,
    not_run,
)
from app.services.keras_classifier import Prediction


def _letter_source(model: str, label: str, confidence: float, second: str = "b", second_confidence: float = 0.1) -> ModelSource:
    ranking = [Prediction(label=label, confidence=confidence), Prediction(label=second, confidence=second_confidence)]
    return ModelSource(
        model, ran=True, ok=True, label=ranking[0].label, confidence=ranking[0].confidence, top_predictions=ranking
    )


def _word_source(label: str, confidence: float) -> ModelSource:
    ranking = [Prediction(label=label, confidence=confidence)]
    return ModelSource("word", ran=True, ok=True, label=label, confidence=confidence, top_predictions=ranking)


# ---------------------------------------------------------------- distributions


def test_merge_averages_agreeing_models():
    merged = merge_letter_distributions(
        [[Prediction("a", 0.9), Prediction("b", 0.1)], [Prediction("a", 0.8), Prediction("c", 0.2)]]
    )
    assert merged[0].label == "a"
    assert merged[0].confidence == pytest.approx(0.85)


def test_merge_penalizes_disagreement():
    merged = merge_letter_distributions(
        [[Prediction("a", 0.9)], [Prediction("b", 0.9)]]
    )
    # Each label only gets half a vote, so the consensus is diluted.
    assert merged[0].confidence == pytest.approx(0.45)
    assert merged[0].label in {"a", "b"}


def test_merge_of_a_single_distribution_keeps_values():
    merged = merge_letter_distributions([[Prediction("a", 0.7), Prediction("b", 0.3)]])
    assert [(p.label, p.confidence) for p in merged] == [("a", 0.7), ("b", 0.3)]


def test_merge_of_nothing_is_empty():
    assert merge_letter_distributions([]) == []


# ------------------------------------------------------------------- combining


def test_agreeing_letter_models_keep_confidence_and_report_agreement():
    combined = combine([_letter_source("alphabet", "a", 0.9), _letter_source("fingerspelling", "a", 0.8)], 0.7, 0.3)
    assert combined.label == "a"
    assert combined.confidence == pytest.approx(0.85)
    assert combined.accepted is True
    assert combined.agreement is True
    assert "soft vote" in combined.method
    assert combined.word is None


def test_disagreeing_letter_models_dilute_confidence():
    combined = combine(
        [
            _letter_source("alphabet", "a", 0.8, second_confidence=0.0),
            _letter_source("fingerspelling", "b", 0.7, second_confidence=0.0),
        ],
        0.7,
        0.3,
    )
    assert combined.agreement is False
    # The soft vote winner 'a' only gets half a vote: (0.8 + 0) / 2 = 0.4 -> rejected.
    assert combined.label == "a"
    assert combined.confidence == pytest.approx(0.4)
    assert combined.accepted is False


def test_high_word_confidence_outscores_letters():
    combined = combine([_letter_source("alphabet", "a", 0.4), _word_source("help", 0.8)], 0.7, 0.3)
    assert combined.label == "help"
    assert combined.accepted is True
    assert combined.word is not None and combined.word.label == "help"
    assert "word model" in combined.method


def test_weak_word_never_overrides_agreeing_letters():
    combined = combine(
        [_letter_source("alphabet", "a", 0.9), _letter_source("fingerspelling", "a", 0.9), _word_source("no", 0.35)],
        0.7,
        0.3,
    )
    assert combined.label == "a"
    assert combined.word.label == "no"  # still surfaced as a candidate


def test_word_below_threshold_is_not_accepted_when_alone():
    combined = combine([_word_source("help", 0.2)], 0.7, 0.3)
    assert combined.label == "help"
    assert combined.accepted is False


def test_single_letter_model_passes_through():
    combined = combine([_letter_source("fingerspelling", "c", 0.95)], 0.7, 0.3)
    assert combined.label == "c"
    assert combined.confidence == pytest.approx(0.95)
    assert combined.agreement is None
    assert "fingerspelling model only" in combined.method


def test_failed_sources_are_reported_but_ignored():
    failed = ModelSource("word", ran=True, detail="Word model is unavailable")
    combined = combine([_letter_source("alphabet", "a", 0.9), failed], 0.7, 0.3)
    assert combined.label == "a"
    assert combined.word is None
    assert any(source.detail == "Word model is unavailable" for source in combined.sources)


def test_not_run_sources_carry_no_vote():
    combined = combine([_letter_source("alphabet", "a", 0.9), not_run("fingerspelling")], 0.7, 0.3)
    assert combined.label == "a"
    assert combined.confidence == pytest.approx(0.9)
    assert combined.agreement is None


def test_combine_without_any_successful_model_raises():
    with pytest.raises(EnsembleError):
        combine([ModelSource("alphabet", ran=True, detail="missing")], 0.7, 0.3)
