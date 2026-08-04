"""Query classification (FR-L3-3)."""

from __future__ import annotations

import json

from bankassist.llm.stub import StubLLMClient
from bankassist.rag.stages.classifier import LABELS, QueryClassifier


def test_all_seven_labels_round_trip() -> None:
    for label in LABELS:
        llm = StubLLMClient([json.dumps({"label": label, "confidence": 0.8})])
        result = QueryClassifier(llm).execute("some question")

        assert result.label == label
        assert result.confidence == 0.8


def test_malformed_response_classifies_as_unknown_with_zero_confidence() -> None:
    llm = StubLLMClient(["not json at all"])

    result = QueryClassifier(llm).execute("what is the KYC requirement?")

    assert result.label == "Unknown"
    assert result.confidence == 0.0


def test_unrecognized_label_classifies_as_unknown() -> None:
    llm = StubLLMClient([json.dumps({"label": "NotARealLabel", "confidence": 0.9})])

    result = QueryClassifier(llm).execute("q")

    assert result.label == "Unknown"
    assert result.confidence == 0.0


def test_confidence_is_clamped_to_the_unit_interval() -> None:
    llm = StubLLMClient([json.dumps({"label": "Policy", "confidence": 1.5})])

    result = QueryClassifier(llm).execute("q")

    assert result.confidence == 1.0


def test_classifier_calls_the_classifier_tier() -> None:
    llm = StubLLMClient([json.dumps({"label": "Policy", "confidence": 0.7})])

    QueryClassifier(llm).execute("q")

    assert llm.last_call().tier == "classifier"


def test_latency_is_recorded_and_nonnegative() -> None:
    llm = StubLLMClient([json.dumps({"label": "FAQ", "confidence": 0.5})])

    result = QueryClassifier(llm).execute("q")

    assert result.latency_ms >= 0.0
