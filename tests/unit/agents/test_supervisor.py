from bankassist.agents.supervisor import decide_route
from bankassist.llm.stub import StubLLMClient


def test_decide_route_parses_valid_json():
    llm = StubLLMClient(['{"route": "POLICY", "confidence": 0.9, "reason": "KYC question"}'])
    decision = decide_route(llm, "What documents are required for KYC?")
    assert decision.route == "POLICY"
    assert decision.confidence == 0.9


def test_decide_route_falls_back_to_clarification_on_malformed_json():
    llm = StubLLMClient(["not json"])
    decision = decide_route(llm, "huh?")
    assert decision.route == "CLARIFICATION"
    assert decision.confidence == 0.0


def test_decide_route_falls_back_on_unknown_route_label():
    llm = StubLLMClient(['{"route": "SPORTS", "confidence": 0.5, "reason": "x"}'])
    decision = decide_route(llm, "who won the match?")
    assert decision.route == "CLARIFICATION"


def test_decide_route_clamps_confidence_to_unit_interval():
    llm = StubLLMClient(['{"route": "BANKING", "confidence": 5, "reason": "x"}'])
    decision = decide_route(llm, "show my transactions")
    assert decision.confidence == 1.0
