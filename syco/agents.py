from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import ModelClient, Decision, build_client


@dataclass
class Agent:
    id: str
    client: ModelClient
    role: str = "plain"
    authority: float = 0.0
    display_confidence: Optional[float] = None
    peer_kind: Optional[str] = None
    peer_model_strength: Optional[str] = None
    answer: Optional[str] = None
    confidence: float = 0.5
    history: list = field(default_factory=list)

    def act(self, prompt, options, peers, ground_truth, rng) -> Decision:
        d = self.client.decide(prompt, options, peers, ground_truth, rng)
        # keeping the prior answer when the mock holds ground
        if d.answer == "__HOLD__" and self.answer is not None:
            d = Decision(answer=self.answer, confidence=self.confidence,
                         rationale="held ground")
        self.answer = d.answer
        self.confidence = d.confidence
        self.history.append(d)
        return d

    def shown_to_peers(self) -> dict:
        return {
            "id": self.id,
            "answer": self.answer,
            "confidence": self.display_confidence
            if self.display_confidence is not None else self.confidence,
            "authority": self.authority,
            "kind": self.peer_kind,
            "model_strength": self.peer_model_strength,
        }


@dataclass
class ScriptedAgent(Agent):
    fixed_answer: Optional[str] = None
    fixed_confidence: float = 0.9

    def act(self, prompt, options, peers, ground_truth, rng) -> Decision:
        ans = self.fixed_answer if self.fixed_answer is not None else options[0]
        self.answer = ans
        self.confidence = self.fixed_confidence
        d = Decision(answer=ans, confidence=self.fixed_confidence,
                     rationale="scripted")
        self.history.append(d)
        return d

    def shown_to_peers(self) -> dict:
        return {
            "id": self.id,
            "answer": self.answer,
            "confidence": self.fixed_confidence,
            "authority": self.authority,
            "kind": self.peer_kind,
            "model_strength": self.peer_model_strength,
        }


def build_agent(spec: dict, options_hint: Optional[list[str]] = None) -> Agent:
    spec = dict(spec)
    agent_id = spec.pop("id")
    kind = spec.pop("type", "model")

    if kind == "scripted":
        # storing answer_policy (correct, incorrect, or a literal option) for per-question resolution
        return ScriptedAgent(
            id=agent_id,
            client=None,
            role=spec.get("role", "peer"),
            authority=float(spec.get("authority", 0.0)),
            fixed_confidence=float(spec.get("confidence", 0.9)),
            fixed_answer=spec.get("answer_policy", None),
            peer_kind=spec.get("peer_kind", None),
            peer_model_strength=spec.get("peer_model_strength", None),
        )

    client = build_client(spec.pop("model", {"type": "mock"}))
    return Agent(
        id=agent_id,
        client=client,
        role=spec.get("role", "plain"),
        authority=float(spec.get("authority", 0.0)),
        display_confidence=spec.get("display_confidence", None),
        peer_kind=spec.get("peer_kind", None),
        peer_model_strength=spec.get("peer_model_strength", None),
    )
