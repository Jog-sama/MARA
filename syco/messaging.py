from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    round: int
    agent_id: str
    role: str
    answer: str
    confidence: float
    text: str = ""
    authority: float = 0.0
    tool_call: Optional[str] = None
    tool_result: Optional[str] = None


@dataclass
class Transcript:
    messages: list = field(default_factory=list)

    def add(self, m: Message):
        self.messages.append(m)

    def latest_by_agent(self, upto_round: int) -> dict:
        out = {}
        for m in self.messages:
            if m.round <= upto_round:
                out[m.agent_id] = m
        return out

    def visible_to(self, agent_id: str, upto_round: int) -> list:
        # showing every other agent's latest position in both peer and hierarchical topologies
        latest = self.latest_by_agent(upto_round)
        return [m for aid, m in latest.items() if aid != agent_id]
