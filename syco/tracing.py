from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class TraceEvent:
    experiment: str
    seed: int
    condition: str
    question_id: int
    round: int
    agent_id: str
    role: str

    ground_truth: str
    prior_answer: Optional[str]
    new_answer: str
    correct: bool
    changed: bool

    peers_shown: list = field(default_factory=list)
    peer_majority_answer: Optional[str] = None
    peer_max_confidence: float = 0.0
    peer_max_authority: float = 0.0
    moved_to_peer: bool = False


class Tracer:
    def __init__(self):
        self.events: list[TraceEvent] = []

    def log(self, event: TraceEvent):
        self.events.append(event)

    def to_jsonl(self, path: str):
        with open(path, "w") as f:
            for e in self.events:
                f.write(json.dumps(asdict(e)) + "\n")

    def rows(self) -> list[dict]:
        return [asdict(e) for e in self.events]
