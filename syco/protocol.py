from __future__ import annotations

import random
from collections import Counter
from typing import Optional

from .agents import Agent, ScriptedAgent
from .tasks import Task, Question
from .tracing import Tracer, TraceEvent


class Orchestrator:
    def __init__(self, experiment: str, agents: list[Agent], task: Task,
                 rounds: int = 3, reveal: str = "after_initial",
                 topology: str = "full", tracer: Optional[Tracer] = None):
        self.experiment = experiment
        self.agents = agents
        self.task = task
        self.rounds = rounds
        self.reveal = reveal
        self.topology = topology
        self.tracer = tracer or Tracer()

    def _resolve_scripted(self, q: Question):
        # turning correct/incorrect policies into this question's options
        for a in self.agents:
            if isinstance(a, ScriptedAgent):
                policy = a.fixed_answer
                if policy == "correct":
                    a.fixed_answer = q.answer
                elif policy in (None, "incorrect"):
                    a.fixed_answer = q.distractor
                a._policy = policy

    def _reset_scripted(self):
        for a in self.agents:
            if isinstance(a, ScriptedAgent) and hasattr(a, "_policy"):
                a.fixed_answer = a._policy

    def run_cell(self, condition: str, seed: int):
        rng = random.Random(seed)
        for qid, q in enumerate(self.task.questions()):
            self._resolve_scripted(q)
            for a in self.agents:
                a.answer, a.confidence, a.history = None, 0.5, []

            for r in range(self.rounds):
                revealed = not (r == 0 and self.reveal == "after_initial")
                # snapshotting answers so no agent reacts to a peer who moved earlier in the round
                snapshot = {a.id: a.shown_to_peers() for a in self.agents
                            if a.answer is not None}

                for a in self.agents:
                    prior = a.answer
                    peers = [] if not revealed else [
                        v for k, v in snapshot.items() if k != a.id]
                    peer_answers = [p["answer"] for p in peers]
                    majority = (Counter(peer_answers).most_common(1)[0][0]
                                if peer_answers else None)

                    a.act(q.prompt, q.options, peers, q.answer, rng)

                    self.tracer.log(TraceEvent(
                        experiment=self.experiment,
                        seed=seed,
                        condition=condition,
                        question_id=qid,
                        round=r,
                        agent_id=a.id,
                        role=a.role,
                        ground_truth=q.answer,
                        prior_answer=prior,
                        new_answer=a.answer,
                        correct=(a.answer == q.answer),
                        changed=(prior is not None and a.answer != prior),
                        peers_shown=peers,
                        peer_majority_answer=majority,
                        peer_max_confidence=max([p["confidence"] for p in peers],
                                                default=0.0),
                        peer_max_authority=max([p["authority"] for p in peers],
                                               default=0.0),
                        moved_to_peer=(prior is not None and majority is not None
                                       and a.answer == majority and a.answer != prior),
                    ))
            self._reset_scripted()
        return self.tracer
