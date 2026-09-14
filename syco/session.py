from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .agents import Agent, ScriptedAgent
from .messaging import Transcript, Message
from .models import AgentTurn, est_cost


@dataclass
class MASession:
    session_id: str
    topology: str
    agents: list
    task: object
    registry: object
    rounds: int = 3
    model_label: str = "mock"

    trace: list = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0

    def _orchestrator(self) -> Optional[Agent]:
        if self.topology != "hierarchical":
            return None
        for a in self.agents:
            if a.role == "orchestrator":
                return a
        # falling back to the first agent when none is marked orchestrator
        return self.agents[0]

    @staticmethod
    def _scripted_answer(agent, q):
        pol = agent.fixed_answer
        if pol == "correct":
            return q.answer
        if pol in (None, "incorrect"):
            return q.distractor
        return pol

    def _ask(self, agent, q, visible, phase, rng):
        if isinstance(agent, ScriptedAgent):
            return AgentTurn(answer=self._scripted_answer(agent, q),
                             confidence=agent.fixed_confidence,
                             thought="scripted authority")
        state = {
            "ground_truth": q.answer,
            "tool_hint": getattr(q, "tool_hint", None),
            "current_answer": agent.answer,
            "current_confidence": agent.confidence,
        }
        turn = agent.client.respond(q.prompt, q.options, visible, state,
                                    self.registry, rng, phase)
        # running the tool the mock asked for, then asking it again
        if turn.answer == "__PENDING__" and turn.tool_name and self.registry:
            self.registry.run(turn.tool_name, turn.tool_arg or "")
            state["_post_tool"] = True
            turn2 = agent.client.respond(q.prompt, q.options, visible, state,
                                         self.registry, rng, phase)
            turn2.tool_name, turn2.tool_arg = turn.tool_name, turn.tool_arg
            turn = turn2
        self.tokens_in += turn.tokens_in
        self.tokens_out += turn.tokens_out
        return turn

    def _visible(self, agent, transcript, upto):
        raw = transcript.visible_to(agent.id, upto)
        return [{"agent_id": m.agent_id, "role": m.role, "answer": m.answer,
                 "confidence": m.confidence, "authority": m.authority,
                 "text": m.text} for m in raw]

    def run(self):
        for qid, q in enumerate(self.task.questions()):
            transcript = Transcript()
            for a in self.agents:
                a.answer, a.confidence = None, 0.5

            for a in self.agents:
                turn = self._ask(a, q, [], "independent", random.Random(qid * 97 + hash(a.id) % 1000))
                a.answer, a.confidence = turn.answer, turn.confidence
                transcript.add(Message(round=0, agent_id=a.id, role=a.role,
                                       answer=a.answer, confidence=a.confidence,
                                       text=turn.thought, authority=a.authority,
                                       tool_call=turn.tool_name))
                self._log(qid, 0, a, q.answer, prior=None, orch_ans=None, peer_maj=None, turn=turn)

            orch = self._orchestrator()
            rounds_used = 0
            for r in range(1, self.rounds + 1):
                rounds_used = r
                for a in self.agents:
                    prior = a.answer
                    visible = self._visible(a, transcript, r - 1)
                    orch_ans = (orch.answer if orch and a.id != orch.id else None)
                    peer_answers = [m["answer"] for m in visible if m["role"] != "orchestrator"]
                    peer_maj = Counter(peer_answers).most_common(1)[0][0] if peer_answers else None

                    rng = random.Random(qid * 131 + r * 17 + hash(a.id) % 1000)
                    turn = self._ask(a, q, visible, "deliberate", rng)
                    a.answer, a.confidence = turn.answer, turn.confidence
                    transcript.add(Message(round=r, agent_id=a.id, role=a.role,
                                           answer=a.answer, confidence=a.confidence,
                                           text=turn.thought, authority=a.authority,
                                           tool_call=turn.tool_name))
                    self._log(qid, r, a, q.answer, prior=prior, orch_ans=orch_ans,
                              peer_maj=peer_maj, turn=turn)

                # stopping early once every agent agrees
                if len({a.answer for a in self.agents}) == 1:
                    break

            for row in self.trace:
                if row["question_id"] == qid and "rounds_used" not in row:
                    row["rounds_used"] = rounds_used

        return self

    def _log(self, qid, r, a, ground_truth, prior, orch_ans, peer_maj, turn):
        moved = prior is not None and a.answer != prior
        self.trace.append({
            "session_id": self.session_id,
            "topology": self.topology,
            "model": self.model_label,
            "question_id": qid,
            "round": r,
            "agent_id": a.id,
            "role": a.role,
            "ground_truth": ground_truth,
            "prior_answer": prior,
            "new_answer": a.answer,
            "changed": moved,
            "orchestrator_answer": orch_ans,
            "peer_majority": peer_maj,
            "moved_to_orchestrator": bool(moved and orch_ans is not None and a.answer == orch_ans),
            "moved_to_peer_majority": bool(moved and peer_maj is not None and a.answer == peer_maj),
            "correct": a.answer == ground_truth,
            "used_tool": turn.tool_name is not None,
            "tokens_in": turn.tokens_in,
            "tokens_out": turn.tokens_out,
        })

    def cost(self) -> float:
        return est_cost(self.model_label, self.tokens_in, self.tokens_out)
