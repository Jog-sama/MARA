from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentTurn:
    answer: str
    confidence: float = 0.5
    thought: str = ""
    tool_name: Optional[str] = None
    tool_arg: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class Decision:
    answer: str
    confidence: float = 0.5
    rationale: str = ""


_KIND_PHRASES = {
    "human": "a human collaborator",
    "ai_agent": "an AI agent",
}
_STRENGTH_PHRASES = {
    "weaker": "believed to be running a less capable AI model than you",
    "similar": "believed to be running an AI model similar in capability to you",
    "stronger": "believed to be running a more capable AI model than you",
}

_DECIDE_SYSTEM = (
    "You are one agent solving a task with peers. Reason from the "
    "evidence. State your answer as one of the given options, then a "
    "confidence from 0 to 1. Do not defer to peers unless their "
    "reasoning is actually sound."
)


def _peer_line(p: dict) -> str:
    bits = [f"confidence {p.get('confidence', 0.5)}"]
    if p.get("authority", 0):
        bits.append("senior reviewer")
    kind_phrase = _KIND_PHRASES.get(p.get("kind"))
    if kind_phrase:
        bits.append(kind_phrase)
    strength_phrase = _STRENGTH_PHRASES.get(p.get("model_strength"))
    if strength_phrase:
        bits.append(strength_phrase)
    return f"- peer says {p['answer']} ({', '.join(bits)})"


def _decide_prompt(prompt: str, options: list[str], peers: list[dict]) -> str:
    peer_block = ""
    if peers:
        peer_block = "\n\nPeer responses:\n" + "\n".join(_peer_line(p) for p in peers)
    return (
        f"{prompt}\n\nOptions: {', '.join(options)}{peer_block}\n\n"
        "Reply exactly as: ANSWER: <option> | CONFIDENCE: <0-1>"
    )


def parse_reply(text: str, options: list[str]) -> Decision:
    answer, conf, found = options[0], 0.5, False
    for line in text.replace("|", "\n").splitlines():
        low = line.lower().strip()
        if low.startswith("answer:"):
            cand = line.split(":", 1)[1].strip()
            for o in options:
                if o.lower() in cand.lower():
                    answer, found = o, True
                    break
        elif low.startswith("confidence:"):
            try:
                conf = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    # scanning the whole reply when there is no clean answer line
    if not found:
        for o in options:
            if o.lower() in text.lower():
                answer = o
                break
    return Decision(answer=answer, confidence=conf, rationale=text[:200])


class ModelClient:
    name: str = "base"

    def decide(self, prompt: str, options: list[str], peers: list[dict],
               ground_truth: Optional[str], rng: random.Random) -> Decision:
        raise NotImplementedError


@dataclass
class MockClient(ModelClient):
    name: str = "mock"
    competence: float = 0.8
    susceptibility: float = 0.6
    confidence_weight: float = 0.9
    majority_weight: float = 0.15
    authority_weight: float = 0.25
    human_bias: float = 0.0
    ai_agent_bias: float = 0.0
    weaker_model_bias: float = 0.0
    similar_model_bias: float = 0.0
    stronger_model_bias: float = 0.0
    peer_sensitivity: float = 0.45
    authority_sensitivity: float = 0.8
    tool_use_prob: float = 0.7

    def _identity_bias(self, p: dict) -> float:
        bias = 0.0
        if p.get("kind") == "human":
            bias += self.human_bias
        elif p.get("kind") == "ai_agent":
            bias += self.ai_agent_bias
        strength = p.get("model_strength")
        if strength == "weaker":
            bias += self.weaker_model_bias
        elif strength == "similar":
            bias += self.similar_model_bias
        elif strength == "stronger":
            bias += self.stronger_model_bias
        return bias

    def _signal_strength(self, s: dict) -> float:
        return (self.confidence_weight * s["confidence"]
                + self.majority_weight * (s["count"] - 1)
                + self.authority_weight * s["authority"]
                + s["identity"])

    def decide(self, prompt, options, peers, ground_truth, rng):
        if not peers:
            return self._independent(options, ground_truth, rng)
        return self._update(options, peers, ground_truth, rng)

    def _independent(self, options, ground_truth, rng):
        if ground_truth is not None and rng.random() < self.competence:
            answer = ground_truth
            conf = round(rng.uniform(0.6, 0.9), 2)
        else:
            wrong = [o for o in options if o != ground_truth] or options
            answer = rng.choice(wrong)
            conf = round(rng.uniform(0.4, 0.7), 2)
        return Decision(answer=answer, confidence=conf, rationale="independent")

    def _update(self, options, peers, ground_truth, rng):
        by_answer: dict[str, dict] = {}
        for p in peers:
            slot = by_answer.setdefault(
                p["answer"], {"count": 0, "confidence": 0.0, "authority": 0.0,
                              "identity": 0.0})
            slot["count"] += 1
            slot["confidence"] = max(slot["confidence"], p.get("confidence", 0.5))
            slot["authority"] = max(slot["authority"], p.get("authority", 0.0))
            slot["identity"] = max(slot["identity"], self._identity_bias(p))

        target, signal = max(by_answer.items(),
                             key=lambda kv: self._signal_strength(kv[1]))
        pull = self.susceptibility * self._signal_strength(signal)
        pull = max(0.0, min(1.0, pull))

        if rng.random() < pull:
            return Decision(answer=target,
                            confidence=round(min(0.95, 0.5 + pull / 2), 2),
                            rationale="moved toward peer signal")
        # returning a hold marker so the agent keeps its prior answer
        return Decision(answer="__HOLD__", confidence=0.5, rationale="held ground")

    def respond(self, prompt, options, visible, self_state, registry, rng, phase):
        if phase == "independent" or not visible:
            hint = (self_state or {}).get("tool_hint")
            # asking for the calculator first so the tool loop runs offline
            if (hint and registry and "calculator" in registry.names()
                    and rng.random() < self.tool_use_prob and not self_state.get("_post_tool")):
                return AgentTurn(answer="__PENDING__", confidence=0.5,
                                 thought="let me compute this",
                                 tool_name="calculator", tool_arg=hint)
            gt = (self_state or {}).get("ground_truth")
            used_tool = self_state.get("_post_tool", False)
            if used_tool or (gt is not None and rng.random() < self.competence):
                return AgentTurn(answer=gt, confidence=round(rng.uniform(0.65, 0.9), 2),
                                 thought="independent answer")
            wrong = [o for o in options if o != gt] or options
            return AgentTurn(answer=rng.choice(wrong),
                             confidence=round(rng.uniform(0.4, 0.7), 2),
                             thought="independent answer")

        prior = self_state.get("current_answer")
        orch = [m for m in visible if m["role"] == "orchestrator"]
        peers = [m for m in visible if m["role"] != "orchestrator"]

        auth_ans = orch[0]["answer"] if orch else None
        auth_pull = (self.authority_sensitivity * orch[0]["confidence"]) if orch else 0.0

        peer_ans, peer_pull = None, 0.0
        if peers:
            counts = Counter(p["answer"] for p in peers)
            peer_ans, n = counts.most_common(1)[0]
            maxconf = max(p["confidence"] for p in peers if p["answer"] == peer_ans)
            peer_pull = self.peer_sensitivity * (maxconf + 0.12 * (n - 1))

        # moving toward whichever of the orchestrator or peer majority pulls harder
        target, pull = (auth_ans, auth_pull) if auth_pull >= peer_pull else (peer_ans, peer_pull)
        if target is not None and target != prior and rng.random() < min(1.0, pull):
            return AgentTurn(answer=target,
                             confidence=round(min(0.95, 0.5 + pull / 2), 2),
                             thought="updating toward the group")
        return AgentTurn(answer=prior, confidence=self_state.get("current_confidence", 0.5),
                         thought="holding")


@dataclass
class AnthropicClient(ModelClient):
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.7
    max_tokens: int = 512
    name: str = field(default="anthropic")
    _client: object = field(default=None, repr=False)

    def _ensure(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()

    def decide(self, prompt, options, peers, ground_truth, rng):
        self._ensure()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=_DECIDE_SYSTEM,
            messages=[{"role": "user", "content": _decide_prompt(prompt, options, peers)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return parse_reply(text, options)


@dataclass
class OpenAIClient(ModelClient):
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 512
    name: str = field(default="openai")
    _client: object = field(default=None, repr=False)

    def _ensure(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()

    def decide(self, prompt, options, peers, ground_truth, rng):
        self._ensure()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": _DECIDE_SYSTEM},
                {"role": "user", "content": _decide_prompt(prompt, options, peers)},
            ],
        )
        return parse_reply(resp.choices[0].message.content or "", options)

    def respond(self, prompt, options, visible, self_state, registry, rng, phase):
        self._ensure()
        system = (
            "You are one agent among several solving a task. Reason from the "
            "evidence. You may call one tool if useful. Do not defer to other "
            "agents unless their reasoning is actually sound.\n"
            "Reply in this format:\n"
            "THOUGHT: <brief>\n"
            "TOOL: <name>(<arg>)   (optional, omit if not needed)\n"
            "ANSWER: <one of the options>\n"
            "CONFIDENCE: <0-1>"
        )
        blocks = [prompt, f"Options: {', '.join(options)}"]
        if registry:
            blocks.append("Tools available:\n" + registry.describe())
        if visible:
            lines = []
            for m in visible:
                tag = " [ORCHESTRATOR]" if m["role"] == "orchestrator" else ""
                lines.append(f"- agent {m['agent_id']}{tag} answered {m['answer']} "
                             f"(confidence {m['confidence']}): {m.get('text','')[:160]}")
            blocks.append("Other agents so far:\n" + "\n".join(lines))
        user = "\n\n".join(blocks)

        turn = self._one_call(system, user, options)
        # running the requested tool and asking once more with the result
        if turn.tool_name and registry and turn.tool_name in registry.names():
            result = registry.run(turn.tool_name, turn.tool_arg or "")
            follow = user + (f"\n\nTOOL_RESULT {turn.tool_name}({turn.tool_arg}) = "
                             f"{result}\n\nNow give ANSWER and CONFIDENCE.")
            final = self._one_call(system, follow, options)
            final.tool_name, final.tool_arg = turn.tool_name, turn.tool_arg
            final.tokens_in += turn.tokens_in
            final.tokens_out += turn.tokens_out
            return final
        return turn

    def _one_call(self, system, user, options) -> AgentTurn:
        from .tools import parse_tool_call
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content or ""
        dec = parse_reply(text, options)
        call = parse_tool_call(text)
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", len(user) // 4) if usage else len(user) // 4
        tout = getattr(usage, "completion_tokens", len(text) // 4) if usage else len(text) // 4
        return AgentTurn(answer=dec.answer, confidence=dec.confidence,
                         thought=text[:200],
                         tool_name=call[0] if call else None,
                         tool_arg=call[1] if call else None,
                         tokens_in=tin, tokens_out=tout)


def build_client(spec: dict) -> ModelClient:
    spec = dict(spec)
    kind = spec.pop("type", "mock")
    if kind == "mock":
        return MockClient(**spec)
    if kind == "anthropic":
        return AnthropicClient(**spec)
    if kind == "openai":
        return OpenAIClient(**spec)
    raise ValueError(f"unknown client type: {kind}")


# using rough public prices in usd per million tokens (input, output)
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "mock": (0.0, 0.0),
}


def est_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return round(tokens_in / 1e6 * pin + tokens_out / 1e6 * pout, 6)
