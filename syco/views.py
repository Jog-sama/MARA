from __future__ import annotations

import copy

from .agents import build_agent
from .tasks import build_task
from .tools import build_registry
from .session import MASession
from . import session_metrics
from .models import est_cost


def _swap_backend(agents_cfg, backend, model):
    # leaving scripted agents and mock blocks as authored
    out = copy.deepcopy(agents_cfg)
    for a in out:
        if a.get("type") == "scripted":
            continue
        if backend == "openai":
            a["model"] = {"type": "openai", "model": model, "temperature": 0.7}
    return out


def build_session(cfg, topology=None, backend="mock", model="gpt-4o-mini"):
    topology = topology or cfg["topology"]
    agents_cfg = _swap_backend(cfg["agents"], backend, model)
    agents = [build_agent(a) for a in agents_cfg]
    task = build_task(cfg["task"])
    registry = build_registry(cfg.get("tools")) if cfg.get("tools") else None
    label = model if backend == "openai" else "mock"
    return MASession(session_id=cfg.get("name", "session"), topology=topology,
                     agents=agents, task=task, registry=registry,
                     rounds=cfg.get("rounds", 3), model_label=label)


def session_view(sess: MASession) -> dict:
    sess_rows = sess.trace
    tasks = sess.task.questions()
    questions = []

    for qi, q in enumerate(tasks):
        qrows = [r for r in sess_rows if r["question_id"] == qi]
        if not qrows:
            continue
        last = max(r["round"] for r in qrows)

        rounds = []
        for rnd in range(last + 1):
            turns = []
            for r in sorted([x for x in qrows if x["round"] == rnd],
                            key=lambda x: (x["role"] != "orchestrator", x["agent_id"])):
                turns.append({
                    "agent": r["agent_id"],
                    "role": r["role"],
                    "answer": r["new_answer"],
                    "correct": r["correct"],
                    "used_tool": r["used_tool"],
                    "deferred": r["moved_to_orchestrator"],
                    "conformed": r["moved_to_peer_majority"],
                    "changed": r["changed"],
                })
            rounds.append({"label": "independent" if rnd == 0 else f"round {rnd}",
                           "turns": turns})

        # judging misses and tool use on non-orchestrator agents only
        real_final = [r for r in qrows if r["round"] == last and r["role"] != "orchestrator"]
        group_correct = bool(real_final) and \
            sum(r["correct"] for r in real_final) >= (len(real_final) + 1) // 2
        used_tool = any(r["used_tool"] for r in qrows if r["role"] != "orchestrator")

        questions.append({
            "index": qi,
            "prompt": q.prompt,
            "correct": q.answer,
            "missed": not group_correct,
            "used_tool": used_tool,
            "rounds": rounds,
        })

    m = session_metrics.compute(sess_rows)
    cost = est_cost(sess.model_label, m.pop("tokens_in"), m.pop("tokens_out"))
    m["est_cost_usd"] = cost
    return {
        "topology": sess.topology,
        "model": sess.model_label,
        "questions": questions,
        "metrics": m,
    }


def benchmark_view(bench_cfg, backend="mock", model="gpt-4o-mini") -> dict:
    arms = []
    for arm in bench_cfg["arms"]:
        arm_cfg = {
            "name": arm["name"],
            "topology": arm["topology"],
            "agents": arm["agents"],
            "task": bench_cfg["task"],
            "tools": bench_cfg.get("tools"),
            "rounds": bench_cfg.get("rounds", 3),
        }
        sess = build_session(arm_cfg, backend=backend, model=model)
        sess.run()
        m = session_metrics.compute(sess.trace)
        cost = est_cost(sess.model_label, m.pop("tokens_in"), m.pop("tokens_out"))
        arms.append({
            "name": arm["name"],
            "topology": arm["topology"],
            "final_accuracy": m["final_accuracy"],
            "authority_deference": m["authority_deference"],
            "peer_conformity": m["peer_conformity"],
            "rounds_to_consensus": m["rounds_to_consensus"],
            "tool_use_rate": m["tool_use_rate"],
            "est_cost_usd": cost,
        })
    return {"task": bench_cfg["task"]["type"],
            "rounds": bench_cfg.get("rounds", 3), "arms": arms}
