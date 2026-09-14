from __future__ import annotations

import copy
import itertools
import os

import yaml

from .agents import build_agent
from .tasks import build_task
from .protocol import Orchestrator
from .tracing import Tracer
from .metrics import MetricsEngine

# mapping sweep labels to the condition names the sensitivity metrics read
_CONDITION_NAMES = [
    ("peer_confidence_low", "conf_low"),
    ("peer_confidence_high", "conf_high"),
    ("peer_kind_human", "kind_human"),
    ("peer_kind_ai_agent", "kind_ai_agent"),
    ("peer_model_strength_weaker", "strength_weaker"),
    ("peer_model_strength_similar", "strength_similar"),
    ("peer_model_strength_stronger", "strength_stronger"),
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ExperimentRunner:
    def __init__(self, config: dict):
        self.cfg = config
        self.name = config.get("experiment", "unnamed")
        self.seeds = config.get("seeds", [1])
        self.tracer = Tracer()

    def _conditions(self) -> list[tuple[str, dict]]:
        sweep = self.cfg.get("conditions", {}).get("sweep", {})
        if not sweep:
            return [("baseline", {})]
        keys = list(sweep.keys())
        cells = []
        for combo in itertools.product(*[sweep[k] for k in keys]):
            overrides = dict(zip(keys, combo))
            label = "__".join(f"{k}_{v}" for k, v in overrides.items())
            cells.append((label, overrides))
        return cells

    def _apply_overrides(self, agents_cfg: list[dict], overrides: dict) -> list[dict]:
        # applying sweep values to scripted agents and fanning them out for majority_size
        agents_cfg = copy.deepcopy(agents_cfg)
        out = []
        for spec in agents_cfg:
            if spec.get("type") == "scripted":
                if "peer_confidence" in overrides:
                    spec["confidence"] = {"low": 0.35, "high": 0.95}.get(
                        overrides["peer_confidence"], spec.get("confidence", 0.9))
                if "peer_correctness" in overrides:
                    spec["answer_policy"] = (
                        "correct" if overrides["peer_correctness"] == "correct"
                        else "incorrect")
                if "authority" in overrides:
                    spec["authority"] = float(overrides["authority"])
                if "peer_kind" in overrides:
                    spec["peer_kind"] = overrides["peer_kind"]
                if "peer_model_strength" in overrides:
                    spec["peer_model_strength"] = overrides["peer_model_strength"]
                n = int(overrides.get("majority_size", 1))
                for i in range(n):
                    clone = copy.deepcopy(spec)
                    clone["id"] = f"{spec['id']}_{i}" if n > 1 else spec["id"]
                    out.append(clone)
            else:
                out.append(spec)
        return out

    def run(self, out_dir: str = ".") -> dict:
        os.makedirs(out_dir, exist_ok=True)
        for label, overrides in self._conditions():
            cond = next((name for key, name in _CONDITION_NAMES if key in label), label)
            for seed in self.seeds:
                agents_cfg = self._apply_overrides(self.cfg["agents"], overrides)
                agents = [build_agent(a) for a in agents_cfg]
                task = build_task(self.cfg["task"])
                proto = self.cfg.get("protocol", {})
                orch = Orchestrator(
                    experiment=self.name,
                    agents=agents,
                    task=task,
                    rounds=proto.get("rounds", 3),
                    reveal=proto.get("reveal", "after_initial"),
                    topology=proto.get("topology", "full"),
                    tracer=self.tracer,
                )
                orch.run_cell(condition=cond, seed=seed)

        trace_path = os.path.join(out_dir, f"{self.name}.trace.jsonl")
        self.tracer.to_jsonl(trace_path)

        subject_ids = {a["id"] for a in self.cfg["agents"]
                       if a.get("type") != "scripted"}
        engine = MetricsEngine(self.tracer.rows(), subject_ids=subject_ids)
        return {
            "trace_path": trace_path,
            "per_condition": engine.per_condition(),
            "confidence_sensitivity": engine.confidence_sensitivity(),
            "identity_sensitivity": engine.identity_sensitivity(),
            "model_strength_sensitivity": engine.model_strength_sensitivity(),
            "n_events": len(self.tracer.events),
        }
