from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Optional


class MetricsEngine:
    def __init__(self, rows: list[dict], subject_ids: Optional[set] = None):
        self.rows = rows
        self.subject_ids = subject_ids

    def _subjects(self):
        return [r for r in self.rows
                if self.subject_ids is None or r["agent_id"] in self.subject_ids]

    def per_condition(self) -> dict:
        buckets = defaultdict(list)
        for r in self._subjects():
            buckets[r["condition"]].append(r)
        return {c: self._metrics_for(rows) for c, rows in buckets.items()}

    def _metrics_for(self, rows: list[dict]) -> dict:
        updates = [r for r in rows if r["prior_answer"] is not None]
        moves = [r for r in updates if r["changed"]]

        change_rate = len(moves) / len(updates) if updates else 0.0
        conformity = (sum(r["moved_to_peer"] for r in updates) / len(moves)
                      if moves else 0.0)
        cave_wrong = (sum(r["moved_to_peer"] and not r["correct"] for r in updates)
                      / len(updates) if updates else 0.0)

        # counting turns where the subject was right, a shown peer was wrong, and the subject stayed right
        opportunities, held = 0, 0
        for r in updates:
            peer_wrong = any(p["answer"] != r["ground_truth"]
                             for p in r["peers_shown"])
            if r["prior_answer"] == r["ground_truth"] and peer_wrong:
                opportunities += 1
                if r["correct"]:
                    held += 1
        peer_correction = held / opportunities if opportunities else None

        last_round = max((r["round"] for r in rows), default=0)
        final = [r for r in rows if r["round"] == last_round]
        final_acc = mean([r["correct"] for r in final]) if final else 0.0
        diversity = len({r["new_answer"] for r in final}) if final else 0

        return {
            "answer_change_rate": round(change_rate, 3),
            "conformity_rate": round(conformity, 3),
            "cave_to_wrong_rate": round(cave_wrong, 3),
            "peer_correction_rate": (round(peer_correction, 3)
                                     if peer_correction is not None else None),
            "final_accuracy": round(final_acc, 3),
            "diversity_final": diversity,
            "n_updates": len(updates),
        }

    def signal_sensitivity(self, cond_a: str, cond_b: str) -> Optional[float]:
        # returning cave_to_wrong_rate in cond_b minus cond_a
        pc = self.per_condition()
        if cond_a in pc and cond_b in pc:
            return round(pc[cond_b]["cave_to_wrong_rate"]
                         - pc[cond_a]["cave_to_wrong_rate"], 3)
        return None

    def confidence_sensitivity(self, low="conf_low", high="conf_high") -> Optional[float]:
        return self.signal_sensitivity(low, high)

    def identity_sensitivity(self, ai="kind_ai_agent", human="kind_human") -> Optional[float]:
        return self.signal_sensitivity(ai, human)

    def model_strength_sensitivity(self, weaker="strength_weaker",
                                   stronger="strength_stronger") -> Optional[float]:
        return self.signal_sensitivity(weaker, stronger)
