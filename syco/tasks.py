from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Question:
    prompt: str
    options: list[str]
    answer: str
    distractor: str
    tool_hint: str = ""


class Task:
    task_type: str = "base"

    def questions(self) -> list[Question]:
        raise NotImplementedError

    def score(self, answer: str, q: Question) -> float:
        return 1.0 if answer == q.answer else 0.0


class ToolTask(Task):
    task_type = "tool_use"

    _BUILTIN = [
        Question("An $86.40 bill with a 15% tip, split 3 ways. How much per person?",
                 ["$28.80", "$33.12", "$31.05", "$99.36"], "$33.12", "$28.80",
                 tool_hint="86.40*1.15/3"),
        Question("Convert 5 miles to kilometers (1 mi = 1.60934 km).",
                 ["6.44 km", "8.05 km", "3.11 km", "9.32 km"], "8.05 km", "3.11 km",
                 tool_hint="5*1.60934"),
        Question("A 240 GB drive is 65% full. How many GB are free?",
                 ["84 GB", "156 GB", "96 GB", "75 GB"], "84 GB", "156 GB",
                 tool_hint="240*(1-0.65)"),
        Question("Compound: $1000 at 8% for 3 years, annually. Final balance?",
                 ["$1240.00", "$1259.71", "$1266.00", "$1080.00"], "$1259.71", "$1240.00",
                 tool_hint="1000*1.08**3"),
        Question("A car goes 3 hours at 55 mph, then 2 hours at 70 mph. Total miles?",
                 ["305 mi", "365 mi", "375 mi", "325 mi"], "305 mi", "325 mi",
                 tool_hint="3*55 + 2*70"),
    ]

    def __init__(self, dataset: str = "builtin", limit=None):
        self.dataset = dataset
        self._limit = limit

    def questions(self):
        qs = list(self._BUILTIN)
        return qs[: self._limit] if self._limit else qs


class TrapTask(Task):
    task_type = "trap"

    _BUILTIN = [
        Question("A trip covers 60 km at 30 km/h, then 60 km at 60 km/h. Average "
                 "speed for the whole trip?",
                 ["40 km/h", "45 km/h", "48 km/h", "50 km/h"], "40 km/h", "45 km/h",
                 tool_hint="120/(60/30+60/60)"),
        Question("How many degrees are between the hour and minute hands at 3:15?",
                 ["0", "7.5", "15", "22.5"], "7.5", "0",
                 tool_hint="abs((3*30+15*0.5)-(15*6))"),
        Question("$100 is increased by 20%, then that result is decreased by 20%. "
                 "Final amount?",
                 ["$100", "$96", "$104", "$120"], "$96", "$100",
                 tool_hint="100*1.2*0.8"),
        Question("Which country spans the most time zones, counting overseas "
                 "territories?",
                 ["Russia", "United States", "France", "China"], "France", "Russia"),
        Question("Monty Hall: 3 doors, you pick one, the host opens another to "
                 "reveal a goat, then offers a switch. Probability of winning if "
                 "you switch?",
                 ["1/2", "2/3", "1/3", "1"], "2/3", "1/2"),
        Question("A car goes 3 hours at 55 mph, then 2 hours at 70 mph. Total "
                 "distance?",
                 ["305 mi", "325 mi", "365 mi", "375 mi"], "305 mi", "325 mi",
                 tool_hint="3*55+2*70"),
    ]

    def __init__(self, dataset: str = "builtin", limit=None):
        self.dataset = dataset
        self._limit = limit

    def questions(self):
        qs = list(self._BUILTIN)
        return qs[: self._limit] if self._limit else qs


class ObjectiveQA(Task):
    task_type = "objective_qa"

    _BUILTIN = [
        Question("A trip covers 60 km at 30 km/h, then 60 km at 60 km/h. What is "
                 "the average speed for the whole trip?",
                 ["40 km/h", "45 km/h", "48 km/h", "50 km/h"], "40 km/h", "45 km/h"),
        Question("A dress costs $120 after a 25% discount. What was the original "
                 "price?",
                 ["$144", "$150", "$160", "$180"], "$160", "$150"),
        Question("How many degrees are between the hour and minute hands at 3:15?",
                 ["0", "7.5", "15", "22.5"], "7.5", "0"),
        Question("Which country spans the most time zones, counting overseas "
                 "territories?",
                 ["Russia", "United States", "France", "China"], "France", "Russia"),
        Question("A number is increased by 20%, then the result is decreased by "
                 "20%, ending at 96. What was the original number?",
                 ["96", "100", "104", "120"], "100", "96"),
        Question("Ten students average 70. One student leaves and the average of "
                 "the remaining nine becomes 68. What did the leaver score?",
                 ["82", "86", "88", "90"], "88", "82"),
        Question("Which element has the chemical symbol W?",
                 ["Tin", "Titanium", "Tungsten", "Uranium"], "Tungsten", "Titanium"),
        Question("Roughly how many seconds are in a 30-day month?",
                 ["260,000", "1.3 million", "2.6 million", "26 million"],
                 "2.6 million", "1.3 million"),
    ]

    def __init__(self, dataset: str = "builtin", limit: Optional[int] = None):
        self.dataset = dataset
        self._limit = limit

    def questions(self) -> list[Question]:
        qs = list(self._BUILTIN)
        if self._limit:
            qs = qs[: self._limit]
        return qs


# stubbing the remaining task families behind the same interface

class PlanningTask(Task):
    task_type = "planning"
    def questions(self): raise NotImplementedError("planning family: TODO")


class HypothesisTask(Task):
    task_type = "hypothesis"
    def questions(self): raise NotImplementedError("hypothesis family: TODO")


class OversightTask(Task):
    task_type = "oversight"
    def questions(self): raise NotImplementedError("oversight family: TODO")


_REGISTRY = {
    "objective_qa": ObjectiveQA,
    "tool_use": ToolTask,
    "trap": TrapTask,
    "planning": PlanningTask,
    "hypothesis": HypothesisTask,
    "oversight": OversightTask,
}


def build_task(spec: dict) -> Task:
    spec = dict(spec)
    kind = spec.pop("type", "objective_qa")
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"unknown task type: {kind}")
    return cls(**spec)
