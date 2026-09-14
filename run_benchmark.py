#!/usr/bin/env python3
import os
import sys

import yaml

from syco.env import load_env
load_env()

from syco.views import benchmark_view


def _cell(v):
    return "   -  ".rjust(22) if v is None else f"{v:>22}"


def main(path):
    cfg = yaml.safe_load(open(path))
    backend = os.environ.get("SYCO_BACKEND", "mock")
    model = os.environ.get("SYCO_MODEL", "gpt-4o-mini")
    b = benchmark_view(cfg, backend=backend, model=model)

    cols = ["final_accuracy", "authority_deference", "peer_conformity",
            "rounds_to_consensus", "tool_use_rate", "est_cost_usd"]
    print(f"\ntask: {b['task']}   rounds: {b['rounds']}   backend: {backend}\n")
    header = "arm".ljust(22) + "".join(c[:20].rjust(22) for c in cols)
    print(header)
    print("-" * len(header))
    for a in b["arms"]:
        print(a["name"].ljust(22) + "".join(_cell(a[c]) for c in cols))
    print("\nH1: authority_deference is hierarchical-only.  "
          "H3: deference vs conformity are separate axes.  "
          "H5: rounds and cost vs accuracy.\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/session_benchmark.yaml")
