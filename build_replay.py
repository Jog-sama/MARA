#!/usr/bin/env python3
import json
import os

import yaml

from syco.env import load_env
load_env()

from syco.views import build_session, session_view, benchmark_view


def main():
    bench_cfg = yaml.safe_load(open("configs/session_benchmark.yaml"))
    hier_cfg = yaml.safe_load(open("configs/session_hierarchical.yaml"))

    bench = benchmark_view(bench_cfg, backend="mock")
    sess = build_session(hier_cfg, backend="mock")
    sess.run()
    view = session_view(sess)

    out = {"benchmark": bench, "session": view, "backend": "mock"}
    os.makedirs("ui", exist_ok=True)
    with open("ui/saved_run.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote ui/saved_run.json")


if __name__ == "__main__":
    main()
