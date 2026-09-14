#!/usr/bin/env python3
import os
import sys

import yaml

from syco.env import load_env
load_env()

from syco.views import build_session, session_view


def main(path):
    cfg = yaml.safe_load(open(path))
    backend = os.environ.get("SYCO_BACKEND", "mock")
    model = os.environ.get("SYCO_MODEL", "gpt-4o-mini")

    sess = build_session(cfg, backend=backend, model=model)
    sess.run()
    view = session_view(sess)

    print(f"\ntopology: {view['topology']}   backend: {backend}   model: {view['model']}")
    for q in view["questions"]:
        flag = "MISSED" if q["missed"] else "solved"
        print(f"\n=== Q{q['index']+1} [{flag}] correct: {q['correct']} ===")
        print(q["prompt"])
        for rd in q["rounds"]:
            print(f"[{rd['label']}]")
            for t in rd["turns"]:
                who = "BOSS" if t["role"] == "orchestrator" else t["agent"]
                tool = " (used tool)" if t["used_tool"] else ""
                mark = "  <- deferred to boss" if t["deferred"] else (
                       "  <- followed peers" if t["conformed"] else "")
                print(f"  {who:>5}: {t['answer']}{tool}{mark}")

    m = view["metrics"]
    print("metrics across all questions:")
    for k, v in m.items():
        print(f"  {k}: {v}")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/session_hierarchical.yaml")
